"""Data access object for submission operations."""

import logging

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..core.protocols import SubmissionStatus
from .models import SubmissionModel

logger = logging.getLogger(__name__)


class SubmissionDAO:
    """Handles all submission-related database operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save_submission(self, submission: SubmissionModel) -> None:
        """Save a new submission."""
        async with self._session_factory() as session:
            session.add(submission)
            await session.commit()

    async def get_submission(self, submission_id: str) -> SubmissionModel | None:
        """Get submission by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            return result.scalar_one_or_none()

    async def update_submission_status(
        self,
        submission_id: str,
        status: SubmissionStatus,
        error_message: str | None = None,
    ) -> None:
        """Update submission status."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.status = status
                if error_message:
                    submission.error_message = error_message
                await session.commit()

    async def update_submission_score(self, submission_id: str, final_score: float) -> None:
        """Update submission final score."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.final_score = final_score
                submission.status = SubmissionStatus.FINISHED
                await session.commit()

    async def update_submission_code(self, submission_id: str, code: str) -> None:
        """Store miner's code content after evaluation.

        This is called by the validator after downloading and evaluating
        the miner's code, so it can be displayed on the dashboard.

        Args:
            submission_id: Submission ID
            code: Miner's train.py code content
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.code_content = code
                await session.commit()

    async def get_submission_code(self, submission_id: str) -> str | None:
        """Get miner's code content for a submission.

        Args:
            submission_id: Submission ID

        Returns:
            Code content or None if not stored
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(SubmissionModel.code_content).where(
                    SubmissionModel.submission_id == submission_id
                )
            )
            return result.scalar_one_or_none()

    async def get_all_submissions(self) -> list[SubmissionModel]:
        """Get all submissions."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).order_by(desc(SubmissionModel.created_at))
            )
            return list(result.scalars().all())

    async def get_pending_submissions(
        self, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get submissions pending validation.

        Args:
            spec_version: If provided, only return submissions matching this version
        """
        async with self._session_factory() as session:
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.PENDING
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(query.order_by(SubmissionModel.created_at))
            return list(result.scalars().all())

    async def get_evaluating_submissions(
        self, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get submissions currently being evaluated.

        Args:
            spec_version: If provided, only return submissions matching this version
        """
        async with self._session_factory() as session:
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.EVALUATING
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(query.order_by(SubmissionModel.created_at))
            return list(result.scalars().all())

    async def get_latest_submission_by_hotkey(self, hotkey: str) -> SubmissionModel | None:
        """Get the most recent submission from a miner.

        Used for rate limiting - checking when miner last submitted.

        Args:
            hotkey: Miner's hotkey

        Returns:
            Most recent submission or None
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(SubmissionModel.miner_hotkey == hotkey)
                .order_by(desc(SubmissionModel.created_at))
                .limit(1)
            )
            return result.scalar_one_or_none()

    # Leaderboard operations

    async def get_top_submission(self, spec_version: int | None = None) -> SubmissionModel | None:
        """Get the top-scoring finished submission (raw, no threshold).

        Args:
            spec_version: If provided, only consider submissions from this version
        """
        async with self._session_factory() as session:
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(
                query.order_by(desc(SubmissionModel.final_score)).limit(1)
            )
            return result.scalar_one_or_none()

    async def get_leaderboard_winner(
        self,
        threshold: float = 0.01,
        spec_version: int | None = None,
    ) -> SubmissionModel | None:
        """Get the rank 1 submission from leaderboard with threshold.

        A new submission only beats an incumbent if it's more than `threshold`
        better. This gives stability to leaders.

        Args:
            threshold: Minimum improvement ratio to beat incumbent
            spec_version: If provided, only consider submissions from this version

        Returns:
            The submission at rank 1, or None if no finished submissions.
        """
        async with self._session_factory() as session:
            # Get all finished submissions ordered by created_at (oldest first)
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(query.order_by(SubmissionModel.created_at.asc()))
            submissions = list(result.scalars().all())

            if not submissions:
                return None

            # Build leaderboard with threshold logic
            leaderboard: list[SubmissionModel] = []

            for submission in submissions:
                score = submission.final_score or 0.0

                # Find insertion position
                insert_pos = len(leaderboard)
                for i, existing in enumerate(leaderboard):
                    existing_score = existing.final_score or 0.0
                    threshold_score = existing_score * (1 + threshold)
                    if score > threshold_score:
                        insert_pos = i
                        break

                leaderboard.insert(insert_pos, submission)

            return leaderboard[0] if leaderboard else None

    async def get_leaderboard(
        self,
        limit: int = 100,
        spec_version: int | None = None,
        threshold: float = 0.01,
    ) -> list[SubmissionModel]:
        """Get leaderboard with threshold winner at #1, rest sorted by raw MFU.

        Position #1: Threshold-adjusted winner (gets emissions)
        Positions #2+: All others sorted by raw MFU descending

        Args:
            limit: Maximum number of submissions to return
            spec_version: If provided, only show submissions from this version
            threshold: Adaptive threshold for determining #1
        """
        # Get threshold winner (position #1)
        winner = await self.get_leaderboard_winner(
            threshold=threshold,
            spec_version=spec_version,
        )

        async with self._session_factory() as session:
            # Get all finished submissions sorted by raw score
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(
                query.order_by(desc(SubmissionModel.final_score)).limit(limit + 1)
            )
            all_submissions = list(result.scalars().all())

        # Build leaderboard: winner first, then others by raw score
        leaderboard: list[SubmissionModel] = []

        if winner:
            leaderboard.append(winner)
            # Add remaining submissions (excluding winner) sorted by raw score
            for sub in all_submissions:
                if sub.submission_id != winner.submission_id:
                    leaderboard.append(sub)
                    if len(leaderboard) >= limit:
                        break
        else:
            # No winner, just return raw sorted
            leaderboard = all_submissions[:limit]

        return leaderboard

    async def get_top_submissions(
        self, limit: int = 5, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get top N submissions by score for similarity checking.

        Args:
            limit: Number of top submissions to return (default 5)
            spec_version: If provided, only consider submissions from this version

        Returns:
            List of top submissions sorted by score (highest first)
        """
        async with self._session_factory() as session:
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(
                query.order_by(desc(SubmissionModel.final_score)).limit(limit)
            )
            return list(result.scalars().all())
