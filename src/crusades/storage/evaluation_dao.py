"""Data access object for evaluation operations."""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import EvaluationModel

logger = logging.getLogger(__name__)


class EvaluationDAO:
    """Handles all evaluation-related database operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save_evaluation(self, evaluation: EvaluationModel) -> None:
        """Save a new evaluation result."""
        async with self._session_factory() as session:
            session.add(evaluation)
            await session.commit()

    async def get_evaluations(self, submission_id: str) -> list[EvaluationModel]:
        """Get all evaluations for a submission."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(EvaluationModel)
                .where(EvaluationModel.submission_id == submission_id)
                .order_by(EvaluationModel.created_at)
            )
            return list(result.scalars().all())

    async def count_evaluations(self, submission_id: str) -> int:
        """Count ALL evaluations for a submission (including failed)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(EvaluationModel)
                .where(
                    EvaluationModel.submission_id == submission_id,
                )
            )
            return result.scalar() or 0
