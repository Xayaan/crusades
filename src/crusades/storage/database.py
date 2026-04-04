"""Database abstraction layer."""

import logging

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config import get_hparams
from ..core.protocols import SubmissionStatus
from .evaluation_dao import EvaluationDAO
from .models import (
    Base,
    EvaluationModel,
    SubmissionModel,
    VerifiedPaymentModel,
)
from .payment_dao import PaymentDAO
from .state_dao import StateDAO
from .submission_dao import SubmissionDAO

logger = logging.getLogger(__name__)


def _migrate_missing_columns(connection) -> None:
    """Add any columns defined in models but missing from the database.

    SQLAlchemy's create_all only creates new tables; it won't add columns to
    existing ones.  This lightweight migration inspects each table and issues
    ALTER TABLE ADD COLUMN for anything missing, using the column's default
    or NULL if no server_default is set.  Safe to run repeatedly.
    """
    inspector = inspect(connection)
    existing_tables = inspector.get_table_names()

    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue

        existing_cols = {col["name"] for col in inspector.get_columns(table.name)}

        for column in table.columns:
            if column.name in existing_cols:
                continue

            col_type = column.type.compile(dialect=connection.dialect)
            nullable = "NULL" if column.nullable else "NOT NULL"

            default_clause = ""
            if column.server_default is not None:
                default_clause = f" DEFAULT {column.server_default.arg}"
            elif not column.nullable and column.default is not None:
                if hasattr(column.default, "arg") and isinstance(
                    column.default.arg, (int, float, str, bool)
                ):
                    val = column.default.arg
                    if isinstance(val, bool):
                        val = 1 if val else 0
                    elif isinstance(val, str):
                        val = f"'{val}'"
                    default_clause = f" DEFAULT {val}"
                else:
                    nullable = "NULL"

            stmt = (
                f"ALTER TABLE {table.name} "
                f"ADD COLUMN {column.name} "
                f"{col_type} {nullable}{default_clause}"
            )
            logger.info(f"Migration: adding column {table.name}.{column.name} ({col_type})")
            connection.execute(text(stmt))


class Database:
    """Async database interface.

    Delegates to focused DAO classes while preserving backward-compatible
    method signatures on the Database object itself.
    """

    def __init__(self, url: str | None = None):
        if url is None:
            url = get_hparams().storage.database_url
        self.engine = create_async_engine(url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # DAO instances
        self.submissions = SubmissionDAO(self.session_factory)
        self.evaluations = EvaluationDAO(self.session_factory)
        self.state = StateDAO(self.session_factory)
        self.payments = PaymentDAO(self.session_factory)

    async def initialize(self) -> None:
        """Create tables if they don't exist, then add any missing columns."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(_migrate_missing_columns)

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()

    # ------------------------------------------------------------------
    # Backward-compatible pass-through methods
    # ------------------------------------------------------------------

    # Submission operations

    async def save_submission(self, submission: SubmissionModel) -> None:
        """Save a new submission."""
        await self.submissions.save_submission(submission)

    async def get_submission(self, submission_id: str) -> SubmissionModel | None:
        """Get submission by ID."""
        return await self.submissions.get_submission(submission_id)

    async def update_submission_status(
        self,
        submission_id: str,
        status: SubmissionStatus,
        error_message: str | None = None,
    ) -> None:
        """Update submission status."""
        await self.submissions.update_submission_status(submission_id, status, error_message)

    async def update_submission_score(self, submission_id: str, final_score: float) -> None:
        """Update submission final score."""
        await self.submissions.update_submission_score(submission_id, final_score)

    async def update_submission_code(self, submission_id: str, code: str) -> None:
        """Store miner's code content after evaluation."""
        await self.submissions.update_submission_code(submission_id, code)

    async def get_submission_code(self, submission_id: str) -> str | None:
        """Get miner's code content for a submission."""
        return await self.submissions.get_submission_code(submission_id)

    async def get_all_submissions(self) -> list[SubmissionModel]:
        """Get all submissions."""
        return await self.submissions.get_all_submissions()

    async def get_pending_submissions(
        self, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get submissions pending validation."""
        return await self.submissions.get_pending_submissions(spec_version)

    async def get_evaluating_submissions(
        self, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get submissions currently being evaluated."""
        return await self.submissions.get_evaluating_submissions(spec_version)

    async def get_latest_submission_by_hotkey(self, hotkey: str) -> SubmissionModel | None:
        """Get the most recent submission from a miner."""
        return await self.submissions.get_latest_submission_by_hotkey(hotkey)

    # Leaderboard operations

    async def get_top_submission(self, spec_version: int | None = None) -> SubmissionModel | None:
        """Get the top-scoring finished submission (raw, no threshold)."""
        return await self.submissions.get_top_submission(spec_version)

    async def get_leaderboard_winner(
        self,
        threshold: float = 0.01,
        spec_version: int | None = None,
    ) -> SubmissionModel | None:
        """Get the rank 1 submission from leaderboard with threshold."""
        return await self.submissions.get_leaderboard_winner(threshold, spec_version)

    async def get_leaderboard(
        self,
        limit: int = 100,
        spec_version: int | None = None,
        threshold: float = 0.01,
    ) -> list[SubmissionModel]:
        """Get leaderboard with threshold winner at #1, rest sorted by raw MFU."""
        return await self.submissions.get_leaderboard(limit, spec_version, threshold)

    async def get_top_submissions(
        self, limit: int = 5, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get top N submissions by score for similarity checking."""
        return await self.submissions.get_top_submissions(limit, spec_version)

    # Evaluation operations

    async def save_evaluation(self, evaluation: EvaluationModel) -> None:
        """Save a new evaluation result."""
        await self.evaluations.save_evaluation(evaluation)

    async def get_evaluations(self, submission_id: str) -> list[EvaluationModel]:
        """Get all evaluations for a submission."""
        return await self.evaluations.get_evaluations(submission_id)

    async def count_evaluations(self, submission_id: str) -> int:
        """Count ALL evaluations for a submission (including failed)."""
        return await self.evaluations.count_evaluations(submission_id)

    # Validator state operations

    async def get_validator_state(self, key: str) -> str | None:
        """Get a validator state value."""
        return await self.state.get_validator_state(key)

    async def set_validator_state(self, key: str, value: str) -> None:
        """Set a validator state value (upsert)."""
        await self.state.set_validator_state(key, value)

    # Payment verification operations

    async def is_payment_used(self, block_hash: str, extrinsic_index: int) -> bool:
        """Check if a payment extrinsic has already been claimed by a submission."""
        return await self.payments.is_payment_used(block_hash, extrinsic_index)

    async def record_verified_payment(
        self,
        submission_id: str,
        miner_hotkey: str,
        miner_coldkey: str,
        block_hash: str,
        extrinsic_index: int,
        amount_rao: int,
    ) -> None:
        """Record a verified payment to prevent double-spend."""
        await self.payments.record_verified_payment(
            submission_id, miner_hotkey, miner_coldkey, block_hash, extrinsic_index, amount_rao
        )

    async def get_payment_for_submission(self, submission_id: str) -> VerifiedPaymentModel | None:
        """Get the payment record for a submission."""
        return await self.payments.get_payment_for_submission(submission_id)

    # Adaptive threshold operations

    async def get_adaptive_threshold(
        self,
        current_block: int,
        base_threshold: float = 0.01,
        decay_percent: float = 0.05,
        decay_interval_blocks: int = 100,
    ) -> float:
        """Get the current adaptive threshold with decay applied."""
        return await self.state.get_adaptive_threshold(
            current_block, base_threshold, decay_percent, decay_interval_blocks
        )

    async def update_adaptive_threshold(
        self,
        new_score: float,
        old_score: float,
        current_block: int,
        base_threshold: float = 0.01,
    ) -> float:
        """Update threshold when a new leader is established."""
        return await self.state.update_adaptive_threshold(
            new_score, old_score, current_block, base_threshold
        )


# Global instance
_database: Database | None = None


async def get_database() -> Database:
    """Get or create global database instance."""
    global _database
    if _database is None:
        _database = Database()
        await _database.initialize()
    return _database
