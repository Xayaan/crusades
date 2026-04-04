"""Shared test fixtures for crusades test suite."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from crusades.core.protocols import SubmissionStatus
from crusades.storage.database import Database
from crusades.storage.models import (
    EvaluationModel,
    SubmissionModel,
)


@pytest_asyncio.fixture
async def db():
    """Create an in-memory async Database instance with tables initialized."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.initialize()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def seeded_db(db: Database):
    """Database seeded with submissions and evaluations in various states."""
    async with db.session_factory() as session:
        now = datetime.now(UTC)

        # Finished submission with evaluations (high score)
        sub1 = SubmissionModel(
            submission_id="sub-finished-1",
            miner_hotkey="hotkey_alice",
            miner_uid=1,
            code_hash="hash_abc",
            bucket_path="https://example.com/train1.py",
            spec_version=19,
            status=SubmissionStatus.FINISHED,
            final_score=65.5,
            code_content="def inner_steps(): pass",
            created_at=now - timedelta(hours=2),
        )
        session.add(sub1)

        eval1 = EvaluationModel(
            evaluation_id=str(uuid.uuid4()),
            submission_id="sub-finished-1",
            evaluator_hotkey="validator_1",
            mfu=65.5,
            tokens_per_second=1200.0,
            total_tokens=24000,
            wall_time_seconds=20.0,
            success=True,
            created_at=now - timedelta(hours=2),
        )
        session.add(eval1)

        # Finished submission (lower score, older)
        sub2 = SubmissionModel(
            submission_id="sub-finished-2",
            miner_hotkey="hotkey_bob",
            miner_uid=2,
            code_hash="hash_def",
            bucket_path="https://example.com/train2.py",
            spec_version=19,
            status=SubmissionStatus.FINISHED,
            final_score=55.0,
            created_at=now - timedelta(hours=5),
        )
        session.add(sub2)

        eval2 = EvaluationModel(
            evaluation_id=str(uuid.uuid4()),
            submission_id="sub-finished-2",
            evaluator_hotkey="validator_1",
            mfu=55.0,
            tokens_per_second=1000.0,
            total_tokens=20000,
            wall_time_seconds=20.0,
            success=True,
            created_at=now - timedelta(hours=5),
        )
        session.add(eval2)

        # Pending submission
        sub3 = SubmissionModel(
            submission_id="sub-pending-1",
            miner_hotkey="hotkey_carol",
            miner_uid=3,
            code_hash="hash_ghi",
            bucket_path="https://example.com/train3.py",
            spec_version=19,
            status=SubmissionStatus.PENDING,
            created_at=now - timedelta(minutes=10),
        )
        session.add(sub3)

        # Failed submission
        sub4 = SubmissionModel(
            submission_id="sub-failed-1",
            miner_hotkey="hotkey_dave",
            miner_uid=4,
            code_hash="hash_jkl",
            bucket_path="https://example.com/train4.py",
            spec_version=19,
            status=SubmissionStatus.FAILED_VALIDATION,
            error_message="SYNTAX_ERROR: invalid syntax",
            created_at=now - timedelta(hours=1),
        )
        session.add(sub4)

        await session.commit()

    return db


@pytest.fixture
def mock_db_client():
    """Return a MockClient for API tests that don't need real DB."""
    from crusades.tui.client import MockClient

    return MockClient()
