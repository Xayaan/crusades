"""Tests for SQLAlchemy ORM models and SubmissionStatus enum.

Note: SQLAlchemy `default` callables only fire on flush/commit, not on
construction. Tests that need defaults use the database fixture.
"""

import uuid

from crusades.core.protocols import SubmissionStatus
from crusades.storage.database import Database
from crusades.storage.models import (
    Base,
    EvaluationModel,
    SubmissionModel,
)


class TestSubmissionStatus:
    """SubmissionStatus enum values match expected strings."""

    def test_all_statuses_defined(self):
        expected = {
            "pending", "validating", "evaluating",
            "finished", "failed_validation", "failed_evaluation", "error",
        }
        actual = {s.value for s in SubmissionStatus}
        assert actual == expected

    def test_str_enum_behavior(self):
        assert str(SubmissionStatus.PENDING) == "pending"
        assert SubmissionStatus.FINISHED == "finished"


class TestSubmissionModelViaDB:
    """Submission model defaults verified through DB round-trip."""

    async def test_default_status_is_pending(self, db: Database):
        sub = SubmissionModel(
            submission_id="defaults-test",
            miner_hotkey="hk",
            miner_uid=1,
            code_hash="abc",
            bucket_path="https://example.com/train.py",
            spec_version=19,
        )
        await db.save_submission(sub)
        retrieved = await db.get_submission("defaults-test")
        assert retrieved.status == SubmissionStatus.PENDING

    async def test_default_payment_verified_false(self, db: Database):
        sub = SubmissionModel(
            submission_id="pay-defaults",
            miner_hotkey="hk",
            miner_uid=1,
            code_hash="abc",
            bucket_path="https://example.com/train.py",
            spec_version=19,
        )
        await db.save_submission(sub)
        retrieved = await db.get_submission("pay-defaults")
        assert retrieved.payment_verified is False

    async def test_optional_fields_are_none(self, db: Database):
        sub = SubmissionModel(
            submission_id="opt-fields",
            miner_hotkey="hk",
            miner_uid=1,
            code_hash="abc",
            bucket_path="https://example.com/train.py",
            spec_version=19,
        )
        await db.save_submission(sub)
        retrieved = await db.get_submission("opt-fields")
        assert retrieved.final_score is None
        assert retrieved.error_message is None
        assert retrieved.code_content is None

    async def test_evaluation_mfu_default(self, db: Database):
        sub = SubmissionModel(
            submission_id="eval-mfu-test",
            miner_hotkey="hk",
            miner_uid=1,
            code_hash="abc",
            bucket_path="https://example.com/train.py",
            spec_version=19,
        )
        await db.save_submission(sub)

        ev = EvaluationModel(
            evaluation_id=str(uuid.uuid4()),
            submission_id="eval-mfu-test",
            evaluator_hotkey="val-1",
            tokens_per_second=100.0,
            total_tokens=1000,
            wall_time_seconds=10.0,
            success=True,
        )
        await db.save_evaluation(ev)
        evals = await db.get_evaluations("eval-mfu-test")
        assert evals[0].mfu == 0.0


class TestMetadataTableNames:
    """All models register their expected table names."""

    def test_table_names(self):
        table_names = {t.name for t in Base.metadata.sorted_tables}
        expected = {"submissions", "evaluations", "validator_state",
                    "adaptive_threshold", "verified_payments"}
        assert expected.issubset(table_names)
