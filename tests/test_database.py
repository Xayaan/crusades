"""Tests for async Database operations."""

import uuid

import pytest

from crusades.core.protocols import SubmissionStatus
from crusades.storage.database import Database
from crusades.storage.models import (
    EvaluationModel,
    SubmissionModel,
)


class TestInitialize:
    """Database initialization and idempotency."""

    async def test_initialize_creates_tables(self, db: Database):
        """Tables exist after initialize."""
        async with db.engine.connect() as conn:
            from sqlalchemy import inspect as sa_inspect

            table_names = await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_table_names()
            )
        assert "submissions" in table_names
        assert "evaluations" in table_names
        assert "validator_state" in table_names

    async def test_initialize_is_idempotent(self, db: Database):
        """Calling initialize twice does not raise."""
        await db.initialize()  # second call


class TestSubmissionCRUD:
    """Submission save, get, update operations."""

    async def test_save_and_get(self, db: Database):
        sub = SubmissionModel(
            submission_id="test-sub-1",
            miner_hotkey="hk1",
            miner_uid=1,
            code_hash="hash1",
            bucket_path="https://example.com/1",
            spec_version=19,
        )
        await db.save_submission(sub)

        retrieved = await db.get_submission("test-sub-1")
        assert retrieved is not None
        assert retrieved.miner_hotkey == "hk1"
        assert retrieved.status == SubmissionStatus.PENDING

    async def test_get_nonexistent_returns_none(self, db: Database):
        result = await db.get_submission("nonexistent")
        assert result is None

    async def test_update_status(self, db: Database):
        sub = SubmissionModel(
            submission_id="test-status",
            miner_hotkey="hk1",
            miner_uid=1,
            code_hash="h",
            bucket_path="https://example.com/x",
            spec_version=19,
        )
        await db.save_submission(sub)

        await db.update_submission_status(
            "test-status",
            SubmissionStatus.FAILED_VALIDATION,
            error_message="bad syntax",
        )
        updated = await db.get_submission("test-status")
        assert updated.status == SubmissionStatus.FAILED_VALIDATION
        assert updated.error_message == "bad syntax"

    async def test_update_score_sets_finished(self, db: Database):
        sub = SubmissionModel(
            submission_id="test-score",
            miner_hotkey="hk1",
            miner_uid=1,
            code_hash="h",
            bucket_path="https://example.com/x",
            spec_version=19,
            status=SubmissionStatus.EVALUATING,
        )
        await db.save_submission(sub)

        await db.update_submission_score("test-score", 72.5)
        updated = await db.get_submission("test-score")
        assert updated.final_score == 72.5
        assert updated.status == SubmissionStatus.FINISHED

    async def test_update_code_content(self, db: Database):
        sub = SubmissionModel(
            submission_id="test-code",
            miner_hotkey="hk1",
            miner_uid=1,
            code_hash="h",
            bucket_path="https://example.com/x",
            spec_version=19,
        )
        await db.save_submission(sub)

        await db.update_submission_code("test-code", "def inner_steps(): pass")
        code = await db.get_submission_code("test-code")
        assert code == "def inner_steps(): pass"

    async def test_get_pending_submissions(self, seeded_db: Database):
        pending = await seeded_db.get_pending_submissions(spec_version=19)
        assert len(pending) == 1
        assert pending[0].submission_id == "sub-pending-1"

    async def test_get_latest_submission_by_hotkey(self, seeded_db: Database):
        latest = await seeded_db.get_latest_submission_by_hotkey("hotkey_alice")
        assert latest is not None
        assert latest.submission_id == "sub-finished-1"

    async def test_get_latest_submission_by_hotkey_none(self, db: Database):
        latest = await db.get_latest_submission_by_hotkey("nonexistent")
        assert latest is None


class TestEvaluationCRUD:
    """Evaluation save and retrieval."""

    async def test_save_and_get_evaluations(self, db: Database):
        sub = SubmissionModel(
            submission_id="eval-sub",
            miner_hotkey="hk1",
            miner_uid=1,
            code_hash="h",
            bucket_path="https://example.com/x",
            spec_version=19,
        )
        await db.save_submission(sub)

        ev = EvaluationModel(
            evaluation_id=str(uuid.uuid4()),
            submission_id="eval-sub",
            evaluator_hotkey="val1",
            mfu=60.0,
            tokens_per_second=1000.0,
            total_tokens=20000,
            wall_time_seconds=20.0,
            success=True,
        )
        await db.save_evaluation(ev)

        evals = await db.get_evaluations("eval-sub")
        assert len(evals) == 1
        assert evals[0].mfu == 60.0

    async def test_count_evaluations(self, seeded_db: Database):
        count = await seeded_db.count_evaluations("sub-finished-1")
        assert count == 1


class TestLeaderboard:
    """Leaderboard with threshold logic."""

    async def test_get_top_submission(self, seeded_db: Database):
        top = await seeded_db.get_top_submission(spec_version=19)
        assert top is not None
        assert top.final_score == 65.5

    async def test_get_leaderboard_winner_default_threshold(self, seeded_db: Database):
        """With default 1% threshold, higher-score submission wins."""
        winner = await seeded_db.get_leaderboard_winner(threshold=0.01, spec_version=19)
        assert winner is not None
        # sub-finished-2 (55.0) created first, sub-finished-1 (65.5) beats it by >1%
        assert winner.submission_id == "sub-finished-1"

    async def test_get_leaderboard_winner_high_threshold(self, seeded_db: Database):
        """With very high threshold, first submission holds rank 1."""
        winner = await seeded_db.get_leaderboard_winner(threshold=0.50, spec_version=19)
        assert winner is not None
        # 65.5 > 55.0 * 1.50 = 82.5? No. So sub-finished-2 stays at #1.
        assert winner.submission_id == "sub-finished-2"

    async def test_get_leaderboard_no_submissions(self, db: Database):
        winner = await db.get_leaderboard_winner(threshold=0.01)
        assert winner is None

    async def test_get_leaderboard_ordered(self, seeded_db: Database):
        board = await seeded_db.get_leaderboard(limit=10, spec_version=19, threshold=0.01)
        assert len(board) == 2
        # Winner at position 0
        assert board[0].final_score >= board[1].final_score


class TestValidatorState:
    """Key-value validator state persistence."""

    async def test_set_and_get(self, db: Database):
        await db.set_validator_state("last_block", "12345")
        val = await db.get_validator_state("last_block")
        assert val == "12345"

    async def test_get_nonexistent(self, db: Database):
        val = await db.get_validator_state("missing_key")
        assert val is None

    async def test_upsert(self, db: Database):
        await db.set_validator_state("key", "v1")
        await db.set_validator_state("key", "v2")
        val = await db.get_validator_state("key")
        assert val == "v2"


class TestPaymentVerification:
    """Payment recording and double-spend prevention."""

    async def test_record_and_check(self, db: Database):
        sub = SubmissionModel(
            submission_id="pay-sub",
            miner_hotkey="hk1",
            miner_uid=1,
            code_hash="h",
            bucket_path="https://example.com/x",
            spec_version=19,
        )
        await db.save_submission(sub)

        await db.record_verified_payment(
            submission_id="pay-sub",
            miner_hotkey="hk1",
            miner_coldkey="ck1",
            block_hash="0xabc",
            extrinsic_index=0,
            amount_rao=100_000_000,
        )
        assert await db.is_payment_used("0xabc", 0) is True
        assert await db.is_payment_used("0xabc", 1) is False

    async def test_duplicate_payment_raises(self, db: Database):
        sub1 = SubmissionModel(
            submission_id="pay-dup-1",
            miner_hotkey="hk1",
            miner_uid=1,
            code_hash="h",
            bucket_path="https://example.com/x",
            spec_version=19,
        )
        sub2 = SubmissionModel(
            submission_id="pay-dup-2",
            miner_hotkey="hk2",
            miner_uid=2,
            code_hash="h2",
            bucket_path="https://example.com/y",
            spec_version=19,
        )
        await db.save_submission(sub1)
        await db.save_submission(sub2)

        await db.record_verified_payment(
            submission_id="pay-dup-1",
            miner_hotkey="hk1",
            miner_coldkey="ck1",
            block_hash="0xdup",
            extrinsic_index=0,
            amount_rao=100_000_000,
        )
        with pytest.raises(ValueError, match="already claimed"):
            await db.record_verified_payment(
                submission_id="pay-dup-2",
                miner_hotkey="hk2",
                miner_coldkey="ck2",
                block_hash="0xdup",
                extrinsic_index=0,
                amount_rao=100_000_000,
            )


class TestAdaptiveThreshold:
    """Adaptive threshold decay and update math."""

    async def test_no_state_returns_base(self, db: Database):
        threshold = await db.get_adaptive_threshold(
            current_block=1000,
            base_threshold=0.01,
        )
        assert threshold == 0.01

    async def test_update_threshold_first_submission(self, db: Database):
        new_thresh = await db.update_adaptive_threshold(
            new_score=60.0,
            old_score=0.0,
            current_block=100,
            base_threshold=0.01,
        )
        # old_score=0 -> improvement=base_threshold -> threshold=base_threshold
        assert new_thresh == 0.01

    async def test_update_threshold_improvement(self, db: Database):
        # First leader at 50%
        await db.update_adaptive_threshold(
            new_score=50.0, old_score=0.0, current_block=100, base_threshold=0.01
        )
        # New leader at 60% -> 20% improvement
        new_thresh = await db.update_adaptive_threshold(
            new_score=60.0, old_score=50.0, current_block=200, base_threshold=0.01
        )
        expected = (60.0 - 50.0) / 50.0  # 0.20
        assert abs(new_thresh - expected) < 1e-9

    async def test_threshold_decays_over_blocks(self, db: Database):
        # Set a threshold of 20% at block 100
        await db.update_adaptive_threshold(
            new_score=60.0, old_score=50.0, current_block=100, base_threshold=0.01
        )

        # Get threshold at block 200 (100 blocks = 1 decay step at default interval)
        decayed = await db.get_adaptive_threshold(
            current_block=200,
            base_threshold=0.01,
            decay_percent=0.05,
            decay_interval_blocks=100,
        )
        # Expected: 0.01 + (0.20 - 0.01) * (0.95)^1 = 0.01 + 0.1805 = 0.1905
        expected = 0.01 + (0.20 - 0.01) * 0.95
        assert abs(decayed - expected) < 1e-6

    async def test_threshold_never_below_base(self, db: Database):
        await db.update_adaptive_threshold(
            new_score=50.1, old_score=50.0, current_block=100, base_threshold=0.01
        )
        # After many decay steps
        decayed = await db.get_adaptive_threshold(
            current_block=100_000,
            base_threshold=0.01,
            decay_percent=0.05,
            decay_interval_blocks=100,
        )
        assert decayed >= 0.01

    async def test_zero_decay_interval_returns_current(self, db: Database):
        await db.update_adaptive_threshold(
            new_score=60.0, old_score=50.0, current_block=100, base_threshold=0.01
        )
        result = await db.get_adaptive_threshold(
            current_block=200,
            base_threshold=0.01,
            decay_percent=0.05,
            decay_interval_blocks=0,
        )
        # With interval=0, returns current_threshold unchanged
        assert result == pytest.approx(0.20, abs=1e-6)
