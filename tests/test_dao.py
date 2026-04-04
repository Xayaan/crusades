"""Tests for DAO classes against a real async SQLite database."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from crusades.core.protocols import SubmissionStatus
from crusades.storage.models import (
    Base,
    EvaluationModel,
    SubmissionModel,
)
from crusades.storage.submission_dao import SubmissionDAO
from crusades.storage.evaluation_dao import EvaluationDAO
from crusades.storage.state_dao import StateDAO
from crusades.storage.payment_dao import PaymentDAO


@pytest.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def _make_submission(
    submission_id: str = "sub_1",
    hotkey: str = "hotkey_abc",
    uid: int = 1,
    code_hash: str = "hash123",
    bucket_path: str = "https://example.com/train.py",
    status: SubmissionStatus = SubmissionStatus.PENDING,
    final_score: float | None = None,
    spec_version: int = 1,
) -> SubmissionModel:
    return SubmissionModel(
        submission_id=submission_id,
        miner_hotkey=hotkey,
        miner_uid=uid,
        code_hash=code_hash,
        bucket_path=bucket_path,
        status=status,
        final_score=final_score,
        spec_version=spec_version,
    )


# ---------------------------------------------------------------------------
# SubmissionDAO
# ---------------------------------------------------------------------------


class TestSubmissionDAO:
    async def test_save_and_get_roundtrip(self, session_factory):
        dao = SubmissionDAO(session_factory)
        sub = _make_submission()
        await dao.save_submission(sub)

        fetched = await dao.get_submission("sub_1")
        assert fetched is not None
        assert fetched.submission_id == "sub_1"
        assert fetched.miner_hotkey == "hotkey_abc"
        assert fetched.status == SubmissionStatus.PENDING

    async def test_get_nonexistent_returns_none(self, session_factory):
        dao = SubmissionDAO(session_factory)
        assert await dao.get_submission("nonexistent") is None

    async def test_update_status(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(_make_submission())

        await dao.update_submission_status("sub_1", SubmissionStatus.EVALUATING)
        fetched = await dao.get_submission("sub_1")
        assert fetched.status == SubmissionStatus.EVALUATING

    async def test_update_status_with_error_message(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(_make_submission())

        await dao.update_submission_status(
            "sub_1", SubmissionStatus.FAILED_EVALUATION, error_message="boom"
        )
        fetched = await dao.get_submission("sub_1")
        assert fetched.status == SubmissionStatus.FAILED_EVALUATION
        assert fetched.error_message == "boom"

    async def test_update_score_sets_finished(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(_make_submission())

        await dao.update_submission_score("sub_1", 42.5)
        fetched = await dao.get_submission("sub_1")
        assert fetched.final_score == 42.5
        assert fetched.status == SubmissionStatus.FINISHED

    async def test_update_and_get_code(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(_make_submission())

        await dao.update_submission_code("sub_1", "def inner_steps(): pass")
        code = await dao.get_submission_code("sub_1")
        assert code == "def inner_steps(): pass"

    async def test_get_pending_submissions(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(_make_submission("s1", status=SubmissionStatus.PENDING))
        await dao.save_submission(_make_submission("s2", status=SubmissionStatus.EVALUATING))
        await dao.save_submission(
            _make_submission("s3", status=SubmissionStatus.PENDING, spec_version=2)
        )

        pending = await dao.get_pending_submissions()
        assert len(pending) == 2

        pending_v2 = await dao.get_pending_submissions(spec_version=2)
        assert len(pending_v2) == 1
        assert pending_v2[0].submission_id == "s3"

    async def test_get_evaluating_submissions(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(_make_submission("s1", status=SubmissionStatus.EVALUATING))
        await dao.save_submission(_make_submission("s2", status=SubmissionStatus.PENDING))

        evaluating = await dao.get_evaluating_submissions()
        assert len(evaluating) == 1
        assert evaluating[0].submission_id == "s1"

    async def test_get_latest_submission_by_hotkey(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(_make_submission("s1", hotkey="hk1"))
        await dao.save_submission(_make_submission("s2", hotkey="hk1"))
        await dao.save_submission(_make_submission("s3", hotkey="hk2"))

        latest = await dao.get_latest_submission_by_hotkey("hk1")
        assert latest is not None
        assert latest.miner_hotkey == "hk1"

    async def test_get_top_submission(self, session_factory):
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(
            _make_submission("s1", status=SubmissionStatus.FINISHED, final_score=10.0)
        )
        await dao.save_submission(
            _make_submission("s2", status=SubmissionStatus.FINISHED, final_score=50.0)
        )
        await dao.save_submission(
            _make_submission("s3", status=SubmissionStatus.PENDING, final_score=None)
        )

        top = await dao.get_top_submission()
        assert top is not None
        assert top.submission_id == "s2"
        assert top.final_score == 50.0

    async def test_get_leaderboard_winner_threshold_retains_incumbent(self, session_factory):
        """A new submission must beat incumbent by >threshold to take #1."""
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(
            _make_submission("s1", status=SubmissionStatus.FINISHED, final_score=10.0)
        )
        # Challenger: 10.05 MFU (only 0.5% better, below 1% threshold)
        await dao.save_submission(
            _make_submission("s2", status=SubmissionStatus.FINISHED, final_score=10.05)
        )

        winner = await dao.get_leaderboard_winner(threshold=0.01)
        assert winner is not None
        assert winner.submission_id == "s1"

    async def test_get_leaderboard_winner_beats_threshold(self, session_factory):
        """Challenger that exceeds threshold takes #1."""
        dao = SubmissionDAO(session_factory)
        await dao.save_submission(
            _make_submission("s1", status=SubmissionStatus.FINISHED, final_score=10.0)
        )
        await dao.save_submission(
            _make_submission("s2", status=SubmissionStatus.FINISHED, final_score=11.0)
        )

        winner = await dao.get_leaderboard_winner(threshold=0.01)
        assert winner is not None
        assert winner.submission_id == "s2"

    async def test_get_top_submissions_limit(self, session_factory):
        dao = SubmissionDAO(session_factory)
        for i in range(5):
            await dao.save_submission(
                _make_submission(
                    f"s{i}",
                    status=SubmissionStatus.FINISHED,
                    final_score=float(i * 10),
                )
            )

        top3 = await dao.get_top_submissions(limit=3)
        assert len(top3) == 3
        # Highest scores first
        assert top3[0].final_score >= top3[1].final_score >= top3[2].final_score


# ---------------------------------------------------------------------------
# EvaluationDAO
# ---------------------------------------------------------------------------


class TestEvaluationDAO:
    async def test_save_and_get_evaluations(self, session_factory):
        sub_dao = SubmissionDAO(session_factory)
        eval_dao = EvaluationDAO(session_factory)

        await sub_dao.save_submission(_make_submission("sub_1"))

        ev = EvaluationModel(
            submission_id="sub_1",
            evaluator_hotkey="val_hk",
            mfu=55.0,
            tokens_per_second=1000.0,
            total_tokens=5000,
            wall_time_seconds=10.0,
            success=True,
        )
        await eval_dao.save_evaluation(ev)

        evals = await eval_dao.get_evaluations("sub_1")
        assert len(evals) == 1
        assert evals[0].mfu == 55.0
        assert evals[0].success is True

    async def test_count_evaluations_includes_failures(self, session_factory):
        sub_dao = SubmissionDAO(session_factory)
        eval_dao = EvaluationDAO(session_factory)

        await sub_dao.save_submission(_make_submission("sub_1"))

        for i, success in enumerate([True, False, True]):
            await eval_dao.save_evaluation(
                EvaluationModel(
                    evaluation_id=f"ev_{i}",
                    submission_id="sub_1",
                    evaluator_hotkey="val_hk",
                    mfu=50.0 if success else 0.0,
                    tokens_per_second=100.0,
                    total_tokens=1000,
                    wall_time_seconds=5.0,
                    success=success,
                    error=None if success else "failed",
                )
            )

        count = await eval_dao.count_evaluations("sub_1")
        assert count == 3


# ---------------------------------------------------------------------------
# StateDAO
# ---------------------------------------------------------------------------


class TestStateDAO:
    async def test_get_set_validator_state(self, session_factory):
        dao = StateDAO(session_factory)

        assert await dao.get_validator_state("my_key") is None

        await dao.set_validator_state("my_key", "my_value")
        assert await dao.get_validator_state("my_key") == "my_value"

        # Upsert
        await dao.set_validator_state("my_key", "updated")
        assert await dao.get_validator_state("my_key") == "updated"

    async def test_adaptive_threshold_no_state_returns_base(self, session_factory):
        dao = StateDAO(session_factory)
        threshold = await dao.get_adaptive_threshold(current_block=1000)
        assert threshold == 0.01

    async def test_adaptive_threshold_decay(self, session_factory):
        dao = StateDAO(session_factory)

        # Set a threshold of 20% at block 1000
        await dao.update_adaptive_threshold(
            new_score=120.0, old_score=100.0, current_block=1000
        )

        # Immediately: threshold = 0.20
        t0 = await dao.get_adaptive_threshold(current_block=1000)
        assert abs(t0 - 0.20) < 0.001

        # After 1 decay step (100 blocks, 5% decay)
        t1 = await dao.get_adaptive_threshold(current_block=1100)
        expected = 0.01 + (0.20 - 0.01) * (1.0 - 0.05)
        assert abs(t1 - expected) < 0.001

        # Far future: converges to base
        t_far = await dao.get_adaptive_threshold(current_block=100000)
        assert t_far < 0.02

    async def test_adaptive_threshold_update_first_submission(self, session_factory):
        dao = StateDAO(session_factory)

        new_t = await dao.update_adaptive_threshold(
            new_score=50.0, old_score=0.0, current_block=500
        )
        assert new_t == 0.01

    async def test_adaptive_threshold_zero_decay_interval(self, session_factory):
        dao = StateDAO(session_factory)

        await dao.update_adaptive_threshold(
            new_score=120.0, old_score=100.0, current_block=1000
        )

        t = await dao.get_adaptive_threshold(
            current_block=2000, decay_interval_blocks=0
        )
        assert abs(t - 0.20) < 0.001


# ---------------------------------------------------------------------------
# PaymentDAO
# ---------------------------------------------------------------------------


class TestPaymentDAO:
    async def test_is_payment_used_false_initially(self, session_factory):
        dao = PaymentDAO(session_factory)
        assert await dao.is_payment_used("0xabc", 0) is False

    async def test_record_and_check_payment(self, session_factory):
        dao = PaymentDAO(session_factory)

        await dao.record_verified_payment(
            submission_id="sub_1",
            miner_hotkey="hk1",
            miner_coldkey="ck1",
            block_hash="0xabc123",
            extrinsic_index=3,
            amount_rao=1000000,
        )

        assert await dao.is_payment_used("0xabc123", 3) is True
        assert await dao.is_payment_used("0xabc123", 4) is False
        assert await dao.is_payment_used("0xother", 3) is False

    async def test_double_spend_raises(self, session_factory):
        dao = PaymentDAO(session_factory)

        await dao.record_verified_payment(
            submission_id="sub_1",
            miner_hotkey="hk1",
            miner_coldkey="ck1",
            block_hash="0xabc",
            extrinsic_index=0,
            amount_rao=500,
        )

        with pytest.raises(ValueError, match="already claimed"):
            await dao.record_verified_payment(
                submission_id="sub_2",
                miner_hotkey="hk2",
                miner_coldkey="ck2",
                block_hash="0xabc",
                extrinsic_index=0,
                amount_rao=500,
            )

    async def test_get_payment_for_submission(self, session_factory):
        dao = PaymentDAO(session_factory)

        await dao.record_verified_payment(
            submission_id="sub_1",
            miner_hotkey="hk1",
            miner_coldkey="ck1",
            block_hash="0xblock",
            extrinsic_index=7,
            amount_rao=999,
        )

        payment = await dao.get_payment_for_submission("sub_1")
        assert payment is not None
        assert payment.block_hash == "0xblock"
        assert payment.extrinsic_index == 7
        assert payment.amount_rao == 999

        assert await dao.get_payment_for_submission("sub_2") is None
