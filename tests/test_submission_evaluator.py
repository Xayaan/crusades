"""Tests for SubmissionEvaluator: finalize_submission and download_from_url."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from crusades.core.protocols import SubmissionStatus
from crusades.storage.models import Base, EvaluationModel, SubmissionModel
from crusades.storage.database import Database
from crusades.services.submission_evaluator import SubmissionEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db():
    """Real Database backed by in-memory SQLite."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.initialize()
    yield database
    await database.close()


def _make_hparams(**overrides):
    """Build a minimal mock hparams with defaults matching production."""
    hp = MagicMock()
    hp.min_success_rate = overrides.get("min_success_rate", 0.5)
    hp.evaluation_runs = overrides.get("evaluation_runs", 3)
    hp.eval_timeout = overrides.get("eval_timeout", 600)
    hp.eval_steps = overrides.get("eval_steps", 5)
    hp.benchmark_batch_size = overrides.get("benchmark_batch_size", 8)
    hp.benchmark_sequence_length = overrides.get("benchmark_sequence_length", 1024)
    hp.benchmark_data_samples = overrides.get("benchmark_data_samples", 10000)
    hp.code_fetch_timeout = overrides.get("code_fetch_timeout", 30)
    hp.adaptive_threshold.base_threshold = 0.01
    hp.adaptive_threshold.decay_percent = 0.05
    hp.adaptive_threshold.decay_interval_blocks = 100
    return hp


async def _seed_submission(db: Database, submission_id: str = "sub_1") -> None:
    """Insert a submission in EVALUATING state."""
    sub = SubmissionModel(
        submission_id=submission_id,
        miner_hotkey="hk_test",
        miner_uid=1,
        code_hash="abc123",
        bucket_path="https://example.com/train.py",
        status=SubmissionStatus.EVALUATING,
        spec_version=1,
    )
    await db.save_submission(sub)


async def _add_evaluation(
    db: Database,
    submission_id: str,
    success: bool,
    mfu: float = 55.0,
    error: str | None = None,
    eval_id: str | None = None,
) -> None:
    ev = EvaluationModel(
        evaluation_id=eval_id or f"ev_{submission_id}_{mfu}_{success}",
        submission_id=submission_id,
        evaluator_hotkey="val_hk",
        mfu=mfu if success else 0.0,
        tokens_per_second=1000.0 if success else 0.0,
        total_tokens=5000 if success else 0,
        wall_time_seconds=10.0,
        success=success,
        error=error,
    )
    await db.save_evaluation(ev)


# ---------------------------------------------------------------------------
# _finalize_submission
# ---------------------------------------------------------------------------


class TestFinalizeSubmission:
    @patch("crusades.services.submission_evaluator.get_hparams")
    async def test_fatal_error_fails_immediately(self, mock_hparams, db):
        mock_hparams.return_value = _make_hparams()
        await _seed_submission(db)
        await _add_evaluation(db, "sub_1", success=False, error="missing_inner_steps")

        evaluator = SubmissionEvaluator(
            db=db,
            affinetes_runner=MagicMock(),
            hotkey="val_hk",
        )

        await evaluator._finalize_submission("sub_1", num_runs=3, fatal_error=True)

        sub = await db.get_submission("sub_1")
        assert sub.status == SubmissionStatus.FAILED_EVALUATION
        assert "Fatal error" in sub.error_message

    @patch("crusades.services.submission_evaluator.get_hparams")
    async def test_low_success_rate_fails(self, mock_hparams, db):
        mock_hparams.return_value = _make_hparams(min_success_rate=0.5)
        await _seed_submission(db)

        # 1 success, 2 failures = 33% success rate, below 50% minimum
        await _add_evaluation(db, "sub_1", success=True, mfu=55.0, eval_id="ev1")
        await _add_evaluation(db, "sub_1", success=False, error="err", eval_id="ev2")
        await _add_evaluation(db, "sub_1", success=False, error="err", eval_id="ev3")

        evaluator = SubmissionEvaluator(
            db=db, affinetes_runner=MagicMock(), hotkey="val_hk"
        )

        await evaluator._finalize_submission("sub_1", num_runs=3)

        sub = await db.get_submission("sub_1")
        assert sub.status == SubmissionStatus.FAILED_EVALUATION
        assert "Success rate" in sub.error_message

    @patch("crusades.services.submission_evaluator.get_hparams")
    async def test_successful_finalization_median_mfu(self, mock_hparams, db):
        mock_hparams.return_value = _make_hparams(min_success_rate=0.5)
        await _seed_submission(db)

        # 3 successes with different MFU scores
        await _add_evaluation(db, "sub_1", success=True, mfu=50.0, eval_id="ev1")
        await _add_evaluation(db, "sub_1", success=True, mfu=60.0, eval_id="ev2")
        await _add_evaluation(db, "sub_1", success=True, mfu=70.0, eval_id="ev3")

        evaluator = SubmissionEvaluator(
            db=db, affinetes_runner=MagicMock(), hotkey="val_hk"
        )

        await evaluator._finalize_submission("sub_1", num_runs=3)

        sub = await db.get_submission("sub_1")
        assert sub.status == SubmissionStatus.FINISHED
        # median_low of [50, 60, 70] = 60
        assert sub.final_score == 60.0

    @patch("crusades.services.submission_evaluator.get_hparams")
    async def test_all_evals_failed_scores_zero(self, mock_hparams, db):
        mock_hparams.return_value = _make_hparams(min_success_rate=0.0)
        await _seed_submission(db)

        await _add_evaluation(db, "sub_1", success=False, error="err1", eval_id="ev1")
        await _add_evaluation(db, "sub_1", success=False, error="err2", eval_id="ev2")
        await _add_evaluation(db, "sub_1", success=False, error="err3", eval_id="ev3")

        evaluator = SubmissionEvaluator(
            db=db, affinetes_runner=MagicMock(), hotkey="val_hk"
        )

        await evaluator._finalize_submission("sub_1", num_runs=3)

        sub = await db.get_submission("sub_1")
        assert sub.status == SubmissionStatus.FAILED_EVALUATION
        assert sub.final_score == 0.0

    @patch("crusades.services.submission_evaluator.get_hparams")
    async def test_not_enough_evals_does_not_finalize(self, mock_hparams, db):
        mock_hparams.return_value = _make_hparams()
        await _seed_submission(db)

        # Only 1 eval when 3 required
        await _add_evaluation(db, "sub_1", success=True, mfu=55.0, eval_id="ev1")

        evaluator = SubmissionEvaluator(
            db=db, affinetes_runner=MagicMock(), hotkey="val_hk"
        )

        await evaluator._finalize_submission("sub_1", num_runs=3)

        sub = await db.get_submission("sub_1")
        # Should remain EVALUATING since not enough evals
        assert sub.status == SubmissionStatus.EVALUATING

    @patch("crusades.services.submission_evaluator.get_hparams")
    async def test_median_low_returns_actual_value(self, mock_hparams, db):
        """median_low of [50, 60] = 50, not the average 55."""
        mock_hparams.return_value = _make_hparams(min_success_rate=0.0)
        await _seed_submission(db)

        await _add_evaluation(db, "sub_1", success=True, mfu=50.0, eval_id="ev1")
        await _add_evaluation(db, "sub_1", success=True, mfu=60.0, eval_id="ev2")

        evaluator = SubmissionEvaluator(
            db=db, affinetes_runner=MagicMock(), hotkey="val_hk"
        )

        await evaluator._finalize_submission("sub_1", num_runs=2)

        sub = await db.get_submission("sub_1")
        assert sub.final_score == 50.0  # median_low, not 55.0


# ---------------------------------------------------------------------------
# _download_from_url
# ---------------------------------------------------------------------------


class TestDownloadFromUrl:
    def _make_evaluator(self):
        return SubmissionEvaluator(
            db=MagicMock(),
            affinetes_runner=MagicMock(),
            hotkey="val_hk",
        )

    @patch("crusades.services.submission_evaluator.get_hparams")
    @patch("crusades.services.submission_evaluator.CodeUrlInfo")
    def test_ssrf_blocked_url(self, MockCodeUrlInfo, mock_hparams):
        mock_hparams.return_value = _make_hparams()
        instance = MockCodeUrlInfo.return_value
        instance.validate_url_security.return_value = (False, "Private IP detected")

        evaluator = self._make_evaluator()
        success, msg = evaluator._download_from_url("http://192.168.1.1/train.py")

        assert success is False
        assert "blocked for security" in msg

    @patch("crusades.services.submission_evaluator.get_hparams")
    @patch("crusades.services.submission_evaluator.CodeUrlInfo")
    @patch("crusades.services.submission_evaluator.urllib.request.build_opener")
    def test_file_too_large(self, mock_opener, MockCodeUrlInfo, mock_hparams):
        mock_hparams.return_value = _make_hparams()
        instance = MockCodeUrlInfo.return_value
        instance.validate_url_security.return_value = (True, "1.2.3.4")

        # Simulate a response that exceeds 500KB
        mock_response = MagicMock()
        mock_response.read.side_effect = [b"x" * 8192] * 70  # 560KB
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_opener.return_value.open.return_value = mock_response

        evaluator = self._make_evaluator()
        success, msg = evaluator._download_from_url("https://example.com/big.py")

        assert success is False
        assert "too large" in msg

    @patch("crusades.services.submission_evaluator.get_hparams")
    @patch("crusades.services.submission_evaluator.CodeUrlInfo")
    @patch("crusades.services.submission_evaluator.urllib.request.build_opener")
    def test_html_page_rejected(self, mock_opener, MockCodeUrlInfo, mock_hparams):
        mock_hparams.return_value = _make_hparams()
        instance = MockCodeUrlInfo.return_value
        instance.validate_url_security.return_value = (True, "1.2.3.4")

        html = b"<html><body>Not a Python file</body></html>"
        mock_response = MagicMock()
        mock_response.read.side_effect = [html, b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_opener.return_value.open.return_value = mock_response

        evaluator = self._make_evaluator()
        success, msg = evaluator._download_from_url("https://example.com/file")

        assert success is False
        assert "HTML" in msg

    @patch("crusades.services.submission_evaluator.get_hparams")
    @patch("crusades.services.submission_evaluator.CodeUrlInfo")
    @patch("crusades.services.submission_evaluator.urllib.request.build_opener")
    def test_missing_inner_steps_rejected(self, mock_opener, MockCodeUrlInfo, mock_hparams):
        mock_hparams.return_value = _make_hparams()
        instance = MockCodeUrlInfo.return_value
        instance.validate_url_security.return_value = (True, "1.2.3.4")

        code = b"def train(model): pass"
        mock_response = MagicMock()
        mock_response.read.side_effect = [code, b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_opener.return_value.open.return_value = mock_response

        evaluator = self._make_evaluator()
        success, msg = evaluator._download_from_url("https://example.com/train.py")

        assert success is False
        assert "inner_steps" in msg

    @patch("crusades.services.submission_evaluator.get_hparams")
    @patch("crusades.services.submission_evaluator.CodeUrlInfo")
    @patch("crusades.services.submission_evaluator.urllib.request.build_opener")
    def test_valid_code_succeeds(self, mock_opener, MockCodeUrlInfo, mock_hparams):
        mock_hparams.return_value = _make_hparams()
        instance = MockCodeUrlInfo.return_value
        instance.validate_url_security.return_value = (True, "1.2.3.4")

        code = b"def inner_steps(model, data, opt, steps, device, num_gpus=1): pass"
        mock_response = MagicMock()
        mock_response.read.side_effect = [code, b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_opener.return_value.open.return_value = mock_response

        evaluator = self._make_evaluator()
        success, result_code = evaluator._download_from_url("https://example.com/train.py")

        assert success is True
        assert "def inner_steps" in result_code

    @patch("crusades.services.submission_evaluator.get_hparams")
    @patch("crusades.services.submission_evaluator.CodeUrlInfo")
    @patch("crusades.services.submission_evaluator.urllib.request.build_opener")
    def test_json_listing_rejected(self, mock_opener, MockCodeUrlInfo, mock_hparams):
        mock_hparams.return_value = _make_hparams()
        instance = MockCodeUrlInfo.return_value
        instance.validate_url_security.return_value = (True, "1.2.3.4")

        json_data = b'{"files": {"train.py": {"content": "..."}}}'
        mock_response = MagicMock()
        mock_response.read.side_effect = [json_data, b""]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_opener.return_value.open.return_value = mock_response

        evaluator = self._make_evaluator()
        success, msg = evaluator._download_from_url("https://gist.github.com/user/abc")

        assert success is False
        assert "JSON" in msg

    def test_gist_url_expansion(self):
        """Verify that gist.github.com URLs are expanded to raw form."""
        evaluator = self._make_evaluator()

        # Patch CodeUrlInfo to capture the expanded URL
        with patch("crusades.services.submission_evaluator.CodeUrlInfo") as MockCodeUrlInfo:
            instance = MockCodeUrlInfo.return_value
            instance.validate_url_security.return_value = (False, "test - stop here")

            evaluator._download_from_url("https://gist.github.com/user/abc123")

            # The URL passed to CodeUrlInfo should be the expanded form
            call_url = MockCodeUrlInfo.call_args[1].get("url") or MockCodeUrlInfo.call_args[0][0]
            # Accept either kwarg or positional
            if hasattr(MockCodeUrlInfo, "call_args_list"):
                # The URL should have been rewritten
                found_url = None
                for call in MockCodeUrlInfo.call_args_list:
                    if call.kwargs.get("url"):
                        found_url = call.kwargs["url"]
                    elif call.args:
                        found_url = call.args[0]
                if found_url:
                    assert "gist.githubusercontent.com" in found_url
                    assert found_url.endswith("/raw")
