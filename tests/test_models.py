"""Tests for domain model dataclasses and EvaluationErrorCode classification."""

from crusades.models.submission import SubmissionScore
from crusades.models.evaluation import MFUResult, VerificationResult
from crusades.core.exceptions import EvaluationErrorCode


class TestSubmissionScore:
    def test_defaults(self):
        score = SubmissionScore()
        assert score.median_mfu == 0.0
        assert score.mfu_scores == []
        assert score.total_runs == 0
        assert score.successful_runs == 0
        assert score.success_rate == 0.0

    def test_mutable_default_isolation(self):
        """Each instance gets its own list — no shared mutable state."""
        a = SubmissionScore()
        b = SubmissionScore()
        a.mfu_scores.append(99.0)
        assert b.mfu_scores == []


class TestMFUResult:
    def test_defaults_reflect_single_a100(self):
        result = MFUResult()
        assert result.gpu_peak_tflops == 312.0
        assert result.num_gpus == 1

    def test_multi_gpu(self):
        result = MFUResult(num_gpus=8, gpu_peak_tflops=312.0)
        assert result.num_gpus == 8


class TestVerificationResult:
    def test_failed_result_carries_error(self):
        result = VerificationResult(
            passed=False,
            error="loss diverged",
            error_code="loss_mismatch",
            checks_failed=[{"check": "loss_check", "detail": "delta > threshold"}],
        )
        assert result.passed is False
        assert result.error == "loss diverged"
        assert len(result.checks_failed) == 1

    def test_passed_result(self):
        result = VerificationResult(
            passed=True,
            checks_passed=["loss_check", "logits_check", "weight_check"],
        )
        assert result.passed is True
        assert len(result.checks_passed) == 3
        assert result.checks_failed == []


class TestEvaluationErrorCode:
    """Verify the classification functions on EvaluationErrorCode."""

    def test_verification_failures_are_classified(self):
        verification_codes = [
            EvaluationErrorCode.LOSS_MISMATCH,
            EvaluationErrorCode.TOKEN_COUNT_MISMATCH,
            EvaluationErrorCode.MISSING_LOGITS,
            EvaluationErrorCode.WEIGHT_MISMATCH,
        ]
        for code in verification_codes:
            assert EvaluationErrorCode.is_verification_failure(code), f"{code} should be verification"

    def test_non_verification_codes(self):
        non_verification = [
            EvaluationErrorCode.TIMEOUT,
            EvaluationErrorCode.DOCKER_FAILED,
            EvaluationErrorCode.UNKNOWN,
        ]
        for code in non_verification:
            assert not EvaluationErrorCode.is_verification_failure(code), f"{code} not verification"

    def test_fatal_codes_are_deterministic(self):
        fatal_codes = [
            EvaluationErrorCode.SYNTAX_ERROR,
            EvaluationErrorCode.MISSING_INNER_STEPS,
            EvaluationErrorCode.NO_GRADIENTS_CAPTURED,
            EvaluationErrorCode.INVALID_LOGITS_SHAPE,
        ]
        for code in fatal_codes:
            assert EvaluationErrorCode.is_fatal(code), f"{code} should be fatal"

    def test_non_fatal_data_dependent_codes(self):
        non_fatal = [
            EvaluationErrorCode.INSUFFICIENT_PARAMS_CHANGED,
            EvaluationErrorCode.GRADIENT_RELATIVE_ERROR_FAILED,
            EvaluationErrorCode.LOSS_MISMATCH,
        ]
        for code in non_fatal:
            assert not EvaluationErrorCode.is_fatal(code), f"{code} should not be fatal"

    def test_miner_fault_vs_infrastructure(self):
        assert EvaluationErrorCode.is_miner_fault(EvaluationErrorCode.SYNTAX_ERROR)
        assert EvaluationErrorCode.is_miner_fault(EvaluationErrorCode.LOSS_MISMATCH)
        assert not EvaluationErrorCode.is_miner_fault(EvaluationErrorCode.DOCKER_FAILED)
        assert not EvaluationErrorCode.is_miner_fault(EvaluationErrorCode.MODEL_LOAD_FAILED)
