"""Tests for EvaluationErrorCode classification methods."""

import pytest

from crusades.core.exceptions import EvaluationError, EvaluationErrorCode

# Every member must be classified by exactly the expected predicates.
# Table-driven: exhaustive coverage of the enum surface.

VERIFICATION_CODES = {
    EvaluationErrorCode.INSUFFICIENT_TRAINABLE_PARAMS,
    EvaluationErrorCode.INSUFFICIENT_PARAMS_CHANGED,
    EvaluationErrorCode.GRADIENT_COVERAGE_FAILED,
    EvaluationErrorCode.GRADIENT_RELATIVE_ERROR_FAILED,
    EvaluationErrorCode.LOSS_MISMATCH,
    EvaluationErrorCode.TOKEN_COUNT_MISMATCH,
    EvaluationErrorCode.NO_GRADIENTS_CAPTURED,
    EvaluationErrorCode.MISSING_LOGITS,
    EvaluationErrorCode.INVALID_LOGITS_SHAPE,
    EvaluationErrorCode.SEQUENCE_TRUNCATION,
    EvaluationErrorCode.WEIGHT_MISMATCH,
}

FATAL_CODES = {
    EvaluationErrorCode.NO_CODE,
    EvaluationErrorCode.SYNTAX_ERROR,
    EvaluationErrorCode.MISSING_INNER_STEPS,
    EvaluationErrorCode.INVALID_RETURN_TYPE,
    EvaluationErrorCode.INSUFFICIENT_TRAINABLE_PARAMS,
    EvaluationErrorCode.NO_GRADIENTS_CAPTURED,
    EvaluationErrorCode.MISSING_LOGITS,
    EvaluationErrorCode.INVALID_LOGITS_SHAPE,
    EvaluationErrorCode.SEQUENCE_TRUNCATION,
}

INFRASTRUCTURE_CODES = {
    EvaluationErrorCode.MODEL_LOAD_FAILED,
    EvaluationErrorCode.DATA_LOAD_FAILED,
    EvaluationErrorCode.DOCKER_FAILED,
    EvaluationErrorCode.TIMEOUT,
}


@pytest.mark.parametrize("code", list(EvaluationErrorCode))
class TestErrorCodeClassification:
    """Each error code is tested against all three classifiers."""

    def test_is_verification_failure(self, code: EvaluationErrorCode):
        expected = code in VERIFICATION_CODES
        assert EvaluationErrorCode.is_verification_failure(code) is expected, (
            f"{code}: expected is_verification_failure={expected}"
        )

    def test_is_fatal(self, code: EvaluationErrorCode):
        expected = code in FATAL_CODES
        assert EvaluationErrorCode.is_fatal(code) is expected, (
            f"{code}: expected is_fatal={expected}"
        )

    def test_is_miner_fault(self, code: EvaluationErrorCode):
        expected = code not in INFRASTRUCTURE_CODES
        assert EvaluationErrorCode.is_miner_fault(code) is expected, (
            f"{code}: expected is_miner_fault={expected}"
        )


def test_evaluation_error_carries_code():
    """EvaluationError preserves message and code."""
    err = EvaluationError("boom", EvaluationErrorCode.TIMEOUT)
    assert str(err) == "boom"
    assert err.code == EvaluationErrorCode.TIMEOUT
    assert err.message == "boom"


def test_evaluation_error_default_code():
    """EvaluationError defaults to UNKNOWN."""
    err = EvaluationError("unknown failure")
    assert err.code == EvaluationErrorCode.UNKNOWN


def test_error_code_values_are_strings():
    """All error codes are valid StrEnum members."""
    for code in EvaluationErrorCode:
        assert isinstance(code.value, str)
        assert code.value == str(code)
