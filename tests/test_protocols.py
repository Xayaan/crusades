"""Tests for protocol dataclasses."""

from crusades.core.protocols import SandboxResult, ValidationResult


class TestSandboxResult:
    def test_creation(self):
        result = SandboxResult(
            success=True,
            tokens_per_second=100.0,
            total_tokens=1000,
            wall_time_seconds=10.0,
            exit_code=0,
            stdout="ok",
            stderr="",
        )
        assert result.success is True
        assert result.final_loss is None
        assert result.final_logits is None
        assert result.tokens_per_second == 100.0

    def test_with_optional_fields(self):
        result = SandboxResult(
            success=False,
            tokens_per_second=0.0,
            total_tokens=0,
            wall_time_seconds=5.0,
            exit_code=1,
            stdout="",
            stderr="error",
            error="timeout",
            final_loss=2.5,
        )
        assert result.success is False
        assert result.error == "timeout"
        assert result.final_loss == 2.5


class TestValidationResult:
    def test_valid(self):
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []

    def test_invalid_with_errors(self):
        result = ValidationResult(valid=False, errors=["syntax error"])
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "syntax error"

    def test_multiple_errors(self):
        result = ValidationResult(
            valid=False,
            errors=["forbidden import", "missing function"],
        )
        assert len(result.errors) == 2
