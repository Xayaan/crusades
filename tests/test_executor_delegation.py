"""Tests for AffinetesRunner executor delegation and EvaluationResult logic."""

from unittest.mock import AsyncMock, patch

import pytest

from crusades.affinetes.runner import AffinetesRunner, EvaluationResult
from crusades.affinetes.executors.base import EvalConfig
from crusades.affinetes.executors.docker_executor import DockerExecutor
from crusades.affinetes.executors.basilica_executor import BasilicaExecutor
from crusades.core.exceptions import EvaluationErrorCode


# ---------------------------------------------------------------------------
# AffinetesRunner executor selection
# ---------------------------------------------------------------------------


class TestAffinetesRunnerDelegation:
    def test_docker_mode_selects_docker_executor(self):
        runner = AffinetesRunner(mode="docker")
        assert isinstance(runner._executor, DockerExecutor)
        assert runner._docker_executor is not None
        assert runner._basilica_executor is None

    def test_basilica_mode_selects_basilica_executor(self):
        runner = AffinetesRunner(mode="basilica", basilica_api_key="fake-key")
        assert isinstance(runner._executor, BasilicaExecutor)
        assert runner._basilica_executor is not None
        assert runner._docker_executor is None

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            AffinetesRunner(mode="unknown")

    async def test_evaluate_delegates_to_docker_executor(self):
        runner = AffinetesRunner(mode="docker", model_url="m", data_url="d")
        mock_result = EvaluationResult(success=True, mfu=55.0, tps=1000.0)

        runner._executor = AsyncMock()
        runner._executor.evaluate = AsyncMock(return_value=mock_result)

        result = await runner.evaluate(
            code="def inner_steps(m,d,o,n,dev,num_gpus=1): pass",
            seed=42,
        )

        runner._executor.evaluate.assert_awaited_once()
        call_kwargs = runner._executor.evaluate.call_args
        assert call_kwargs.kwargs["code"] == "def inner_steps(m,d,o,n,dev,num_gpus=1): pass"
        assert call_kwargs.kwargs["seed"] == "42"
        assert result.success is True
        assert result.mfu == 55.0

    async def test_evaluate_rejects_missing_inner_steps(self):
        runner = AffinetesRunner(mode="docker", model_url="m", data_url="d")
        result = await runner.evaluate(code="x = 1", seed=0)
        assert result.success is False
        assert "inner_steps" in result.error

    async def test_evaluate_rejects_missing_urls(self):
        runner = AffinetesRunner(mode="docker")
        result = await runner.evaluate(
            code="def inner_steps(): pass", seed=0
        )
        assert result.success is False
        assert "required" in result.error

    async def test_create_basilica_deployment_requires_basilica_mode(self):
        runner = AffinetesRunner(mode="docker")
        with pytest.raises(RuntimeError, match="requires mode='basilica'"):
            await runner.create_basilica_deployment()

    async def test_destroy_basilica_deployment_requires_basilica_mode(self):
        runner = AffinetesRunner(mode="docker")
        with pytest.raises(RuntimeError, match="requires mode='basilica'"):
            await runner.destroy_basilica_deployment(None)

    async def test_evaluate_on_deployment_requires_basilica_mode(self):
        runner = AffinetesRunner(mode="docker")
        with pytest.raises(RuntimeError, match="requires mode='basilica'"):
            await runner.evaluate_on_deployment(
                ctx=None, code="x", seed="0"
            )

    def test_eval_config_passed_to_executor(self):
        runner = AffinetesRunner(
            mode="docker",
            timeout=999,
            gpu_peak_tflops=200.0,
            max_plausible_mfu=80.0,
            min_mfu=30.0,
        )
        config = runner._docker_executor.config
        assert config.timeout == 999
        assert config.gpu_peak_tflops == 200.0
        assert config.max_plausible_mfu == 80.0
        assert config.min_mfu == 30.0


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_from_dict(self):
        data = {
            "success": True,
            "mfu": 62.5,
            "tps": 2000.0,
            "total_tokens": 10000,
            "wall_time_seconds": 30.0,
            "error": None,
            "error_code": None,
            "seed": "abc",
            "task_id": 3,
            "diagnostics": {"gpu": "A100"},
        }
        result = EvaluationResult.from_dict(data)
        assert result.success is True
        assert result.mfu == 62.5
        assert result.task_id == 3
        assert result.diagnostics == {"gpu": "A100"}

    def test_from_dict_defaults(self):
        result = EvaluationResult.from_dict({})
        assert result.success is False
        assert result.mfu == 0.0

    def test_failure_factory(self):
        result = EvaluationResult.failure("timeout", task_id=5, error_code="timeout")
        assert result.success is False
        assert result.error == "timeout"
        assert result.task_id == 5

    def test_is_verification_failure(self):
        result = EvaluationResult(
            success=False, error_code=EvaluationErrorCode.LOSS_MISMATCH
        )
        assert result.is_verification_failure() is True

        result2 = EvaluationResult(
            success=False, error_code=EvaluationErrorCode.TIMEOUT
        )
        assert result2.is_verification_failure() is False

    def test_is_fatal(self):
        fatal = EvaluationResult(
            success=False, error_code=EvaluationErrorCode.SYNTAX_ERROR
        )
        assert fatal.is_fatal() is True

        non_fatal = EvaluationResult(
            success=False, error_code=EvaluationErrorCode.INSUFFICIENT_PARAMS_CHANGED
        )
        assert non_fatal.is_fatal() is False

        success = EvaluationResult(success=True)
        assert success.is_fatal() is False

        unknown = EvaluationResult(success=False, error_code=None)
        assert unknown.is_fatal() is False

    def test_is_miner_fault(self):
        miner = EvaluationResult(
            success=False, error_code=EvaluationErrorCode.SYNTAX_ERROR
        )
        assert miner.is_miner_fault() is True

        infra = EvaluationResult(
            success=False, error_code=EvaluationErrorCode.DOCKER_FAILED
        )
        assert infra.is_miner_fault() is False

        success = EvaluationResult(success=True)
        assert success.is_miner_fault() is False

    def test_is_miner_fault_no_error_code_assumes_miner(self):
        result = EvaluationResult(success=False, error="unknown")
        assert result.is_miner_fault() is True
