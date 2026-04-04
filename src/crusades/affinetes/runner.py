"""Affinetes runner for evaluating miner submissions.

URL-Based Architecture:
- Miners host train.py at any URL (Gist, raw GitHub, etc.)
- Validator downloads code from committed URL
- Code is passed directly to the evaluation environment

Execution modes:
1. Docker mode - Local GPU evaluation via Docker container
2. Basilica mode - Remote cloud GPU evaluation via Basilica SDK

Environment Variables:
- BASILICA_API_TOKEN: API token for Basilica cloud GPU service
- VALIDATOR_EVAL_IMAGE: Docker image for local evaluation (default: templar-eval:latest)
- BASILICA_EVAL_IMAGE: Docker image for Basilica (default: ghcr.io/one-covenant/templar-eval:latest)
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from crusades.core.exceptions import EvaluationErrorCode

from .executors.base import EvalConfig
from .executors.basilica_executor import BasilicaExecutor
from .executors.docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


@dataclass
class BasilicaDeploymentContext:
    """Holds state for a reusable Basilica deployment across multiple eval runs."""

    deployment: object  # Basilica deployment handle
    auth_token: str
    url: str
    name: str
    log_file: Path | None = None
    log_stream_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class EvaluationResult:
    """Result from evaluating a miner's submission."""

    success: bool
    mfu: float = 0.0  # Model FLOPs Utilization (primary metric)
    tps: float = 0.0  # Tokens per second (secondary metric)
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    error_code: str | None = None  # Structured error code for reliable error handling
    seed: str = ""
    task_id: int = 0
    diagnostics: dict = field(default_factory=dict)
    code: str | None = None  # Miner's code for storage

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        """Create from dictionary response."""
        return cls(
            success=data.get("success", False),
            mfu=float(data.get("mfu", 0.0)),
            tps=float(data.get("tps", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            wall_time_seconds=float(data.get("wall_time_seconds", 0.0)),
            error=data.get("error"),
            error_code=data.get("error_code"),
            seed=str(data.get("seed", "")),
            task_id=int(data.get("task_id", 0)),
            diagnostics=data.get("diagnostics", {}),
            code=data.get("code"),
        )

    @classmethod
    def failure(
        cls, error: str, task_id: int = 0, error_code: str | None = None
    ) -> "EvaluationResult":
        """Create a failure result."""
        return cls(success=False, error=error, error_code=error_code, task_id=task_id)

    def is_verification_failure(self) -> bool:
        """Check if this result failed due to verification checks."""
        if not self.error_code:
            return False
        try:
            code = EvaluationErrorCode(self.error_code)
            return EvaluationErrorCode.is_verification_failure(code)
        except ValueError:
            return False

    def is_miner_fault(self) -> bool:
        """Check if the error is likely the miner's fault."""
        if self.success:
            return False
        if not self.error_code:
            return True  # Assume miner fault if no code
        try:
            code = EvaluationErrorCode(self.error_code)
            return EvaluationErrorCode.is_miner_fault(code)
        except ValueError:
            return True

    def is_fatal(self) -> bool:
        """Check if this error is fatal/deterministic (no point retrying).

        Fatal errors will fail the same way on every retry because
        they are caused by the miner's code logic, not transient issues.
        """
        if self.success:
            return False
        if not self.error_code:
            return False  # Unknown error, worth retrying
        try:
            code = EvaluationErrorCode(self.error_code)
            return EvaluationErrorCode.is_fatal(code)
        except ValueError:
            return False  # Unknown code, worth retrying


class AffinetesRunner:
    """Runs evaluations via Docker or Basilica.

    URL-Based Architecture:
    - Miner hosts train.py at any URL
    - Validator downloads code from committed URL
    - Code is passed directly to the evaluation container

    Example:
        runner = AffinetesRunner(mode="docker")
        result = await runner.evaluate(
            code="def inner_steps(...): ...",
            seed="12345",
        )
        if result.success:
            print(f"TPS: {result.tps}")
    """

    def __init__(
        self,
        mode: Literal["docker", "basilica"] = "docker",
        basilica_api_key: str | None = None,
        docker_memory_limit: str = "32g",
        docker_shm_size: str = "8g",
        num_gpus: int = 1,
        timeout: int = 600,
        model_url: str | None = None,
        data_url: str | None = None,
        # Verification settings
        max_loss_difference: float = 0.3,
        min_params_changed_ratio: float = 0.8,
        # Weight verification
        weight_relative_error_max: float = 0.008,
        # Timer integrity
        timer_divergence_threshold: float = 0.005,
        # MFU calculation
        gpu_peak_tflops: float = 312.0,
        max_plausible_mfu: float = 75.0,
        min_mfu: float = 50.0,
        validator_image: str | None = None,
        # Basilica-specific settings
        basilica_image: str | None = None,
        basilica_ttl_seconds: int = 3600,
        basilica_gpu_count: int = 1,
        basilica_gpu_models: list[str] | None = None,
        basilica_min_gpu_memory_gb: int = 40,
        basilica_cpu: str = "4",
        basilica_memory: str = "32Gi",
        basilica_interconnect: str | None = None,
        basilica_geo: str | None = None,
        basilica_spot: bool = False,
    ):
        """Initialize the runner.

        Args:
            mode: Execution mode ("docker" for local, "basilica" for remote)
            basilica_api_key: Basilica API key (or BASILICA_API_TOKEN env var)
            docker_memory_limit: Docker memory limit (e.g., "32g")
            docker_shm_size: Shared memory size for Docker (e.g., "8g")
            num_gpus: Number of GPUs to use (0=CPU-only, 1=single GPU, >1=multi-GPU)
            timeout: Evaluation timeout in seconds
            model_url: Default model URL (HuggingFace model ID)
            data_url: Default data URL (HuggingFace dataset)
            max_loss_difference: Max allowed |candidate_loss - reference_loss|
            min_params_changed_ratio: Min % params that must change
            weight_relative_error_max: Max relative error for final
                weight check (e.g., 0.008 = 0.8%)
            timer_divergence_threshold: Max allowed divergence between
                timer sources (e.g., 0.005 = 0.5%)
            gpu_peak_tflops: GPU peak TFLOPS for MFU calculation
            max_plausible_mfu: Reject MFU above this threshold (anti-cheat)
            min_mfu: Reject submissions below this MFU floor
            validator_image: Docker image for local evaluation
            basilica_image: Docker image for Basilica (must be in registry)
            basilica_ttl_seconds: TTL for Basilica deployment (default 1 hour)
            basilica_gpu_count: Number of GPUs (1-8)
            basilica_gpu_models: Acceptable GPU models (e.g., ["A100", "H100"])
            basilica_min_gpu_memory_gb: Minimum GPU memory in GB
            basilica_cpu: CPU limit (e.g., "4")
            basilica_memory: Memory limit (e.g., "32Gi")
        """
        self.mode = mode
        self.default_model_url = model_url
        self.default_data_url = data_url

        # Build shared eval config
        config = EvalConfig(
            timeout=timeout,
            model_url=model_url,
            data_url=data_url,
            max_loss_difference=max_loss_difference,
            min_params_changed_ratio=min_params_changed_ratio,
            weight_relative_error_max=weight_relative_error_max,
            timer_divergence_threshold=timer_divergence_threshold,
            gpu_peak_tflops=gpu_peak_tflops,
            max_plausible_mfu=max_plausible_mfu,
            min_mfu=min_mfu,
        )

        # Create the appropriate executor based on mode
        if mode == "docker":
            self._docker_executor = DockerExecutor(
                config=config,
                docker_memory_limit=docker_memory_limit,
                docker_shm_size=docker_shm_size,
                num_gpus=num_gpus,
                validator_image=validator_image,
            )
            self._basilica_executor: BasilicaExecutor | None = None
            self._executor = self._docker_executor
        elif mode == "basilica":
            self._docker_executor = None
            self._basilica_executor = BasilicaExecutor(
                config=config,
                basilica_api_key=basilica_api_key,
                basilica_image=basilica_image,
                basilica_ttl_seconds=basilica_ttl_seconds,
                basilica_gpu_count=basilica_gpu_count,
                basilica_gpu_models=basilica_gpu_models,
                basilica_min_gpu_memory_gb=basilica_min_gpu_memory_gb,
                basilica_cpu=basilica_cpu,
                basilica_memory=basilica_memory,
                basilica_interconnect=basilica_interconnect,
                basilica_geo=basilica_geo,
                basilica_spot=basilica_spot,
            )
            self._executor = self._basilica_executor
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'docker' or 'basilica'.")

    async def evaluate(
        self,
        code: str,
        seed: str | int = 0,
        model_url: str | None = None,
        data_url: str | None = None,
        steps: int = 5,
        batch_size: int = 8,
        sequence_length: int = 1024,
        data_samples: int = 10000,
        task_id: int = 0,
    ) -> EvaluationResult:
        """Evaluate a miner's train.py code.

        Args:
            code: Miner's train.py code (already downloaded from URL)
            seed: Random seed for evaluation
            model_url: HuggingFace model name
            data_url: HuggingFace dataset name
            steps: Number of training steps
            batch_size: Batch size
            sequence_length: Sequence length
            data_samples: Number of data samples
            task_id: Evaluation task identifier

        Returns:
            EvaluationResult with MFU score
        """
        model_url = model_url or self.default_model_url
        data_url = data_url or self.default_data_url

        if not model_url or not data_url:
            return EvaluationResult.failure(
                "model_url and data_url are required",
                task_id=task_id,
            )

        if not code or "def inner_steps" not in code:
            return EvaluationResult.failure(
                "Invalid code: must contain 'def inner_steps' function",
                task_id=task_id,
            )

        return await self._executor.evaluate(
            code=code,
            seed=str(seed),
            model_url=model_url,
            data_url=data_url,
            steps=steps,
            batch_size=batch_size,
            sequence_length=sequence_length,
            data_samples=data_samples,
            task_id=task_id,
        )

    async def build_validator_image(self, env_path: Path | None = None) -> bool:
        """Build the validator's evaluation Docker image.

        Args:
            env_path: Path to environments/templar directory

        Returns:
            True if build succeeded
        """
        if self._docker_executor is None:
            # Create a temporary DockerExecutor for building
            executor = DockerExecutor(config=EvalConfig())
        else:
            executor = self._docker_executor
        return await executor.build_validator_image(env_path)

    async def create_basilica_deployment(self) -> BasilicaDeploymentContext:
        """Provision a Basilica deployment that can be reused for multiple evals.

        The deployment stays alive until ``destroy_basilica_deployment`` is called.
        Each eval run spawns a fresh torchrun subprocess inside the container,
        so GPU state is fully clean between runs.

        Raises on deployment failure (caller should catch and handle).
        """
        if self._basilica_executor is None:
            raise RuntimeError("create_basilica_deployment requires mode='basilica'")
        return await self._basilica_executor.create_basilica_deployment()

    async def destroy_basilica_deployment(self, ctx: BasilicaDeploymentContext) -> None:
        """Tear down a Basilica deployment and stop log streaming."""
        if self._basilica_executor is None:
            raise RuntimeError("destroy_basilica_deployment requires mode='basilica'")
        await self._basilica_executor.destroy_basilica_deployment(ctx)

    async def evaluate_on_deployment(
        self,
        ctx: BasilicaDeploymentContext,
        code: str,
        seed: str,
        model_url: str | None = None,
        data_url: str | None = None,
        steps: int = 5,
        batch_size: int = 8,
        sequence_length: int = 1024,
        data_samples: int = 10000,
        task_id: int = 0,
    ) -> EvaluationResult:
        """Run a single evaluation on an existing Basilica deployment.

        The container spawns a fresh torchrun subprocess for each call, so GPU
        state is fully clean.  The previous subprocess is killed (if still
        running) before the new one starts -- handled by env.py.
        """
        if self._basilica_executor is None:
            raise RuntimeError("evaluate_on_deployment requires mode='basilica'")
        return await self._basilica_executor.evaluate_on_deployment(
            ctx=ctx,
            code=code,
            seed=seed,
            model_url=model_url,
            data_url=data_url,
            steps=steps,
            batch_size=batch_size,
            sequence_length=sequence_length,
            data_samples=data_samples,
            task_id=task_id,
        )


def create_runner(
    mode: str = "docker",
    **kwargs,
) -> AffinetesRunner:
    """Factory function to create an AffinetesRunner.

    Args:
        mode: "docker" or "basilica"
        **kwargs: Additional arguments

    Returns:
        Configured AffinetesRunner
    """
    if mode == "basilica":
        kwargs.setdefault("basilica_api_key", os.getenv("BASILICA_API_TOKEN"))

    return AffinetesRunner(mode=mode, **kwargs)
