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
import json
import logging
import math
import os
import secrets
import subprocess
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx

from crusades.core.exceptions import EvaluationErrorCode

# Optional: Basilica SDK for cloud GPU evaluation
try:
    from basilica import BasilicaClient

    BASILICA_AVAILABLE = True
except ImportError:
    BasilicaClient = None
    BASILICA_AVAILABLE = False

logger = logging.getLogger(__name__)


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

    # Default Docker image for local evaluation
    DEFAULT_DOCKER_IMAGE = os.getenv("VALIDATOR_EVAL_IMAGE", "templar-eval:latest")

    # Default Basilica image (must be pushed to registry like ghcr.io)
    DEFAULT_BASILICA_IMAGE = os.getenv(
        "BASILICA_EVAL_IMAGE", "ghcr.io/one-covenant/templar-eval:latest"
    )

    _INTERNAL_NETWORK_PREFIX = "crusades_nccl_"

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
            weight_relative_error_max: Max relative error for final weight check (e.g., 0.008 = 0.8%)
            timer_divergence_threshold: Max allowed divergence between timer sources (e.g., 0.005 = 0.5%)
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
        self.basilica_api_key = basilica_api_key or os.getenv("BASILICA_API_TOKEN")
        self.docker_memory_limit = docker_memory_limit
        self.docker_shm_size = docker_shm_size
        self.num_gpus = num_gpus
        self.timeout = timeout
        self.default_model_url = model_url
        self.default_data_url = data_url
        # Verification settings
        self.max_loss_difference = max_loss_difference
        self.min_params_changed_ratio = min_params_changed_ratio
        # Weight verification
        self.weight_relative_error_max = weight_relative_error_max
        # Timer integrity
        self.timer_divergence_threshold = timer_divergence_threshold
        # MFU calculation
        self.gpu_peak_tflops = gpu_peak_tflops
        self.max_plausible_mfu = max_plausible_mfu
        self.min_mfu = min_mfu
        self.validator_image = validator_image or self.DEFAULT_DOCKER_IMAGE
        self.basilica_image = basilica_image or self.DEFAULT_BASILICA_IMAGE
        self.basilica_ttl_seconds = basilica_ttl_seconds
        self.basilica_gpu_count = basilica_gpu_count
        self.basilica_gpu_models = basilica_gpu_models or ["A100", "H100"]
        self.basilica_min_gpu_memory_gb = basilica_min_gpu_memory_gb
        self.basilica_cpu = basilica_cpu
        self.basilica_memory = basilica_memory
        self.basilica_interconnect = basilica_interconnect
        self.basilica_geo = basilica_geo
        self.basilica_spot = basilica_spot

        # Basilica deployment cache (per-instance, not global)
        self._basilica_deployment = None
        self._basilica_deployment_time: float = 0

        if mode == "basilica":
            if not self.basilica_api_key:
                logger.warning("Basilica mode: BASILICA_API_TOKEN not set")
            logger.info("Basilica mode initialized")
            logger.info(f"   Image: {self.basilica_image}")
            logger.info(f"   TTL: {self.basilica_ttl_seconds}s")
            logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
            logger.info(f"   Min GPU Memory: {self.basilica_min_gpu_memory_gb}GB")
            logger.info(f"   CPU/Memory: {self.basilica_cpu} / {self.basilica_memory}")

    @classmethod
    def _create_eval_network(cls) -> str | None:
        """Create a per-evaluation internal Docker network for multi-GPU NCCL.

        Each evaluation gets its own isolated network to prevent cross-container
        communication between concurrent evaluations.  Returns the network name
        on success, ``None`` on failure (falls back to ``--network none``).
        """
        import uuid

        network_name = f"{cls._INTERNAL_NETWORK_PREFIX}{uuid.uuid4().hex[:12]}"
        try:
            result = subprocess.run(
                ["docker", "network", "create", "--internal", network_name],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.error(
                    f"Failed to create Docker network '{network_name}': "
                    f"{result.stderr.decode().strip()}"
                )
                return None
            return network_name
        except Exception as e:
            logger.error(f"Error creating Docker network '{network_name}': {e}")
            return None

    @staticmethod
    def _remove_eval_network(network_name: str) -> None:
        """Remove a per-evaluation Docker network (best-effort)."""
        try:
            subprocess.run(
                ["docker", "network", "rm", network_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

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

        if self.mode == "docker":
            return await self._evaluate_docker(
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
        elif self.mode == "basilica":
            return await self._evaluate_basilica(
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
        else:
            return EvaluationResult.failure(
                f"Unknown mode: {self.mode}",
                task_id=task_id,
            )

    async def _evaluate_docker(
        self,
        code: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation locally using Docker.

        Code is mounted directly into the container - no downloads needed.
        """
        logger.info("Running Docker evaluation")
        logger.info(f"   Code size: {len(code)} bytes")

        # Check if validator image exists
        check_cmd = ["docker", "image", "inspect", self.validator_image]
        check_result = subprocess.run(check_cmd, capture_output=True)

        if check_result.returncode != 0:
            return EvaluationResult.failure(
                f"Validator image not found: {self.validator_image}. "
                f"Build it first: cd environments/templar && docker build -t {self.validator_image} .",
                task_id=task_id,
            )

        # Write miner's code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            prefix="train_",
        ) as f:
            f.write(code)
            train_path = f.name

        # Make readable by container's non-root user
        os.chmod(train_path, 0o644)

        # Create evaluation script that reads code from mounted file
        eval_script = f'''
import asyncio
import json
import os
import sys
sys.path.insert(0, '/app')

from env import Actor

async def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Read miner's code
    with open('/app/scripts/miner_train.py') as f:
        code = f.read()

    actor = Actor()
    result = await actor.evaluate(
        task_id={task_id},
        seed="{seed}",
        model_url="{model_url}",
        data_url="{data_url}",
        steps={steps},
        batch_size={batch_size},
        sequence_length={sequence_length},
        data_samples={data_samples},
        timeout={self.timeout},
        code=code,
        max_loss_difference={self.max_loss_difference},
        use_random_init=True,
        min_trainable_params_ratio=1.0,
        min_params_changed_ratio={self.min_params_changed_ratio},
        # Weight verification
        weight_relative_error_max={self.weight_relative_error_max},
        # Timer integrity
        timer_divergence_threshold={self.timer_divergence_threshold},
        # MFU calculation
        gpu_peak_tflops={self.gpu_peak_tflops},
        max_plausible_mfu={self.max_plausible_mfu},
        min_mfu={self.min_mfu},
        require_cuda_timing=True,
        num_gpus={self.num_gpus},
    )
    if local_rank == 0:
        print("EVAL_RESULT:" + json.dumps(result))

asyncio.run(main())
'''

        # Write eval script to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(eval_script)
            script_path = f.name

        # Make readable by container's non-root user
        os.chmod(script_path, 0o644)

        try:
            # Build Docker run command
            # NOTE: Mount to /app/scripts/ (not /tmp/) because we use --tmpfs on /tmp
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{script_path}:/app/scripts/eval_script.py:ro",
                "-v",
                f"{train_path}:/app/scripts/miner_train.py:ro",
            ]

            if self.num_gpus > 0:
                docker_cmd.extend(["--gpus", str(self.num_gpus)])

            # Memory limits; scale shm for multi-GPU NCCL
            shm_size = self.docker_shm_size
            if self.num_gpus > 1:
                base_shm_gb = int(self.docker_shm_size.rstrip("gG"))
                shm_size = f"{max(base_shm_gb, 2 * self.num_gpus)}g"
            docker_cmd.extend(
                [
                    "--memory",
                    self.docker_memory_limit,
                    "--shm-size",
                    shm_size,
                ]
            )

            # Sandbox: scale pids-limit for torchrun
            pids_limit = 1024 * max(self.num_gpus, 1)
            eval_network = None
            if self.num_gpus <= 1:
                docker_cmd.extend(["--network", "none"])
            else:
                eval_network = self._create_eval_network()
                if eval_network is not None:
                    docker_cmd.extend(["--network", eval_network, "--dns", "0.0.0.0"])
                else:
                    docker_cmd.extend(["--network", "none"])
            docker_cmd.extend(
                [
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--read-only",
                    "--pids-limit",
                    str(pids_limit),
                    # Writable /tmp for temporary files (exec needed for torch.compile)
                    "--tmpfs",
                    "/tmp:rw,exec,nosuid,size=4g",
                    # Writable Triton cache for torch.compile kernels (exec needed for .so files)
                    "--tmpfs",
                    "/home/appuser/.triton:rw,exec,size=2g",
                    # NOTE: Don't mount tmpfs on ~/.cache/huggingface - model is pre-cached there!
                ]
            )

            # Timeout
            docker_cmd.extend(
                [
                    "--stop-timeout",
                    str(self.timeout),
                ]
            )

            # Image and command
            if self.num_gpus > 1:
                master_port = 29500 + secrets.randbelow(10001)
                docker_cmd.extend(
                    [
                        self.validator_image,
                        "torchrun",
                        "--nproc_per_node",
                        str(self.num_gpus),
                        "--master_port",
                        str(master_port),
                        "/app/scripts/eval_script.py",
                    ]
                )
            else:
                docker_cmd.extend(
                    [
                        self.validator_image,
                        "python",
                        "/app/scripts/eval_script.py",
                    ]
                )

            logger.info(f"Running evaluation in {self.validator_image}...")
            logger.debug(f"   Full Docker command: {' '.join(docker_cmd)}")
            logger.info(f"   Docker command: {' '.join(docker_cmd[:6])}...")

            # Run with timeout - stream logs in real-time
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout for unified logging
            )

            # Stream logs in real-time and collect for parsing
            # Use chunk-based reading to handle long lines (progress bars, etc.)
            stdout_lines = []

            def _should_log(line: str) -> bool:
                """Filter out noisy logs, only show important ones."""
                # Always skip these
                if not line or line.startswith("EVAL_RESULT:"):
                    return False
                # Skip HTTP request logs
                if "HTTP Request:" in line or "httpx" in line:
                    return False
                # Skip progress bars
                if "Loading weights:" in line or "Fetching" in line:
                    return False
                # Skip deprecation/warning noise
                if "is deprecated" in line or "UserWarning" in line:
                    return False
                # Skip HuggingFace download noise
                if "huggingface" in line.lower() and "INFO" in line:
                    return False
                # Always show important logs
                if any(
                    kw in line
                    for kw in [
                        "VERIFICATION",
                        "CHECK",
                        "PASSED",
                        "FAILED",
                        "ERROR",
                        "error",
                        "Exception",
                        "Traceback",
                        "env |",  # env.py logs
                    ]
                ):
                    return True
                # Show other logs at debug level only
                return False

            try:

                async def read_stream():
                    buffer = ""
                    while True:
                        # Read in chunks to avoid buffer limit issues with long lines
                        chunk = await asyncio.wait_for(
                            process.stdout.read(8192),  # 8KB chunks
                            timeout=self.timeout + 60,
                        )
                        if not chunk:
                            # Process remaining buffer at end
                            if buffer:
                                for line in buffer.split("\n"):
                                    line = line.rstrip()
                                    if line:
                                        stdout_lines.append(line)
                                        if _should_log(line):
                                            logger.info(f"   [DOCKER] {line}")
                            break

                        buffer += chunk.decode()
                        # Process complete lines as they arrive
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip()
                            stdout_lines.append(line)
                            if _should_log(line):
                                logger.info(f"   [DOCKER] {line}")

                await read_stream()
                await process.wait()

            except TimeoutError:
                process.kill()
                return EvaluationResult.failure(
                    f"Evaluation timed out after {self.timeout}s",
                    task_id=task_id,
                )

            stdout_text = "\n".join(stdout_lines)

            if process.returncode != 0:
                # Log the output for debugging
                logger.error(f"Docker container failed with exit code {process.returncode}")
                if stdout_text:
                    logger.error(f"Docker output: {stdout_text[:500]}")
                return EvaluationResult.failure(
                    f"Container failed with exit code: {process.returncode}. Output: {stdout_text[:200]}",
                    task_id=task_id,
                )

            # Parse result
            for line in stdout_text.split("\n"):
                if line.startswith("EVAL_RESULT:"):
                    result_json = line[len("EVAL_RESULT:") :]
                    try:
                        result_data = json.loads(result_json)
                        result = EvaluationResult.from_dict(result_data)
                        result.code = code  # Include code in result
                        return result
                    except json.JSONDecodeError as e:
                        return EvaluationResult.failure(
                            f"Invalid result JSON: {e}",
                            task_id=task_id,
                        )

            return EvaluationResult.failure(
                f"No evaluation result in output. stdout: {stdout_text[:200]}",
                task_id=task_id,
            )

        finally:
            try:
                os.unlink(script_path)
                os.unlink(train_path)
            except Exception:
                pass
            if eval_network is not None:
                self._remove_eval_network(eval_network)

    async def _evaluate_basilica(
        self,
        code: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation remotely via Basilica SDK.

        Uses BasilicaClient to deploy a custom Docker image and call
        the /evaluate endpoint for MFU evaluation.

        Flow:
        1. Deploy image to Basilica (or reuse existing deployment)
        2. Wait for deployment to be ready (/health endpoint)
        3. POST to /evaluate with miner's code
        4. Return MFU results
        """
        logger.info("=" * 60)
        logger.info("[BASILICA] Starting remote GPU evaluation")
        logger.info("=" * 60)
        logger.info("[BASILICA] Configuration:")
        logger.info(f"   Image: {self.basilica_image}")
        logger.info(f"   Model: {model_url}")
        logger.info(f"   Dataset: {data_url}")
        logger.info(f"   Steps: {steps}, Batch size: {batch_size}")
        logger.info(f"   Task ID: {task_id}, Seed: {seed}")
        logger.info(f"   Code size: {len(code)} bytes")

        if not BASILICA_AVAILABLE:
            logger.error("[BASILICA] SDK not installed!")
            return EvaluationResult.failure(
                "basilica SDK not installed. Run: uv add basilica",
                task_id=task_id,
            )

        try:
            # Get or create Basilica deployment (cleaned up by validator.py after all runs)
            logger.info("[BASILICA] Acquiring deployment...")
            deployment = await self._get_basilica_deployment()

            if deployment is None:
                logger.error("[BASILICA] Failed to create deployment!")
                return EvaluationResult.failure(
                    "Failed to create Basilica deployment",
                    task_id=task_id,
                )

            logger.info("-" * 60)
            logger.info("[BASILICA] Deployment ready!")
            logger.info(f"   URL: {deployment.url}")
            if hasattr(deployment, "id"):
                logger.info(f"   Deployment ID: {deployment.id}")
            logger.info("-" * 60)

            # Check health endpoint first
            logger.info("[BASILICA] Checking health endpoint...")
            async with httpx.AsyncClient(timeout=30) as client:
                try:
                    health_response = await client.get(f"{deployment.url}/health")
                    if health_response.status_code == 200:
                        logger.info("[BASILICA] Health check: OK")
                    else:
                        logger.warning(f"[BASILICA] Health check: {health_response.status_code}")
                except Exception as e:
                    logger.warning(f"[BASILICA] Health check failed: {e}")

            # Call the /evaluate endpoint
            payload = {
                "task_id": task_id,
                "seed": seed,
                "model_url": model_url,
                "data_url": data_url,
                "steps": steps,
                "batch_size": batch_size,
                "timeout": self.timeout,
                "sequence_length": sequence_length,
                "data_samples": data_samples,
                "code": code,
                "max_loss_difference": self.max_loss_difference,
                "use_random_init": True,
                "min_trainable_params_ratio": 1.0,
                "min_params_changed_ratio": self.min_params_changed_ratio,
                # Weight verification
                "weight_relative_error_max": self.weight_relative_error_max,
                # Timer integrity
                "timer_divergence_threshold": self.timer_divergence_threshold,
                # MFU calculation
                "gpu_peak_tflops": self.gpu_peak_tflops,
                "max_plausible_mfu": self.max_plausible_mfu,
                "min_mfu": self.min_mfu,
                "num_gpus": self.basilica_gpu_count,
            }

            logger.info("[BASILICA] Sending evaluation request...")
            logger.info(f"   POST {deployment.url}/evaluate")
            logger.info(f"   Timeout: {self.timeout + 600}s")

            start_time = time.time()

            async with httpx.AsyncClient(timeout=self.timeout + 600) as client:
                response = await client.post(
                    f"{deployment.url}/evaluate",
                    json=payload,
                )

                elapsed = time.time() - start_time
                logger.info(f"[BASILICA] Response received in {elapsed:.1f}s")
                logger.info(f"   Status code: {response.status_code}")

                if response.status_code != 200:
                    error_text = response.text[:500]
                    logger.error("[BASILICA] Evaluation failed!")
                    logger.error(f"   Error: {error_text}")
                    return EvaluationResult.failure(
                        f"Basilica /evaluate error: {response.status_code} - {error_text}",
                        task_id=task_id,
                    )

                result_data = response.json()
                if not isinstance(result_data, dict):
                    return EvaluationResult.failure(
                        f"Basilica returned non-object response: {type(result_data).__name__}",
                        task_id=task_id,
                    )

                # Strict response integrity checks
                returned_seed = str(result_data.get("seed", ""))
                if returned_seed and returned_seed != seed:
                    return EvaluationResult.failure(
                        f"Basilica response seed mismatch: expected {seed}, got {returned_seed}",
                        task_id=task_id,
                    )

                returned_task_id = result_data.get("task_id")
                if returned_task_id is not None and int(returned_task_id) != int(task_id):
                    return EvaluationResult.failure(
                        f"Basilica response task_id mismatch: expected {task_id}, got {returned_task_id}",
                        task_id=task_id,
                    )

                for numeric_field in ("mfu", "tps", "wall_time_seconds"):
                    if numeric_field in result_data:
                        value = float(result_data[numeric_field])
                        if not math.isfinite(value):
                            return EvaluationResult.failure(
                                f"Basilica response has non-finite {numeric_field}",
                                task_id=task_id,
                            )

                result = EvaluationResult.from_dict(result_data)
                result.code = code

                logger.info("=" * 60)
                logger.info("[BASILICA] Evaluation complete!")
                logger.info(f"   Success: {result.success}")
                logger.info(f"   MFU: {result.mfu:.2f}%")
                logger.info(f"   TPS: {result.tps:,.2f} tokens/second")
                logger.info(f"   Total tokens: {result.total_tokens:,}")
                logger.info(f"   Wall time: {result.wall_time_seconds:.2f}s")
                if result.diagnostics:
                    logger.info(f"   Diagnostics: {result.diagnostics}")
                if result.error:
                    logger.error(f"   Error: {result.error}")
                logger.info("=" * 60)

                return result

        except TimeoutError:
            logger.error(f"[BASILICA] Timeout after {self.timeout}s!")
            return EvaluationResult.failure(
                f"Basilica timeout after {self.timeout}s",
                task_id=task_id,
            )
        except Exception as e:
            logger.error(f"[BASILICA] Error: {e}")
            logger.error(traceback.format_exc())
            return EvaluationResult.failure(
                f"Basilica error: {e}",
                task_id=task_id,
            )

    async def _get_basilica_deployment(self):
        """Get or create a Basilica deployment.

        Reuses existing deployment if within TTL, otherwise creates new one.
        Always cleans up stale/expired deployments before creating fresh ones.
        """
        now = time.time()
        ttl_buffer = 300  # 5 minute buffer before TTL expires

        if (
            self._basilica_deployment is not None
            and now - self._basilica_deployment_time < self.basilica_ttl_seconds - ttl_buffer
        ):
            remaining = self.basilica_ttl_seconds - (now - self._basilica_deployment_time)
            logger.info("[BASILICA] Reusing existing deployment")
            logger.info(f"   URL: {self._basilica_deployment.url}")
            logger.info(f"   TTL remaining: {remaining:.0f}s ({remaining / 60:.1f} min)")
            return self._basilica_deployment

        # Delete stale cached deployment before creating a new one
        if self._basilica_deployment is not None:
            logger.info("[BASILICA] Deleting expired cached deployment before creating new one")
            await self.delete_basilica_deployment()

        logger.info("[BASILICA] Creating NEW deployment")
        logger.info(f"   Image: {self.basilica_image}")
        logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
        logger.info(f"   Min GPU memory: {self.basilica_min_gpu_memory_gb}GB")
        if self.basilica_interconnect:
            logger.info(f"   Interconnect: {self.basilica_interconnect}")
        logger.info(f"   CPU: {self.basilica_cpu}, Memory: {self.basilica_memory}")
        if self.basilica_geo:
            logger.info(f"   Geo: {self.basilica_geo}")
        if self.basilica_spot:
            logger.info("   Spot: enabled")
        logger.info(
            f"   TTL: {self.basilica_ttl_seconds}s ({self.basilica_ttl_seconds / 60:.0f} min)"
        )
        logger.info("[BASILICA] Requesting GPU from Basilica... (this may take 2-5 minutes)")

        deploy_name = f"templar-eval-{uuid.uuid4().hex[:8]}"
        logger.info(f"   Deployment name: {deploy_name}")

        deployment = None
        try:
            deploy_start = time.time()
            client = BasilicaClient()

            # Split into create + get + wait (instead of client.deploy()) so
            # we always hold a Deployment reference for cleanup on failure.
            # All calls use async variants to avoid blocking the event loop.
            deploy_kwargs = {
                "instance_name": deploy_name,
                "image": self.basilica_image,
                "port": 8000,
                "ttl_seconds": self.basilica_ttl_seconds,
                "gpu_count": self.basilica_gpu_count,
                "gpu_models": self.basilica_gpu_models,
                "min_gpu_memory_gb": self.basilica_min_gpu_memory_gb,
                "cpu": self.basilica_cpu,
                "memory": self.basilica_memory,
            }
            if self.basilica_interconnect:
                deploy_kwargs["interconnect"] = self.basilica_interconnect
            if self.basilica_geo:
                deploy_kwargs["geo"] = self.basilica_geo
            if self.basilica_spot:
                deploy_kwargs["spot"] = self.basilica_spot
            response = await client.create_deployment_async(**deploy_kwargs)
            deployment = await client.get_async(response.instance_name)
            logger.info(f"   Basilica ID: {deployment.name}")

            await deployment.wait_until_ready_async(timeout=self.timeout)
            await deployment.refresh_async()

            deploy_time = time.time() - deploy_start
            logger.info(f"[BASILICA] Deployment ready in {deploy_time:.1f}s")
            logger.info(f"   Deployment URL: {deployment.url}")
            logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
            logger.info(
                f"   TTL: {self.basilica_ttl_seconds}s (expires in {self.basilica_ttl_seconds / 60:.0f} min)"
            )

            self._basilica_deployment = deployment
            self._basilica_deployment_time = time.time()

            return deployment

        except Exception as e:
            logger.error("[BASILICA] Failed to deploy!")
            logger.error(f"   Error: {e}")
            logger.error(traceback.format_exc())
            if deployment is not None:
                try:
                    await deployment.delete_async()
                    logger.info(f"[BASILICA] Cleaned up failed deployment '{deployment.name}'")
                except Exception as del_err:
                    logger.warning(f"[BASILICA] Failed to clean up deployment: {del_err}")
            return None

    async def delete_basilica_deployment(self) -> None:
        """Delete the current Basilica deployment to free resources.

        Called by validator.py in a finally block after all evaluation
        runs for a submission complete (not between individual runs).
        Also called internally when a cached deployment has expired.
        """
        if self._basilica_deployment is None:
            return

        name = getattr(self._basilica_deployment, "name", "unknown")
        try:
            await self._basilica_deployment.delete_async()
            logger.info(f"[BASILICA] Deployment '{name}' deleted")
        except Exception as e:
            logger.warning(f"[BASILICA] Failed to delete deployment '{name}': {e}")

        self._basilica_deployment = None
        self._basilica_deployment_time = 0

    async def build_validator_image(self, env_path: Path | None = None) -> bool:
        """Build the validator's evaluation Docker image.

        Args:
            env_path: Path to environments/templar directory

        Returns:
            True if build succeeded
        """
        if env_path is None:
            candidates = [
                Path(__file__).parent.parent.parent.parent / "environments" / "templar",
                Path.cwd() / "environments" / "templar",
            ]
            for candidate in candidates:
                if candidate.exists() and (candidate / "Dockerfile").exists():
                    env_path = candidate
                    break

        if env_path is None or not env_path.exists():
            logger.error("Could not find environments/templar directory")
            return False

        logger.info(f"Building validator image: {self.validator_image}")
        logger.info(f"   From: {env_path}")

        cmd = [
            "docker",
            "build",
            "-t",
            self.validator_image,
            str(env_path),
        ]

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            logger.error("Failed to build validator image")
            return False

        logger.info(f"Successfully built: {self.validator_image}")
        return True


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
