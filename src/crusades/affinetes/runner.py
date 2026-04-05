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
    from basilica import BasilicaClient, DeploymentFailed, DeploymentTimeout

    BASILICA_AVAILABLE = True
except ImportError:
    BasilicaClient = None
    DeploymentFailed = None
    DeploymentTimeout = None
    BASILICA_AVAILABLE = False

logger = logging.getLogger(__name__)

_DOCKER_CONTAINER_PREFIX = "crusades-eval-"


def cleanup_stale_eval_containers() -> int:
    """Kill and remove any leftover evaluation containers from previous sessions.

    Call on validator startup to reclaim GPUs before launching new evals.
    Returns the number of containers cleaned up.
    """
    cleaned = 0
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={_DOCKER_CONTAINER_PREFIX}",
                "--format",
                "{{.ID}} {{.Names}} {{.Status}}",
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            return 0

        for line in result.stdout.decode().strip().splitlines():
            if not line:
                continue
            parts = line.split(None, 2)
            container_id = parts[0]
            container_name = parts[1] if len(parts) > 1 else container_id
            logger.warning(f"Cleaning up stale eval container: {container_name} ({container_id})")
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=30,
            )
            cleaned += 1
    except Exception as e:
        logger.warning(f"Error during stale container cleanup: {e}")

    if cleaned:
        logger.info(f"Cleaned up {cleaned} stale evaluation container(s)")

    # Also prune orphaned evaluation networks
    try:
        result = subprocess.run(
            [
                "docker",
                "network",
                "ls",
                "--filter",
                f"name={AffinetesRunner._INTERNAL_NETWORK_PREFIX}",
                "--format",
                "{{.Name}}",
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            for net_name in result.stdout.decode().strip().splitlines():
                if net_name:
                    subprocess.run(
                        ["docker", "network", "rm", net_name],
                        capture_output=True,
                        timeout=10,
                    )
                    logger.info(f"Removed orphaned eval network: {net_name}")
    except Exception:
        pass

    return cleaned


def _force_remove_container(container_name: str) -> None:
    """Force-stop and remove a Docker container by name (best-effort)."""
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass


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

    # Default Docker image for local evaluation
    DEFAULT_DOCKER_IMAGE = os.getenv("VALIDATOR_EVAL_IMAGE", "templar-eval:latest")

    # Default Basilica image (must be pushed to registry like ghcr.io)
    DEFAULT_BASILICA_IMAGE = os.getenv(
        "BASILICA_EVAL_IMAGE", "ghcr.io/one-covenant/templar-eval:latest"
    )

    _INTERNAL_NETWORK_PREFIX = "crusades_nccl_"

    _current_docker_container: str | None = None

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

        container_name = f"{_DOCKER_CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"

        try:
            # Build Docker run command
            # NOTE: Mount to /app/scripts/ (not /tmp/) because we use --tmpfs on /tmp
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
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

            # Grace period before SIGKILL when docker stop is called (not eval timeout)
            docker_cmd.extend(
                [
                    "--stop-timeout",
                    "30",
                ]
            )

            # NCCL bandwidth throttling: disable P2P (NVLink) and NVLink SHARP
            # to simulate PCIe-like interconnect (~32 GB/s vs ~600 GB/s).
            # This makes collective-heavy strategies (FSDP, TP) less efficient,
            # incentivizing pipeline parallelism which only sends activations
            # between stages.
            if self.num_gpus > 1:
                docker_cmd.extend(
                    [
                        "-e",
                        "NCCL_P2P_DISABLE=1",
                        "-e",
                        "NCCL_NVLS_ENABLE=0",
                        "-e",
                        "NCCL_SHM_USE_CUDA_MEMCPY=0",
                        "-e",
                        "NCCL_IB_DISABLE=1",
                        "-e",
                        "NCCL_MAX_NCHANNELS=1",
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
            AffinetesRunner._current_docker_container = container_name
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
                logger.warning(
                    f"Evaluation timed out after {self.timeout}s, "
                    f"force-removing container {container_name}"
                )
                _force_remove_container(container_name)
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
            AffinetesRunner._current_docker_container = None
            _force_remove_container(container_name)
            try:
                os.unlink(script_path)
                os.unlink(train_path)
            except Exception:
                pass
            if eval_network is not None:
                self._remove_eval_network(eval_network)

    @classmethod
    def stop_current_docker_eval(cls) -> None:
        """Force-stop the currently running Docker eval container, if any.

        Safe to call from signal handlers or shutdown hooks.
        """
        container = cls._current_docker_container
        if container:
            logger.warning(f"Stopping active eval container on shutdown: {container}")
            _force_remove_container(container)
            cls._current_docker_container = None

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

        Uses BasilicaClient.deploy_async to provision a GPU container and call
        the /evaluate endpoint for MFU evaluation. Each call creates a fresh
        deployment and deletes it when done, ensuring clean GPU state.

        Flow:
        1. deploy_async: create deployment, wait until DNS + HTTP ready
        2. POST /evaluate → get job_id back (instant, no proxy timeout)
        3. Poll GET /eval-status/{job_id} every 30s until done
        4. Delete deployment (in finally block)
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

        deployment = None
        log_stream_task = None
        start_time = time.time()
        http_timeout = max(self.timeout + 600, 1800)
        deploy_name = f"templar-eval-{uuid.uuid4().hex[:8]}"
        try:
            logger.info("[BASILICA] Creating fresh deployment for this run...")
            logger.info(f"   Image: {self.basilica_image}")
            logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
            logger.info(f"   Min GPU memory: {self.basilica_min_gpu_memory_gb}GB")
            if self.basilica_interconnect:
                logger.info(f"   Interconnect: {self.basilica_interconnect}")
            logger.info(f"   CPU: {self.basilica_cpu}, Memory: {self.basilica_memory}")
            logger.info(
                f"   TTL: {self.basilica_ttl_seconds}s ({self.basilica_ttl_seconds / 60:.0f} min)"
            )
            logger.info(f"   Deployment name: {deploy_name}")

            auth_token = secrets.token_urlsafe(32)
            logger.info("[BASILICA] Auth token generated for this deployment (EVAL_AUTH_TOKEN set)")
            client = BasilicaClient()
            deploy_kwargs = {
                "name": deploy_name,
                "image": self.basilica_image,
                "port": 8000,
                "ttl_seconds": self.basilica_ttl_seconds,
                "gpu_count": self.basilica_gpu_count,
                "gpu_models": self.basilica_gpu_models,
                "min_gpu_memory_gb": self.basilica_min_gpu_memory_gb,
                "cpu": self.basilica_cpu,
                "memory": self.basilica_memory,
                "timeout": 1800,
                "env": {"EVAL_AUTH_TOKEN": auth_token},
            }
            if self.basilica_interconnect:
                deploy_kwargs["interconnect"] = self.basilica_interconnect
            if self.basilica_geo:
                deploy_kwargs["geo"] = self.basilica_geo
            if self.basilica_spot:
                deploy_kwargs["spot"] = self.basilica_spot

            deploy_start = time.time()
            deployment = await client.deploy_async(**deploy_kwargs)
            deploy_time = time.time() - deploy_start

            dep_name = getattr(deployment, "name", None) or getattr(deployment, "id", "unknown")
            logger.info(f"[BASILICA] Deployment ready in {deploy_time:.1f}s")
            logger.info(f"   URL: {deployment.url}")
            logger.info(f"   Deployment ID: {dep_name}")
            logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")

            log_dir = Path("logs/basilica")
            log_file = log_dir / f"{dep_name}_{int(time.time())}.log"
            log_stream_task = asyncio.create_task(self._stream_basilica_logs(deployment, log_file))
            logger.info(f"[BASILICA] Log streaming started → {log_file}")

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
                "weight_relative_error_max": self.weight_relative_error_max,
                "timer_divergence_threshold": self.timer_divergence_threshold,
                "gpu_peak_tflops": self.gpu_peak_tflops,
                "max_plausible_mfu": self.max_plausible_mfu,
                "min_mfu": self.min_mfu,
                "num_gpus": self.basilica_gpu_count,
            }

            logger.info("[BASILICA] Sending evaluation request (async)...")
            logger.info(f"   POST {deployment.url}/evaluate")
            logger.info(f"   Timeout budget: {http_timeout}s")
            logger.info(f"   Payload size: {len(str(payload))} chars")

            post_start = time.time()

            # ── Step 1: Submit job (returns 202 + job_id immediately) ──
            auth_headers = {"Authorization": f"Bearer {auth_token}"}
            async with httpx.AsyncClient(timeout=60, headers=auth_headers) as submit_client:
                submit_resp = await submit_client.post(f"{deployment.url}/evaluate", json=payload)

            if submit_resp.status_code not in (200, 202):
                error_text = submit_resp.text[:500]
                logger.error(f"[BASILICA] Job submission failed: {submit_resp.status_code}")
                logger.error(f"   Error: {error_text}")
                return EvaluationResult.failure(
                    f"Basilica /evaluate submission error: {submit_resp.status_code} - {error_text}",
                    task_id=task_id,
                )

            submit_data = submit_resp.json()
            job_id = submit_data.get("job_id")
            if not job_id:
                return EvaluationResult.failure(
                    f"Basilica /evaluate did not return job_id: {submit_data}",
                    task_id=task_id,
                )

            logger.info(f"[BASILICA] Job accepted: {job_id}")

            # ── Step 2: Poll /eval-status/{job_id} until done ──
            poll_url = f"{deployment.url}/eval-status/{job_id}"
            poll_interval = 30
            consecutive_errors = 0
            poll_timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)

            async with httpx.AsyncClient(timeout=poll_timeout, headers=auth_headers) as poll_client:
                while True:
                    elapsed = time.time() - post_start
                    if elapsed >= http_timeout:
                        logger.error(
                            f"[BASILICA] Polling timeout after {elapsed:.0f}s (budget: {http_timeout}s)"
                        )
                        return EvaluationResult.failure(
                            f"Basilica evaluation timed out after {elapsed:.0f}s (polling)",
                            task_id=task_id,
                        )

                    remaining = http_timeout - elapsed
                    max_consecutive_errors = max(
                        10, int(remaining // (poll_interval + poll_timeout.read))
                    )

                    await asyncio.sleep(poll_interval)
                    elapsed = time.time() - post_start

                    try:
                        poll_resp = await poll_client.get(poll_url)

                        consecutive_errors = 0

                        if poll_resp.status_code == 404:
                            logger.warning(
                                f"[BASILICA] Job {job_id} not found (404) — "
                                f"container may have restarted"
                            )
                            return EvaluationResult.failure(
                                f"Basilica job {job_id} lost (404 on poll)",
                                task_id=task_id,
                            )

                        if poll_resp.status_code != 200:
                            logger.warning(
                                f"[BASILICA] Poll got {poll_resp.status_code}, "
                                f"will retry in {poll_interval}s ({elapsed:.0f}s elapsed)"
                            )
                            continue

                        poll_data = poll_resp.json()
                        status = poll_data.get("status")

                        if status == "pending":
                            logger.info(
                                f"   [POLLING] Evaluation in progress... {elapsed:.0f}s elapsed"
                            )
                            continue

                        if status in ("done", "failed"):
                            result_data = poll_data.get("result")
                            if result_data is None:
                                return EvaluationResult.failure(
                                    f"Basilica job {job_id} status={status} but no result",
                                    task_id=task_id,
                                )
                            logger.info(
                                f"[BASILICA] Job {job_id} completed with status={status} "
                                f"in {elapsed:.1f}s"
                            )
                            break

                        logger.warning(
                            f"[BASILICA] Unknown poll status: {status}, will retry in {poll_interval}s"
                        )

                    except (
                        httpx.TimeoutException,
                        httpx.ConnectError,
                        httpx.RemoteProtocolError,
                    ) as poll_err:
                        consecutive_errors += 1
                        logger.warning(
                            f"[BASILICA] Poll error ({consecutive_errors}/{max_consecutive_errors}): "
                            f"{type(poll_err).__name__} ({elapsed:.0f}s elapsed)"
                        )
                        if consecutive_errors >= max_consecutive_errors:
                            return EvaluationResult.failure(
                                f"Basilica polling failed {max_consecutive_errors} consecutive "
                                f"times after {elapsed:.0f}s: {poll_err}",
                                task_id=task_id,
                            )
                        continue

            elapsed = time.time() - post_start
            logger.info(f"[BASILICA] Evaluation result received in {elapsed:.1f}s")
            if not isinstance(result_data, dict):
                return EvaluationResult.failure(
                    f"Basilica returned non-object response: {type(result_data).__name__}",
                    task_id=task_id,
                )

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

        except DeploymentTimeout as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Deployment timed out after {elapsed:.0f}s!")
            logger.error(f"   {e}")
            return EvaluationResult.failure(
                f"Basilica deployment timeout after {elapsed:.0f}s: {e}",
                task_id=task_id,
            )
        except DeploymentFailed as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Deployment failed after {elapsed:.0f}s!")
            logger.error(f"   {e}")
            return EvaluationResult.failure(
                f"Basilica deployment failed: {e}",
                task_id=task_id,
            )
        except (TimeoutError, httpx.TimeoutException) as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Timeout after {elapsed:.0f}s (limit: {http_timeout}s)!")
            logger.error(f"   Exception type: {type(e).__name__}: {e}")
            return EvaluationResult.failure(
                f"Basilica timeout after {elapsed:.0f}s",
                task_id=task_id,
            )
        except httpx.ConnectError as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Connection error after {elapsed:.0f}s!")
            logger.error(f"   {type(e).__name__}: {e}")
            return EvaluationResult.failure(
                f"Basilica connection error: {e}",
                task_id=task_id,
            )
        except httpx.RemoteProtocolError as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Remote protocol error after {elapsed:.0f}s!")
            logger.error(f"   {type(e).__name__}: {e}")
            return EvaluationResult.failure(
                f"Basilica remote protocol error after {elapsed:.0f}s: {e}",
                task_id=task_id,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Error after {elapsed:.0f}s: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            logger.error(traceback.format_exc())
            return EvaluationResult.failure(
                f"Basilica error after {elapsed:.0f}s: {e}",
                task_id=task_id,
            )
        finally:
            if log_stream_task is not None:
                log_stream_task.cancel()
                try:
                    await log_stream_task
                except (asyncio.CancelledError, Exception):
                    pass
            if deployment is not None:
                logger.info("[BASILICA] Cleaning up deployment after evaluation run...")
                try:
                    await deployment.delete_async()
                    logger.info(
                        f"[BASILICA] Deployment '{getattr(deployment, 'name', 'unknown')}' deleted"
                    )
                except Exception as del_err:
                    logger.warning(f"[BASILICA] Failed to delete deployment: {del_err}")

    async def _stream_basilica_logs(self, deployment, log_path: Path, poll_interval: int = 5):
        """Poll Basilica deployment logs every `poll_interval` seconds and
        write them to a local file. Detects consecutive failures and writes
        a marker so container crashes are visible."""
        loop = asyncio.get_event_loop()
        last_len = 0
        consecutive_errors = 0
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(
                    f"# Basilica logs for deployment {getattr(deployment, 'name', 'unknown')}\n"
                )
                f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
                f.write(f"# Poll interval: {poll_interval}s\n\n")

            while True:
                try:
                    raw = await loop.run_in_executor(None, deployment.logs)
                    if raw and len(raw) > last_len:
                        new_content = raw[last_len:]
                        last_len = len(raw)
                        with open(log_path, "a") as f:
                            f.write(new_content)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.debug(f"[BASILICA-LOGS] Poll error #{consecutive_errors}: {e}")
                    if consecutive_errors >= 5:
                        logger.warning(
                            f"[BASILICA-LOGS] {consecutive_errors} consecutive poll"
                            " failures — container may have crashed"
                        )
                        with open(log_path, "a") as f:
                            ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                            f.write(
                                f"\n--- POLL FAILURE x{consecutive_errors}"
                                f" at {ts} ---\n"
                                f"--- Last error: {e} ---\n"
                            )
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            try:
                raw = await loop.run_in_executor(None, deployment.logs)
                if raw and len(raw) > last_len:
                    with open(log_path, "a") as f:
                        f.write(raw[last_len:])
            except Exception:
                pass
            with open(log_path, "a") as f:
                f.write(f"\n# Ended: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
            logger.info(f"[BASILICA-LOGS] Saved to {log_path}")

    # ------------------------------------------------------------------
    # Reusable Basilica deployment: create once, run N evals, destroy
    # ------------------------------------------------------------------

    async def create_basilica_deployment(self) -> BasilicaDeploymentContext:
        """Provision a Basilica deployment that can be reused for multiple evals.

        The deployment stays alive until ``destroy_basilica_deployment`` is called.
        Each eval run spawns a fresh torchrun subprocess inside the container,
        so GPU state is fully clean between runs.

        Raises on deployment failure (caller should catch and handle).
        """
        if not BASILICA_AVAILABLE:
            raise RuntimeError("basilica SDK not installed. Run: uv add basilica")

        auth_token = secrets.token_urlsafe(32)
        deploy_name = f"templar-eval-{uuid.uuid4().hex[:8]}"

        logger.info("=" * 60)
        logger.info("[BASILICA] Creating reusable deployment for multi-run evaluation")
        logger.info("=" * 60)
        logger.info(f"   Image: {self.basilica_image}")
        logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
        logger.info(f"   Min GPU memory: {self.basilica_min_gpu_memory_gb}GB")
        if self.basilica_interconnect:
            logger.info(f"   Interconnect: {self.basilica_interconnect}")
        logger.info(f"   CPU: {self.basilica_cpu}, Memory: {self.basilica_memory}")
        logger.info(f"   Deployment name: {deploy_name}")
        logger.info("[BASILICA] Auth token generated (EVAL_AUTH_TOKEN set)")

        client = BasilicaClient()
        deploy_kwargs = {
            "name": deploy_name,
            "image": self.basilica_image,
            "port": 8000,
            "ttl_seconds": self.basilica_ttl_seconds,
            "gpu_count": self.basilica_gpu_count,
            "gpu_models": self.basilica_gpu_models,
            "min_gpu_memory_gb": self.basilica_min_gpu_memory_gb,
            "cpu": self.basilica_cpu,
            "memory": self.basilica_memory,
            "timeout": 1800,
            "env": {"EVAL_AUTH_TOKEN": auth_token},
        }
        if self.basilica_interconnect:
            deploy_kwargs["interconnect"] = self.basilica_interconnect
        if self.basilica_geo:
            deploy_kwargs["geo"] = self.basilica_geo
        if self.basilica_spot:
            deploy_kwargs["spot"] = self.basilica_spot

        deploy_start = time.time()
        deployment = await client.deploy_async(**deploy_kwargs)
        deploy_time = time.time() - deploy_start

        dep_id = getattr(deployment, "name", None) or getattr(deployment, "id", "unknown")
        logger.info(f"[BASILICA] Deployment ready in {deploy_time:.1f}s")
        logger.info(f"   URL: {deployment.url}")
        logger.info(f"   Deployment ID: {dep_id}")

        log_dir = Path("logs/basilica")
        log_file = log_dir / f"{dep_id}_{int(time.time())}.log"
        log_task = asyncio.create_task(self._stream_basilica_logs(deployment, log_file))
        logger.info(f"[BASILICA] Log streaming started → {log_file}")

        return BasilicaDeploymentContext(
            deployment=deployment,
            auth_token=auth_token,
            url=deployment.url,
            name=deploy_name,
            log_file=log_file,
            log_stream_task=log_task,
        )

    async def destroy_basilica_deployment(self, ctx: BasilicaDeploymentContext) -> None:
        """Tear down a Basilica deployment and stop log streaming."""
        logger.info(
            f"[BASILICA] Destroying deployment '{ctx.name}' (alive for {ctx.age_seconds:.0f}s)"
        )
        if ctx.log_stream_task is not None:
            ctx.log_stream_task.cancel()
            try:
                await ctx.log_stream_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await ctx.deployment.delete_async()
            logger.info(f"[BASILICA] Deployment '{ctx.name}' deleted successfully")
        except Exception as e:
            logger.warning(f"[BASILICA] Failed to delete deployment '{ctx.name}': {e}")

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
        running) before the new one starts — handled by env.py.
        """
        model_url = model_url or self.default_model_url
        data_url = data_url or self.default_data_url

        if not model_url or not data_url:
            return EvaluationResult.failure("model_url and data_url are required", task_id=task_id)
        if not code or "def inner_steps" not in code:
            return EvaluationResult.failure(
                "Invalid code: must contain 'def inner_steps' function",
                task_id=task_id,
            )

        http_timeout = max(self.timeout + 600, 1800)
        start_time = time.time()

        logger.info(f"[BASILICA] Evaluation run on deployment '{ctx.name}'")
        logger.info(f"   Task ID: {task_id}, Seed: {seed}")
        logger.info(f"   Deployment age: {ctx.age_seconds:.0f}s")

        try:
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
                "weight_relative_error_max": self.weight_relative_error_max,
                "timer_divergence_threshold": self.timer_divergence_threshold,
                "gpu_peak_tflops": self.gpu_peak_tflops,
                "max_plausible_mfu": self.max_plausible_mfu,
                "min_mfu": self.min_mfu,
                "num_gpus": self.basilica_gpu_count,
            }

            auth_headers = {"Authorization": f"Bearer {ctx.auth_token}"}

            # Submit job
            logger.info(f"   POST {ctx.url}/evaluate")
            post_start = time.time()
            async with httpx.AsyncClient(timeout=60, headers=auth_headers) as submit_client:
                submit_resp = await submit_client.post(f"{ctx.url}/evaluate", json=payload)

            if submit_resp.status_code not in (200, 202):
                error_text = submit_resp.text[:500]
                logger.error(f"[BASILICA] Job submission failed: {submit_resp.status_code}")
                return EvaluationResult.failure(
                    f"Basilica /evaluate submission error: {submit_resp.status_code} - {error_text}",
                    task_id=task_id,
                )

            submit_data = submit_resp.json()
            job_id = submit_data.get("job_id")
            if not job_id:
                return EvaluationResult.failure(
                    f"Basilica /evaluate did not return job_id: {submit_data}",
                    task_id=task_id,
                )

            logger.info(f"[BASILICA] Job accepted: {job_id}")

            # Poll for result
            poll_url = f"{ctx.url}/eval-status/{job_id}"
            poll_interval = 30
            consecutive_errors = 0
            poll_timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)

            async with httpx.AsyncClient(timeout=poll_timeout, headers=auth_headers) as poll_client:
                while True:
                    elapsed = time.time() - post_start
                    if elapsed >= http_timeout:
                        logger.error(f"[BASILICA] Polling timeout after {elapsed:.0f}s")
                        return EvaluationResult.failure(
                            f"Basilica evaluation timed out after {elapsed:.0f}s (polling)",
                            task_id=task_id,
                        )

                    remaining = http_timeout - elapsed
                    max_consecutive_errors = max(
                        10, int(remaining // (poll_interval + poll_timeout.read))
                    )

                    await asyncio.sleep(poll_interval)
                    elapsed = time.time() - post_start

                    try:
                        poll_resp = await poll_client.get(poll_url)
                        consecutive_errors = 0

                        if poll_resp.status_code == 404:
                            logger.warning(
                                f"[BASILICA] Job {job_id} not found (404) — "
                                "container may have restarted"
                            )
                            return EvaluationResult.failure(
                                f"Basilica job {job_id} lost (404 on poll)",
                                task_id=task_id,
                            )

                        if poll_resp.status_code != 200:
                            logger.warning(
                                f"[BASILICA] Poll got {poll_resp.status_code}, "
                                f"will retry in {poll_interval}s ({elapsed:.0f}s elapsed)"
                            )
                            continue

                        poll_data = poll_resp.json()
                        status = poll_data.get("status")

                        if status == "pending":
                            logger.info(
                                f"   [POLLING] Evaluation in progress... {elapsed:.0f}s elapsed"
                            )
                            continue

                        if status in ("done", "failed"):
                            result_data = poll_data.get("result")
                            if result_data is None:
                                return EvaluationResult.failure(
                                    f"Basilica job {job_id} status={status} but no result",
                                    task_id=task_id,
                                )
                            logger.info(
                                f"[BASILICA] Job {job_id} completed "
                                f"status={status} in {elapsed:.1f}s"
                            )
                            break

                        logger.warning(f"[BASILICA] Unknown poll status: {status}")

                    except (
                        httpx.TimeoutException,
                        httpx.ConnectError,
                        httpx.RemoteProtocolError,
                    ) as poll_err:
                        consecutive_errors += 1
                        logger.warning(
                            f"[BASILICA] Poll error "
                            f"({consecutive_errors}/{max_consecutive_errors}): "
                            f"{type(poll_err).__name__} ({elapsed:.0f}s elapsed)"
                        )
                        if consecutive_errors >= max_consecutive_errors:
                            return EvaluationResult.failure(
                                f"Basilica polling failed {max_consecutive_errors} "
                                f"consecutive times after {elapsed:.0f}s: {poll_err}",
                                task_id=task_id,
                            )
                        continue

            # Validate result
            elapsed = time.time() - post_start
            if not isinstance(result_data, dict):
                return EvaluationResult.failure(
                    f"Basilica returned non-object response: {type(result_data).__name__}",
                    task_id=task_id,
                )

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

            logger.info(
                f"[BASILICA] Run complete on '{ctx.name}': "
                f"success={result.success} MFU={result.mfu:.2f}% "
                f"wall_time={result.wall_time_seconds:.2f}s ({elapsed:.1f}s total)"
            )
            if result.error:
                logger.warning(f"   Error: {result.error}")

            return result

        except (TimeoutError, httpx.TimeoutException) as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Timeout after {elapsed:.0f}s on '{ctx.name}': {e}")
            return EvaluationResult.failure(
                f"Basilica timeout after {elapsed:.0f}s", task_id=task_id
            )
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Connection error on '{ctx.name}' after {elapsed:.0f}s: {e}")
            return EvaluationResult.failure(f"Basilica connection error: {e}", task_id=task_id)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[BASILICA] Unexpected error on '{ctx.name}' after {elapsed:.0f}s: {e}")
            logger.error(traceback.format_exc())
            return EvaluationResult.failure(
                f"Basilica error after {elapsed:.0f}s: {e}", task_id=task_id
            )

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
