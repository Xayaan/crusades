"""Docker-based executor strategy for local GPU evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .base import EvalConfig

if TYPE_CHECKING:
    from crusades.affinetes.runner import EvaluationResult

logger = logging.getLogger(__name__)


class DockerExecutor:
    """Runs evaluations locally using Docker containers.

    Code is mounted directly into the container - no downloads needed.
    """

    # Default Docker image for local evaluation
    DEFAULT_DOCKER_IMAGE = os.getenv("VALIDATOR_EVAL_IMAGE", "templar-eval:latest")

    _INTERNAL_NETWORK_PREFIX = "crusades_nccl_"

    def __init__(
        self,
        config: EvalConfig,
        docker_memory_limit: str = "32g",
        docker_shm_size: str = "8g",
        num_gpus: int = 1,
        validator_image: str | None = None,
    ):
        self.config = config
        self.docker_memory_limit = docker_memory_limit
        self.docker_shm_size = docker_shm_size
        self.num_gpus = num_gpus
        self.validator_image = validator_image or self.DEFAULT_DOCKER_IMAGE

    @classmethod
    def _create_eval_network(cls) -> str | None:
        """Create a per-evaluation internal Docker network for multi-GPU NCCL.

        Each evaluation gets its own isolated network to prevent cross-container
        communication between concurrent evaluations.  Returns the network name
        on success, ``None`` on failure (falls back to ``--network none``).
        """
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
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation locally using Docker."""
        from crusades.affinetes.runner import EvaluationResult

        logger.info("Running Docker evaluation")
        logger.info(f"   Code size: {len(code)} bytes")

        # Check if validator image exists
        check_cmd = ["docker", "image", "inspect", self.validator_image]
        check_result = subprocess.run(check_cmd, capture_output=True)

        if check_result.returncode != 0:
            return EvaluationResult.failure(
                f"Validator image not found: {self.validator_image}. "
                "Build it first: cd environments/templar "
                f"&& docker build -t {self.validator_image} .",
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
        timeout={self.config.timeout},
        code=code,
        max_loss_difference={self.config.max_loss_difference},
        use_random_init=True,
        min_trainable_params_ratio=1.0,
        min_params_changed_ratio={self.config.min_params_changed_ratio},
        # Weight verification
        weight_relative_error_max={self.config.weight_relative_error_max},
        # Timer integrity
        timer_divergence_threshold={self.config.timer_divergence_threshold},
        # MFU calculation
        gpu_peak_tflops={self.config.gpu_peak_tflops},
        max_plausible_mfu={self.config.max_plausible_mfu},
        min_mfu={self.config.min_mfu},
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
                    str(self.config.timeout),
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
                            timeout=self.config.timeout + 60,
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
                    f"Evaluation timed out after {self.config.timeout}s",
                    task_id=task_id,
                )

            stdout_text = "\n".join(stdout_lines)

            if process.returncode != 0:
                # Log the output for debugging
                logger.error(f"Docker container failed with exit code {process.returncode}")
                if stdout_text:
                    logger.error(f"Docker output: {stdout_text[:500]}")
                return EvaluationResult.failure(
                    f"Container failed with exit code: "
                    f"{process.returncode}. "
                    f"Output: {stdout_text[:200]}",
                    task_id=task_id,
                )

            # Parse result
            for line in stdout_text.split("\n"):
                if line.startswith("EVAL_RESULT:"):
                    result_json = line[len("EVAL_RESULT:"):]
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

    async def build_validator_image(self, env_path: Path | None = None) -> bool:
        """Build the validator's evaluation Docker image.

        Args:
            env_path: Path to environments/templar directory

        Returns:
            True if build succeeded
        """
        if env_path is None:
            candidates = [
                Path(__file__).parent.parent.parent.parent.parent / "environments" / "templar",
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

        import subprocess

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
