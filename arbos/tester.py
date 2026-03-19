"""MFU testing for train.py candidates — Basilica (cloud) or local Docker.

Basilica mode: creates a fresh cloud deployment per evaluation, deletes it after.
Local mode:    runs the evaluation inside a local Docker container with GPUs.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger("arbos.tester")

try:
    from basilica import BasilicaClient

    BASILICA_AVAILABLE = True
except ImportError:
    BasilicaClient = None
    BASILICA_AVAILABLE = False


@dataclass
class EvalResult:
    success: bool
    mfu: float = 0.0
    tps: float = 0.0
    total_tokens: int = 0
    wall_time: float = 0.0
    error: str | None = None
    error_code: str | None = None
    diagnostics: dict = field(default_factory=dict)


class BasilicaTester:
    """Creates a fresh Basilica deployment per evaluation, deletes it after."""

    def __init__(self, hparams: dict):
        self._hparams = hparams
        self._bconfig = hparams.get("basilica", {})
        self._ttl = self._bconfig.get("ttl_seconds", 7200)
        self._eval_timeout = hparams.get("eval_timeout", 3600)
        self._cleanup_stale_deployments()

    def _cleanup_stale_deployments(self):
        """Log any leftover deployments — but don't delete them.

        The validator may have active deployments on the same Basilica account,
        and the SDK doesn't expose the instance_name we set at creation time,
        so we can't distinguish arbos vs validator deployments safely.
        Each evaluation already deletes its own deployment in the finally block,
        and stale ones auto-expire via TTL.
        """
        if not BASILICA_AVAILABLE:
            return
        try:
            client = BasilicaClient()
            resp = client.list_deployments()
            if resp.deployments:
                logger.info(
                    f"Note: {len(resp.deployments)} active deployment(s) on this account "
                    "(not deleting — may belong to the validator)"
                )
        except Exception as e:
            logger.warning(f"Could not check deployments: {e}")

    def _build_payload(self, code: str) -> dict:
        h = self._hparams
        v = h["verification"]
        m = h["mfu"]
        return {
            "task_id": int(time.time()),
            "seed": f"arbos:{int(time.time())}",
            "model_url": h["benchmark_model_name"],
            "data_url": h["benchmark_dataset_name"],
            "steps": h["eval_steps"],
            "batch_size": h["benchmark_batch_size"],
            "sequence_length": h["benchmark_sequence_length"],
            "data_samples": h["benchmark_data_samples"],
            "timeout": self._eval_timeout,
            "code": code,
            "use_random_init": True,
            "min_trainable_params_ratio": 1.0,
            "max_loss_difference": v["max_loss_difference"],
            "min_params_changed_ratio": v["min_params_changed_ratio"],
            "weight_relative_error_max": v["weight_relative_error_max"],
            "timer_divergence_threshold": v["timer_divergence_threshold"],
            "gpu_peak_tflops": m["gpu_peak_tflops"],
            "max_plausible_mfu": m["max_plausible_mfu"],
            "min_mfu": m["min_mfu"],
            "num_gpus": self._bconfig.get("gpu_count", h.get("docker", {}).get("num_gpus", 2)),
        }

    async def _evaluate_async(self, code: str) -> EvalResult:
        if not BASILICA_AVAILABLE:
            return EvalResult(
                success=False,
                error="basilica-sdk not installed. Run: pip install basilica-sdk",
            )

        bc = self._bconfig
        deploy_name = f"arbos-eval-{uuid.uuid4().hex[:8]}"
        deployment = None

        try:
            # --- Create deployment ---
            logger.info("Creating Basilica deployment...")
            logger.info(f"  GPU: {bc.get('gpu_count', 2)}x {bc.get('gpu_models', ['A100'])}")
            logger.info(f"  Image: {bc.get('image', 'ghcr.io/one-covenant/templar-eval:latest')}")
            logger.info(f"  Name: {deploy_name}")

            client = BasilicaClient()
            deploy_kwargs = {
                "instance_name": deploy_name,
                "image": bc.get("image", "ghcr.io/one-covenant/templar-eval:latest"),
                "port": 8000,
                "ttl_seconds": self._ttl,
                "gpu_count": bc.get("gpu_count", 2),
                "gpu_models": bc.get("gpu_models", ["A100"]),
                "min_gpu_memory_gb": bc.get("min_gpu_memory_gb", 80),
                "cpu": bc.get("cpu", "8"),
                "memory": bc.get("memory", "160Gi"),
            }
            if bc.get("interconnect"):
                deploy_kwargs["interconnect"] = bc["interconnect"]
            if bc.get("geo"):
                deploy_kwargs["geo"] = bc["geo"]

            deploy_start = time.time()
            response = await client.create_deployment_async(**deploy_kwargs)
            deployment = await client.get_async(response.instance_name)
            logger.info(f"  Basilica ID: {deployment.name}")

            logger.info(f"Waiting for deployment to be ready (timeout: {self._eval_timeout}s)...")
            await deployment.wait_until_ready_async(timeout=self._eval_timeout)
            await deployment.refresh_async()

            deploy_elapsed = time.time() - deploy_start
            logger.info(f"Deployment ready in {deploy_elapsed:.0f}s at {deployment.url}")

            # --- Send evaluation ---
            payload = self._build_payload(code)
            http_timeout = self._eval_timeout + 600

            logger.info("Sending evaluation request...")
            logger.info(f"  URL: {deployment.url}/evaluate")
            logger.info(f"  HTTP timeout: {http_timeout}s")
            eval_start = time.time()

            async with httpx.AsyncClient(timeout=http_timeout) as http:
                resp = await http.post(f"{deployment.url}/evaluate", json=payload)

            eval_elapsed = time.time() - eval_start
            logger.info(f"Response received in {eval_elapsed:.0f}s (HTTP {resp.status_code})")

            if resp.status_code != 200:
                return EvalResult(
                    success=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:500]}",
                )

            data = resp.json()
            return EvalResult(
                success=data.get("success", False),
                mfu=float(data.get("mfu", 0.0)),
                tps=float(data.get("tps", 0.0)),
                total_tokens=int(data.get("total_tokens", 0)),
                wall_time=float(data.get("wall_time_seconds", 0.0)),
                error=data.get("error"),
                error_code=data.get("error_code"),
                diagnostics=data.get("diagnostics", {}),
            )

        except httpx.TimeoutException:
            return EvalResult(
                success=False, error=f"Evaluation timed out after {self._eval_timeout}s"
            )
        except Exception as e:
            return EvalResult(success=False, error=str(e))

        finally:
            # --- Always delete deployment ---
            if deployment is not None:
                try:
                    await deployment.delete_async()
                    logger.info(f"Deployment '{deploy_name}' deleted")
                except Exception as e:
                    logger.warning(f"Failed to delete deployment '{deploy_name}': {e}")

    def evaluate(self, code: str) -> EvalResult:
        """Synchronous wrapper for evaluation."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._evaluate_async(code))
        finally:
            loop.close()

    def cleanup(self):
        """No-op — deployments are deleted after each evaluation."""
        pass


_EVAL_RUNNER_SCRIPT = """\
import asyncio, json, os, sys
sys.path.insert(0, "/app")
from env import Actor

async def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    with open("/test/params.json") as f:
        p = json.load(f)
    actor = Actor()
    result = await actor.evaluate(
        task_id=p["task_id"], seed=p["seed"],
        model_url=p["model_url"], data_url=p["data_url"],
        steps=p["steps"], batch_size=p["batch_size"],
        timeout=p["timeout"], sequence_length=p.get("sequence_length"),
        data_samples=p["data_samples"], code=p["code"],
        max_loss_difference=p["max_loss_difference"],
        use_random_init=p["use_random_init"],
        min_trainable_params_ratio=p["min_trainable_params_ratio"],
        min_params_changed_ratio=p["min_params_changed_ratio"],
        weight_relative_error_max=p["weight_relative_error_max"],
        timer_divergence_threshold=p["timer_divergence_threshold"],
        gpu_peak_tflops=p["gpu_peak_tflops"],
        max_plausible_mfu=p["max_plausible_mfu"],
        min_mfu=p["min_mfu"],
        require_cuda_timing=True,
        num_gpus=p["num_gpus"],
    )
    if local_rank == 0:
        print("EVAL_RESULT:" + json.dumps(result))

asyncio.run(main())
"""


class LocalDockerTester:
    """Runs evaluation in a local Docker container with GPUs.

    Same evaluation logic as Basilica (env.py Actor), but executed locally.
    Requires: Docker with NVIDIA Container Toolkit, the templar-eval image,
    and sufficient GPUs.
    """

    def __init__(
        self,
        hparams: dict,
        project_root: Path | None = None,
        image: str | None = None,
        num_gpus: int | None = None,
        gpu_devices: str | None = None,
    ):
        self._hparams = hparams
        self._docker_cfg = hparams.get("docker", {})
        self._num_gpus = num_gpus or self._docker_cfg.get("num_gpus", 2)
        self._gpu_devices = gpu_devices
        self._eval_timeout = hparams.get("eval_timeout", 3600)

        if project_root is None:
            project_root = Path(__file__).parent.parent
        self._project_root = project_root

        self._image = image or "templar-eval:latest"

    def _build_payload(self, code: str) -> dict:
        h = self._hparams
        v = h["verification"]
        m = h["mfu"]
        return {
            "task_id": int(time.time()),
            "seed": f"arbos:{int(time.time())}",
            "model_url": h["benchmark_model_name"],
            "data_url": h["benchmark_dataset_name"],
            "steps": h["eval_steps"],
            "batch_size": h["benchmark_batch_size"],
            "sequence_length": h["benchmark_sequence_length"],
            "data_samples": h["benchmark_data_samples"],
            "timeout": self._eval_timeout,
            "code": code,
            "use_random_init": True,
            "min_trainable_params_ratio": 1.0,
            "max_loss_difference": v["max_loss_difference"],
            "min_params_changed_ratio": v["min_params_changed_ratio"],
            "weight_relative_error_max": v["weight_relative_error_max"],
            "timer_divergence_threshold": v["timer_divergence_threshold"],
            "gpu_peak_tflops": m["gpu_peak_tflops"],
            "max_plausible_mfu": m["max_plausible_mfu"],
            "min_mfu": m["min_mfu"],
            "num_gpus": self._num_gpus,
        }

    def _docker_cmd(self, code_path: str, params_path: str, script_path: str) -> list[str]:
        if self._gpu_devices:
            gpu_flag = f'"device={self._gpu_devices}"'
        else:
            gpu_flag = str(self._num_gpus)

        root = self._project_root
        cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            gpu_flag,
            "--ipc=host",
            "--ulimit",
            "memlock=-1:-1",
            "-e",
            "NCCL_P2P_LEVEL=NVL",
            "-e",
            "NCCL_SHM_USE_CUDA_MEMCPY=1",
            "-e",
            "NCCL_NVLS_ENABLE=1",
            "-e",
            "NCCL_IB_DISABLE=1",
            "-e",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "-e",
            "PYTHONPATH=/app",
            "-v",
            f"{code_path}:/test/train.py:ro",
            "-v",
            f"{params_path}:/test/params.json:ro",
            "-v",
            f"{script_path}:/test/eval_runner.py:ro",
            "-v",
            f"{root}/environments/templar/env.py:/app/env.py:ro",
            "-v",
            f"{root}/src/crusades/core/security_defs.py:/app/crusades/core/security_defs.py:ro",
            self._image,
        ]

        if self._num_gpus > 1:
            import secrets

            master_port = 29500 + secrets.randbelow(10001)
            cmd.extend(
                [
                    "torchrun",
                    "--nproc_per_node",
                    str(self._num_gpus),
                    "--master_port",
                    str(master_port),
                    "/test/eval_runner.py",
                ]
            )
        else:
            cmd.extend(["python3", "/test/eval_runner.py"])

        return cmd

    def evaluate(self, code: str) -> EvalResult:
        payload = self._build_payload(code)

        tmp_files: list[str] = []
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="arbos_code_"
            ) as f:
                f.write(code)
                tmp_files.append(f.name)
                code_path = f.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="arbos_params_"
            ) as f:
                json.dump(payload, f)
                tmp_files.append(f.name)
                params_path = f.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="arbos_runner_"
            ) as f:
                f.write(_EVAL_RUNNER_SCRIPT)
                tmp_files.append(f.name)
                script_path = f.name

            cmd = self._docker_cmd(code_path, params_path, script_path)
            logger.info(f"Running local Docker evaluation ({self._num_gpus} GPU(s))...")
            logger.debug(f"Docker command: {' '.join(cmd)}")

            timeout = self._eval_timeout + 300
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            deadline = time.monotonic() + timeout
            result_data = None
            try:
                for line in proc.stdout:
                    if time.monotonic() > deadline:
                        proc.kill()
                        raise subprocess.TimeoutExpired(cmd, timeout)
                    line = line.rstrip()
                    if line.startswith("EVAL_RESULT:"):
                        result_data = json.loads(line[len("EVAL_RESULT:") :])
                    else:
                        logger.info(f"[docker] {line}")
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return EvalResult(
                    success=False,
                    error=f"Docker evaluation timed out after {timeout}s",
                )

            proc.wait(timeout=30)

            if result_data is None:
                return EvalResult(
                    success=False,
                    error=f"No EVAL_RESULT in Docker output (exit code {proc.returncode})",
                )

            return EvalResult(
                success=result_data.get("success", False),
                mfu=float(result_data.get("mfu", 0.0)),
                tps=float(result_data.get("tps", 0.0)),
                total_tokens=int(result_data.get("total_tokens", 0)),
                wall_time=float(result_data.get("wall_time_seconds", 0.0)),
                error=result_data.get("error"),
                error_code=result_data.get("error_code"),
                diagnostics=result_data.get("diagnostics", {}),
            )

        except subprocess.TimeoutExpired:
            proc.kill()
            return EvalResult(
                success=False, error=f"Docker evaluation timed out after {self._eval_timeout}s"
            )
        except Exception as e:
            return EvalResult(success=False, error=str(e))

        finally:
            for p in tmp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def cleanup(self):
        """No-op — container is removed after each run."""
        pass
