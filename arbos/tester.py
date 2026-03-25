"""MFU testing for train.py candidates — Basilica (cloud) or local Docker.

Basilica mode: creates a fresh cloud deployment per evaluation, deletes it after.
Local mode:    runs the evaluation inside a local Docker container with GPUs.
"""

import asyncio
import json
import logging
import math
import os
import subprocess
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger("arbos.tester")

try:
    from basilica import BasilicaClient, DeploymentFailed, DeploymentTimeout

    BASILICA_AVAILABLE = True
except ImportError:
    BasilicaClient = None
    DeploymentFailed = None
    DeploymentTimeout = None
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
            "num_gpus": self._bconfig.get("gpu_count", h.get("docker", {}).get("num_gpus", 4)),
        }

    async def _stream_logs(self, deployment, log_path: Path, poll_interval: int = 5):
        """Poll Basilica deployment logs and write them to a local file."""
        loop = asyncio.get_event_loop()
        last_len = 0
        consecutive_errors = 0
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            dep_name = getattr(deployment, "name", "unknown")
            with open(log_path, "w") as f:
                f.write(f"# Basilica logs for arbos deployment {dep_name}\n")
                f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
                f.write(f"# Poll interval: {poll_interval}s\n\n")

            while True:
                try:
                    raw = await loop.run_in_executor(None, deployment.logs)
                    if raw and len(raw) > last_len:
                        with open(log_path, "a") as f:
                            f.write(raw[last_len:])
                        last_len = len(raw)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.debug(f"[ARBOS-LOGS] Poll error #{consecutive_errors}: {e}")
                    if consecutive_errors >= 5:
                        logger.warning(
                            f"[ARBOS-LOGS] {consecutive_errors} consecutive poll"
                            " failures — container may have crashed"
                        )
                        with open(log_path, "a") as f:
                            ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                            f.write(
                                f"\n--- POLL FAILURE x{consecutive_errors} at {ts} ---\n"
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
            logger.info(f"[ARBOS-LOGS] Saved to {log_path}")

    async def _evaluate_async(self, code: str) -> EvalResult:
        if not BASILICA_AVAILABLE:
            return EvalResult(
                success=False,
                error="basilica-sdk not installed. Run: pip install basilica-sdk",
            )

        bc = self._bconfig
        deploy_name = f"arbos-eval-{uuid.uuid4().hex[:8]}"
        deployment = None
        log_stream_task = None
        start_time = time.time()
        http_timeout = max(self._eval_timeout + 600, 1800)

        try:
            # --- Create deployment via deploy_async (high-level API) ---
            logger.info("Creating Basilica deployment...")
            logger.info(f"  GPU: {bc.get('gpu_count', 4)}x {bc.get('gpu_models', ['A100'])}")
            logger.info(f"  Image: {bc.get('image', 'ghcr.io/one-covenant/templar-eval:latest')}")
            logger.info(f"  CPU: {bc.get('cpu', '24')}, Memory: {bc.get('memory', '480Gi')}")
            if bc.get("interconnect"):
                logger.info(f"  Interconnect: {bc['interconnect']}")
            logger.info(f"  TTL: {self._ttl}s ({self._ttl / 60:.0f} min)")
            logger.info(f"  Name: {deploy_name}")

            client = BasilicaClient()
            deploy_kwargs = {
                "name": deploy_name,
                "image": bc.get("image", "ghcr.io/one-covenant/templar-eval:latest"),
                "port": 8000,
                "ttl_seconds": self._ttl,
                "gpu_count": bc.get("gpu_count", 4),
                "gpu_models": bc.get("gpu_models", ["A100"]),
                "min_gpu_memory_gb": bc.get("min_gpu_memory_gb", 80),
                "cpu": bc.get("cpu", "24"),
                "memory": bc.get("memory", "480Gi"),
                "timeout": 1800,
            }
            if bc.get("interconnect"):
                deploy_kwargs["interconnect"] = bc["interconnect"]
            if bc.get("geo"):
                deploy_kwargs["geo"] = bc["geo"]
            if bc.get("spot"):
                deploy_kwargs["spot"] = bc["spot"]

            deploy_start = time.time()
            deployment = await client.deploy_async(**deploy_kwargs)
            deploy_elapsed = time.time() - deploy_start

            dep_name = getattr(deployment, "name", None) or getattr(deployment, "id", "unknown")
            logger.info(f"Deployment ready in {deploy_elapsed:.0f}s")
            logger.info(f"  URL: {deployment.url}")
            logger.info(f"  Deployment ID: {dep_name}")

            # --- Start log streaming ---
            log_dir = Path("arbos/logs/basilica")
            log_file = log_dir / f"{dep_name}_{int(time.time())}.log"
            log_stream_task = asyncio.create_task(self._stream_logs(deployment, log_file))
            logger.info(f"  Log streaming started → {log_file}")

            # --- Submit evaluation job ---
            payload = self._build_payload(code)

            logger.info("Submitting evaluation job...")
            logger.info(f"  URL: {deployment.url}/evaluate")
            logger.info(f"  Timeout budget: {http_timeout}s")
            eval_start = time.time()

            async with httpx.AsyncClient(timeout=60) as submit_client:
                submit_resp = await submit_client.post(f"{deployment.url}/evaluate", json=payload)

            if submit_resp.status_code not in (200, 202):
                return EvalResult(
                    success=False,
                    error=f"Job submission failed: HTTP {submit_resp.status_code}: "
                    f"{submit_resp.text[:500]}",
                )

            submit_data = submit_resp.json()
            job_id = submit_data.get("job_id")
            if not job_id:
                return EvalResult(
                    success=False,
                    error=f"/evaluate did not return job_id: {submit_data}",
                )

            logger.info(f"  Job accepted: {job_id}")

            # --- Poll /eval-status/{job_id} until done ---
            poll_url = f"{deployment.url}/eval-status/{job_id}"
            poll_interval = 30
            consecutive_errors = 0
            poll_timeout = httpx.Timeout(connect=10, read=60, write=10, pool=10)

            async with httpx.AsyncClient(timeout=poll_timeout) as poll_client:
                while True:
                    elapsed = time.time() - eval_start
                    if elapsed >= http_timeout:
                        return EvalResult(
                            success=False,
                            error=f"Polling timed out after {elapsed:.0f}s "
                            f"(budget: {http_timeout}s)",
                        )

                    remaining = http_timeout - elapsed
                    max_consecutive_errors = max(
                        10, int(remaining // (poll_interval + poll_timeout.read))
                    )

                    await asyncio.sleep(poll_interval)
                    elapsed = time.time() - eval_start

                    try:
                        poll_resp = await poll_client.get(poll_url)
                        consecutive_errors = 0

                        if poll_resp.status_code == 404:
                            return EvalResult(
                                success=False,
                                error=f"Job {job_id} lost (404 on poll) — "
                                "container may have restarted",
                            )

                        if poll_resp.status_code != 200:
                            logger.warning(
                                f"Poll got {poll_resp.status_code}, retrying "
                                f"in {poll_interval}s ({elapsed:.0f}s elapsed)"
                            )
                            continue

                        poll_data = poll_resp.json()
                        status = poll_data.get("status")

                        if status == "pending":
                            logger.info(
                                f"  [POLLING] Evaluation in progress... {elapsed:.0f}s elapsed"
                            )
                            continue

                        if status in ("done", "failed"):
                            result_data = poll_data.get("result")
                            if result_data is None:
                                return EvalResult(
                                    success=False,
                                    error=f"Job {job_id} status={status} but no result",
                                )
                            logger.info(
                                f"  Job {job_id} completed: status={status} in {elapsed:.0f}s"
                            )
                            break

                        logger.warning(
                            f"Unknown poll status: {status}, retrying in {poll_interval}s"
                        )

                    except (
                        httpx.TimeoutException,
                        httpx.ConnectError,
                        httpx.RemoteProtocolError,
                    ) as poll_err:
                        consecutive_errors += 1
                        logger.warning(
                            f"Poll error ({consecutive_errors}/"
                            f"{max_consecutive_errors}): "
                            f"{type(poll_err).__name__} ({elapsed:.0f}s elapsed)"
                        )
                        if consecutive_errors >= max_consecutive_errors:
                            return EvalResult(
                                success=False,
                                error=f"Polling failed {max_consecutive_errors} "
                                f"consecutive times after {elapsed:.0f}s: {poll_err}",
                            )
                        continue

            eval_elapsed = time.time() - eval_start
            logger.info(f"Evaluation result received in {eval_elapsed:.0f}s")

            # --- Validate result ---
            if not isinstance(result_data, dict):
                return EvalResult(
                    success=False,
                    error=f"Result is not a dict: {type(result_data).__name__}",
                )

            for numeric_field in ("mfu", "tps", "wall_time_seconds"):
                if numeric_field in result_data:
                    value = float(result_data[numeric_field])
                    if not math.isfinite(value):
                        return EvalResult(
                            success=False,
                            error=f"Result has non-finite {numeric_field}: {value}",
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

        except DeploymentTimeout as e:
            elapsed = time.time() - start_time
            logger.error(f"Deployment timed out after {elapsed:.0f}s: {e}")
            return EvalResult(
                success=False,
                error=f"Basilica deployment timeout after {elapsed:.0f}s: {e}",
            )
        except DeploymentFailed as e:
            elapsed = time.time() - start_time
            logger.error(f"Deployment failed after {elapsed:.0f}s: {e}")
            return EvalResult(success=False, error=f"Basilica deployment failed: {e}")
        except (TimeoutError, httpx.TimeoutException) as e:
            elapsed = time.time() - start_time
            logger.error(f"Timeout after {elapsed:.0f}s: {type(e).__name__}: {e}")
            return EvalResult(
                success=False,
                error=f"Basilica timeout after {elapsed:.0f}s: {e}",
            )
        except httpx.ConnectError as e:
            elapsed = time.time() - start_time
            logger.error(f"Connection error after {elapsed:.0f}s: {e}")
            return EvalResult(success=False, error=f"Basilica connection error: {e}")
        except httpx.RemoteProtocolError as e:
            elapsed = time.time() - start_time
            logger.error(f"Remote protocol error after {elapsed:.0f}s: {e}")
            return EvalResult(
                success=False,
                error=f"Basilica remote protocol error after {elapsed:.0f}s: {e}",
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error after {elapsed:.0f}s: {e}")
            logger.error(traceback.format_exc())
            return EvalResult(success=False, error=str(e))

        finally:
            if log_stream_task is not None:
                log_stream_task.cancel()
                try:
                    await log_stream_task
                except (asyncio.CancelledError, Exception):
                    pass
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
        self._session_id = f"arbos-{uuid.uuid4().hex[:12]}"

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
            "--label",
            f"arbos.session={self._session_id}",
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

    def _kill_container(self, proc: subprocess.Popen, wait: int = 15):
        """Force-kill Docker container backing *proc* and wait for exit."""
        proc.kill()
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            logger.warning("Container did not exit after SIGKILL, force-removing...")
            try:
                subprocess.run(
                    ["docker", "kill", "--signal=SIGKILL", str(proc.pid)],
                    timeout=10,
                    capture_output=True,
                )
            except Exception:
                pass
            proc.wait(timeout=30)

    def _post_run_cleanup(self):
        """Clean up GPU state and /dev/shm after a Docker run.

        Only kills containers started by THIS arbos session (identified by
        the ``arbos.session`` label) so manually launched test containers
        and other arbos instances are not affected.
        """
        try:
            leftover = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"label=arbos.session={self._session_id}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for cid in leftover.stdout.strip().split("\n"):
                if cid:
                    logger.warning(f"Killing leftover container {cid}")
                    subprocess.run(["docker", "kill", cid], capture_output=True, timeout=10)
        except Exception as e:
            logger.debug(f"Container cleanup error: {e}")

        import glob

        for shm in glob.glob("/dev/shm/nccl-*"):
            try:
                os.unlink(shm)
            except OSError:
                pass

    def evaluate(self, code: str) -> EvalResult:
        payload = self._build_payload(code)

        tmp_files: list[str] = []
        proc = None
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
                        self._kill_container(proc)
                        raise subprocess.TimeoutExpired(cmd, timeout)
                    line = line.rstrip()
                    if line.startswith("EVAL_RESULT:"):
                        result_data = json.loads(line[len("EVAL_RESULT:") :])
                    else:
                        logger.info(f"[docker] {line}")
            except subprocess.TimeoutExpired:
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
            if proc is not None:
                self._kill_container(proc)
            return EvalResult(
                success=False, error=f"Docker evaluation timed out after {self._eval_timeout}s"
            )
        except Exception as e:
            return EvalResult(success=False, error=str(e))

        finally:
            self._post_run_cleanup()
            for p in tmp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def cleanup(self):
        """Kill any leftover containers and clean shared memory."""
        self._post_run_cleanup()
