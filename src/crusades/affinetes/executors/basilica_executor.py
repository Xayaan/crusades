"""Basilica-based executor strategy for remote cloud GPU evaluation."""

from __future__ import annotations

import asyncio
import logging
import math
import secrets
import time
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .base import EvalConfig

if TYPE_CHECKING:
    from crusades.affinetes.runner import BasilicaDeploymentContext, EvaluationResult

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


class BasilicaExecutor:
    """Runs evaluations remotely via Basilica SDK.

    Uses BasilicaClient.deploy_async to provision a GPU container and call
    the /evaluate endpoint for MFU evaluation.
    """

    # Default Basilica image (must be pushed to registry like ghcr.io)
    DEFAULT_BASILICA_IMAGE = "ghcr.io/one-covenant/templar-eval:latest"

    def __init__(
        self,
        config: EvalConfig,
        basilica_api_key: str | None = None,
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
        self.config = config
        self.basilica_api_key = basilica_api_key or __import__("os").getenv("BASILICA_API_TOKEN")
        self.basilica_image = basilica_image or __import__("os").getenv(
            "BASILICA_EVAL_IMAGE", self.DEFAULT_BASILICA_IMAGE
        )
        self.basilica_ttl_seconds = basilica_ttl_seconds
        self.basilica_gpu_count = basilica_gpu_count
        self.basilica_gpu_models = basilica_gpu_models or ["A100", "H100"]
        self.basilica_min_gpu_memory_gb = basilica_min_gpu_memory_gb
        self.basilica_cpu = basilica_cpu
        self.basilica_memory = basilica_memory
        self.basilica_interconnect = basilica_interconnect
        self.basilica_geo = basilica_geo
        self.basilica_spot = basilica_spot

        if not self.basilica_api_key:
            logger.warning("Basilica mode: BASILICA_API_TOKEN not set")
        logger.info("Basilica mode initialized")
        logger.info(f"   Image: {self.basilica_image}")
        logger.info(f"   TTL: {self.basilica_ttl_seconds}s")
        logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
        logger.info(f"   Min GPU Memory: {self.basilica_min_gpu_memory_gb}GB")
        logger.info(f"   CPU/Memory: {self.basilica_cpu} / {self.basilica_memory}")

    def _build_deploy_kwargs(self, deploy_name: str, auth_token: str) -> dict:
        """Build the keyword arguments for BasilicaClient.deploy_async."""
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
        return deploy_kwargs

    def _build_eval_payload(
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
    ) -> dict:
        """Build the evaluation request payload."""
        return {
            "task_id": task_id,
            "seed": seed,
            "model_url": model_url,
            "data_url": data_url,
            "steps": steps,
            "batch_size": batch_size,
            "timeout": self.config.timeout,
            "sequence_length": sequence_length,
            "data_samples": data_samples,
            "code": code,
            "max_loss_difference": self.config.max_loss_difference,
            "use_random_init": True,
            "min_trainable_params_ratio": 1.0,
            "min_params_changed_ratio": self.config.min_params_changed_ratio,
            "weight_relative_error_max": self.config.weight_relative_error_max,
            "timer_divergence_threshold": self.config.timer_divergence_threshold,
            "gpu_peak_tflops": self.config.gpu_peak_tflops,
            "max_plausible_mfu": self.config.max_plausible_mfu,
            "min_mfu": self.config.min_mfu,
            "num_gpus": self.basilica_gpu_count,
        }

    def _validate_result_data(
        self, result_data: object, seed: str, task_id: int
    ) -> EvaluationResult | None:
        """Validate Basilica response data. Returns an error result if invalid, None if valid."""
        from crusades.affinetes.runner import EvaluationResult

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

        return None

    async def _poll_for_result(
        self,
        poll_url: str,
        auth_headers: dict,
        job_id: str,
        post_start: float,
        http_timeout: float,
        task_id: int,
    ) -> EvaluationResult | dict:
        """Poll /eval-status/{job_id} until done.

        Returns result_data dict or EvaluationResult on error.
        """
        from crusades.affinetes.runner import EvaluationResult

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
                        return result_data

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

    async def _submit_and_poll(
        self,
        url: str,
        auth_token: str,
        payload: dict,
        seed: str,
        code: str,
        task_id: int,
        http_timeout: float,
    ) -> EvaluationResult:
        """Submit an evaluation job and poll until complete."""
        from crusades.affinetes.runner import EvaluationResult

        auth_headers = {"Authorization": f"Bearer {auth_token}"}

        logger.info("[BASILICA] Sending evaluation request (async)...")
        logger.info(f"   POST {url}/evaluate")
        logger.info(f"   Timeout budget: {http_timeout}s")
        logger.info(f"   Payload size: {len(str(payload))} chars")

        post_start = time.time()

        # Step 1: Submit job (returns 202 + job_id immediately)
        async with httpx.AsyncClient(timeout=60, headers=auth_headers) as submit_client:
            submit_resp = await submit_client.post(f"{url}/evaluate", json=payload)

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

        # Step 2: Poll /eval-status/{job_id} until done
        poll_url = f"{url}/eval-status/{job_id}"
        poll_result = await self._poll_for_result(
            poll_url=poll_url,
            auth_headers=auth_headers,
            job_id=job_id,
            post_start=post_start,
            http_timeout=http_timeout,
            task_id=task_id,
        )

        # If poll returned an EvaluationResult, it's an error
        if isinstance(poll_result, EvaluationResult):
            return poll_result

        result_data = poll_result
        elapsed = time.time() - post_start
        logger.info(f"[BASILICA] Evaluation result received in {elapsed:.1f}s")

        # Validate result
        validation_error = self._validate_result_data(result_data, seed, task_id)
        if validation_error is not None:
            return validation_error

        result = EvaluationResult.from_dict(result_data)
        result.code = code
        return result

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
        """Run evaluation remotely via Basilica SDK.

        Creates a fresh deployment, runs evaluation, and deletes deployment.

        Flow:
        1. deploy_async: create deployment, wait until DNS + HTTP ready
        2. POST /evaluate -> get job_id back (instant, no proxy timeout)
        3. Poll GET /eval-status/{job_id} every 30s until done
        4. Delete deployment (in finally block)
        """
        from crusades.affinetes.runner import EvaluationResult

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
        http_timeout = max(self.config.timeout + 600, 1800)
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
            deploy_kwargs = self._build_deploy_kwargs(deploy_name, auth_token)

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
            log_stream_task = asyncio.create_task(
                self._stream_basilica_logs(deployment, log_file)
            )
            logger.info(f"[BASILICA] Log streaming started -> {log_file}")

            payload = self._build_eval_payload(
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

            result = await self._submit_and_poll(
                url=deployment.url,
                auth_token=auth_token,
                payload=payload,
                seed=seed,
                code=code,
                task_id=task_id,
                http_timeout=http_timeout,
            )

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
        from crusades.affinetes.runner import BasilicaDeploymentContext

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
        deploy_kwargs = self._build_deploy_kwargs(deploy_name, auth_token)

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
        logger.info(f"[BASILICA] Log streaming started -> {log_file}")

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
        running) before the new one starts -- handled by env.py.
        """
        from crusades.affinetes.runner import EvaluationResult

        model_url = model_url or self.config.model_url
        data_url = data_url or self.config.data_url

        if not model_url or not data_url:
            return EvaluationResult.failure("model_url and data_url are required", task_id=task_id)
        if not code or "def inner_steps" not in code:
            return EvaluationResult.failure(
                "Invalid code: must contain 'def inner_steps' function",
                task_id=task_id,
            )

        http_timeout = max(self.config.timeout + 600, 1800)
        start_time = time.time()

        logger.info(f"[BASILICA] Evaluation run on deployment '{ctx.name}'")
        logger.info(f"   Task ID: {task_id}, Seed: {seed}")
        logger.info(f"   Deployment age: {ctx.age_seconds:.0f}s")

        try:
            payload = self._build_eval_payload(
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
                    f"Basilica /evaluate submission error: "
                    f"{submit_resp.status_code} - {error_text}",
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

            async with httpx.AsyncClient(
                timeout=poll_timeout, headers=auth_headers
            ) as poll_client:
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
            validation_error = self._validate_result_data(result_data, seed, task_id)
            if validation_error is not None:
                return validation_error

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
            logger.error(
                f"[BASILICA] Connection error on '{ctx.name}' after {elapsed:.0f}s: {e}"
            )
            return EvaluationResult.failure(f"Basilica connection error: {e}", task_id=task_id)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[BASILICA] Unexpected error on '{ctx.name}' after {elapsed:.0f}s: {e}"
            )
            logger.error(traceback.format_exc())
            return EvaluationResult.failure(
                f"Basilica error after {elapsed:.0f}s: {e}", task_id=task_id
            )
