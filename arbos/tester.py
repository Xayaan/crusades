"""Basilica-based MFU testing for train.py candidates.

Each evaluation creates a fresh deployment, runs the test, and deletes
the deployment immediately — no long-lived GPU reservations.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

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
            "gradient_norm_ratio_max": v["gradient_norm_ratio_max"],
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
