"""Evaluates miner submissions by downloading code and running benchmarks."""

import asyncio
import hashlib
import logging
import secrets
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse, urlunparse

import crusades
from crusades.affinetes import AffinetesRunner, BasilicaDeploymentContext, EvaluationResult
from crusades.chain.commitments import CodeUrlInfo
from crusades.config import get_hparams
from crusades.core.protocols import SubmissionStatus
from crusades.storage.database import Database
from crusades.storage.models import EvaluationModel

logger = logging.getLogger(__name__)


class SubmissionEvaluator:
    """Downloads code, verifies hashes, runs evaluations, and finalizes submissions.

    Responsibilities:
    - Downloading code from URL (with SSRF protection)
    - Hash verification
    - Running evaluations via AffinetesRunner
    - Finalizing submissions (score calculation)
    - Updating adaptive threshold on new leaders
    """

    def __init__(
        self,
        db: Database,
        affinetes_runner: AffinetesRunner,
        hotkey: str,
        commitment_reader: object | None = None,
        weight_setter_callback: Callable[[], Awaitable[None]] | None = None,
        cleanup_memory_callback: Callable[[], None] | None = None,
        save_state_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.db = db
        self.affinetes_runner = affinetes_runner
        self.hotkey = hotkey
        self.commitment_reader = commitment_reader
        self._weight_setter_callback = weight_setter_callback
        self._cleanup_memory_callback = cleanup_memory_callback
        self._save_state_callback = save_state_callback

    def _download_from_url(self, code_url: str) -> tuple[bool, str]:
        """Download train.py code from a URL with SSRF protection.

        Validates that:
        1. URL resolves to a non-private IP address (SSRF protection)
        2. Redirects don't lead to private IP addresses
        3. Response is a single Python file, not HTML/folder

        Known limitation: DNS rebinding TOCTOU -- the hostname is resolved and
        validated before the HTTP request, but a malicious DNS server could
        return a different (private) IP for the second resolution during
        ``opener.open()``. The risk is low because redirect destinations are
        also validated, and the response body is size-limited and content-checked.

        Args:
            code_url: The URL containing train.py code

        Returns:
            Tuple of (success, code_or_error)
        """
        # Expand short gist.github.com URLs to raw fetch form.
        # Miners commit the short form to fit the 128-byte on-chain limit.
        parsed = urlparse(code_url)
        if parsed.hostname and "gist.github.com" in parsed.hostname.lower():
            new_host = parsed.hostname.replace("gist.github.com", "gist.githubusercontent.com")
            path = parsed.path.rstrip("/")
            if not path.endswith("/raw"):
                path += "/raw"
            code_url = urlunparse(parsed._replace(netloc=new_host, path=path))

        # SSRF Protection: Validate URL before making request
        code_url_info = CodeUrlInfo(url=code_url)
        is_safe, validation_result = code_url_info.validate_url_security()

        if not is_safe:
            logger.warning(f"SSRF protection blocked URL: {validation_result}")
            return False, f"URL blocked for security: {validation_result}"

        logger.debug(f"URL validated, resolved to IP: {validation_result}")

        try:
            # Use a custom opener that validates redirect destinations
            class SSRFSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
                """Custom redirect handler that validates redirect destinations for SSRF."""

                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    """Validate redirect destination before following."""
                    # Validate the redirect URL
                    redirect_info = CodeUrlInfo(url=newurl)
                    is_redirect_safe, redirect_result = redirect_info.validate_url_security()

                    if not is_redirect_safe:
                        logger.warning(
                            f"SSRF protection blocked redirect to: {newurl} - {redirect_result}"
                        )
                        raise urllib.error.URLError(
                            f"Redirect blocked for security: {redirect_result}"
                        )

                    logger.debug(f"Redirect validated: {newurl} -> {redirect_result}")
                    return super().redirect_request(req, fp, code, msg, headers, newurl)

            # Build opener with SSRF-safe redirect handler
            opener = urllib.request.build_opener(SSRFSafeRedirectHandler())

            max_size = 500_000
            hparams = get_hparams()
            req = urllib.request.Request(code_url, headers={"User-Agent": "templar-crusades"})
            with opener.open(req, timeout=hparams.code_fetch_timeout) as response:
                # Read in chunks with size limit
                chunks = []
                total_bytes = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_size:
                        return False, f"File too large (>{max_size} bytes). Max 500KB"
                    chunks.append(chunk)
                code = b"".join(chunks).decode("utf-8")

                # Reject HTML (folder page, not raw file)
                code_start_lower = code[:500].lower()
                if "<html" in code_start_lower or "<!doctype html" in code_start_lower:
                    return False, "URL returns HTML page, not a code file"

                # Reject JSON file listings
                if code.strip().startswith("{") and '"files"' in code[:500]:
                    return False, "URL returns JSON (file listing), not code"

                # Must contain inner_steps
                if "def inner_steps" not in code:
                    return False, "Code does not contain 'def inner_steps' function"

                return True, code

        except urllib.error.HTTPError as e:
            return False, f"HTTP error {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            return False, f"URL error: {e.reason}"
        except Exception as e:
            return False, f"Error downloading code: {e}"

    async def evaluate_submissions(self) -> None:
        """Evaluate submissions by downloading from code URL.

        Only evaluates submissions matching current competition version.
        """
        hparams = get_hparams()
        competition_version = crusades.COMPETITION_VERSION
        # Only evaluate submissions from current version
        evaluating = await self.db.get_evaluating_submissions(spec_version=competition_version)
        num_runs = hparams.evaluation_runs

        logger.info(
            f"Found {len(evaluating)} submissions in EVALUATING status (v{competition_version})"
        )

        for sub_idx, submission in enumerate(evaluating):
            # Check if we should set weights between submissions
            # This ensures weight setting isn't blocked by long evaluation runs
            if sub_idx > 0 and self._weight_setter_callback is not None:
                await self._weight_setter_callback()

            code_url = submission.bucket_path

            if not code_url or not code_url.startswith("http"):
                logger.error(f"Invalid code URL for {submission.submission_id}: {code_url}")
                continue

            existing_evals = await self.db.get_evaluations(submission.submission_id)
            my_evals = [e for e in existing_evals if e.evaluator_hotkey == self.hotkey]

            if len(my_evals) >= num_runs:
                continue

            runs_remaining = num_runs - len(my_evals)
            logger.info(f"Evaluating {submission.submission_id}")
            logger.info(f"   URL: {code_url[:60]}...")
            logger.info(f"   Runs: {len(my_evals) + 1}/{num_runs}")

            # Download code from URL
            success, code_or_error = self._download_from_url(code_url)

            if not success:
                logger.error(f"Failed to download code: {code_or_error}")
                await self.db.update_submission_status(
                    submission.submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message=f"Failed to download code: {code_or_error}",
                )
                continue

            miner_code = code_or_error
            logger.info(f"   Downloaded {len(miner_code)} bytes")

            # Verify code hash from commitment (integrity check)
            # Hash is 32-char truncated SHA256 from packed commitment format
            committed_hash = submission.code_hash
            actual_hash = hashlib.sha256(miner_code.encode("utf-8")).hexdigest()[:32]
            if not committed_hash:
                logger.warning(
                    "   Legacy submission without code_hash — skipping hash verification. "
                    "This is a security risk; legacy submissions should be phased out."
                )
            elif actual_hash != committed_hash:
                logger.error(
                    f"Code hash mismatch! URL content changed after commitment.\n"
                    f"   Committed: {committed_hash}\n"
                    f"   Actual:    {actual_hash}"
                )
                await self.db.update_submission_status(
                    submission.submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message="Code hash mismatch — URL content changed after commitment",
                )
                continue
            else:
                logger.info(f"   Code hash verified: {actual_hash}")

            # Run evaluations -- one Basilica deployment per miner, N runs on it.
            # Each /evaluate call spawns a fresh torchrun subprocess inside the
            # container so GPU state is fully clean between runs.  The
            # deployment is destroyed after all runs (or on failure).
            fatal_error = False
            deployment_ctx: BasilicaDeploymentContext | None = None
            use_shared_deployment = self.affinetes_runner.mode == "basilica" and runs_remaining > 1

            try:
                if use_shared_deployment:
                    logger.info(
                        f"[BASILICA] Creating shared deployment for {runs_remaining} "
                        f"evaluation runs of {submission.submission_id}"
                    )
                    try:
                        deployment_ctx = await self.affinetes_runner.create_basilica_deployment()
                        logger.info(
                            f"[BASILICA] Shared deployment '{deployment_ctx.name}' ready "
                            f"— will run {runs_remaining} evals on same GPUs"
                        )
                    except Exception as deploy_err:
                        logger.error(f"[BASILICA] Failed to create shared deployment: {deploy_err}")
                        logger.info("[BASILICA] Falling back to per-run deployments")
                        deployment_ctx = None
                        use_shared_deployment = False

                for run_idx in range(runs_remaining):
                    current_run = len(my_evals) + run_idx + 1
                    seed = (
                        f"{submission.miner_uid}:{current_run}"
                        f":{int(time.time())}:{secrets.token_hex(16)}"
                    )

                    logger.info(f"Evaluation run {current_run}/{num_runs} (seed: {seed})")

                    hard_timeout = hparams.eval_timeout + 3000
                    try:
                        if deployment_ctx is not None:
                            result = await asyncio.wait_for(
                                self.affinetes_runner.evaluate_on_deployment(
                                    ctx=deployment_ctx,
                                    code=miner_code,
                                    seed=seed,
                                    steps=hparams.eval_steps,
                                    batch_size=hparams.benchmark_batch_size,
                                    sequence_length=hparams.benchmark_sequence_length,
                                    data_samples=hparams.benchmark_data_samples,
                                    task_id=current_run,
                                ),
                                timeout=hard_timeout,
                            )
                        else:
                            result = await asyncio.wait_for(
                                self.affinetes_runner.evaluate(
                                    code=miner_code,
                                    seed=seed,
                                    steps=hparams.eval_steps,
                                    batch_size=hparams.benchmark_batch_size,
                                    sequence_length=hparams.benchmark_sequence_length,
                                    data_samples=hparams.benchmark_data_samples,
                                    task_id=current_run,
                                ),
                                timeout=hard_timeout,
                            )
                    except TimeoutError:
                        logger.error(f"Run {current_run}: hard timeout after {hard_timeout}s")
                        result = EvaluationResult(
                            success=False,
                            error=f"Hard timeout after {hard_timeout}s",
                            error_code="HARD_TIMEOUT",
                            task_id=current_run,
                        )

                    if result.success:
                        logger.info(
                            f"Run {current_run} PASSED: MFU={result.mfu:.2f}% TPS={result.tps:,.2f}"
                        )
                    else:
                        logger.warning(f"Run {current_run} FAILED: {result.error}")

                    evaluation = EvaluationModel(
                        submission_id=submission.submission_id,
                        evaluator_hotkey=self.hotkey,
                        mfu=result.mfu,
                        tokens_per_second=result.tps,
                        total_tokens=result.total_tokens,
                        wall_time_seconds=result.wall_time_seconds,
                        success=result.success,
                        error=result.error,
                    )
                    await self.db.save_evaluation(evaluation)
                    if self._cleanup_memory_callback is not None:
                        self._cleanup_memory_callback()

                    if result.is_fatal():
                        logger.warning(
                            f"Fatal error detected ({result.error_code}), skipping remaining runs"
                        )
                        fatal_error = True
                        break

            finally:
                if deployment_ctx is not None:
                    logger.info(
                        f"[BASILICA] Tearing down shared deployment '{deployment_ctx.name}' "
                        f"after evaluation of {submission.submission_id}"
                    )
                    await self.affinetes_runner.destroy_basilica_deployment(deployment_ctx)
                    deployment_ctx = None

            # Persist state (URL already recorded during commitment processing)
            if self._save_state_callback is not None:
                await self._save_state_callback()

            # Store miner's code in database
            await self.db.update_submission_code(submission.submission_id, miner_code)
            logger.info(f"Stored code for {submission.submission_id} in database")

            # Finalize submission (pass fatal_error to skip unnecessary checks)
            await self._finalize_submission(
                submission.submission_id, num_runs, fatal_error=fatal_error
            )

    async def _finalize_submission(
        self, submission_id: str, num_runs: int, fatal_error: bool = False
    ) -> None:
        """Calculate final score (MFU) and update submission status.

        Args:
            submission_id: The submission to finalize
            num_runs: Expected number of evaluation runs
            fatal_error: If True, a deterministic failure was detected and
                         evaluation was stopped early - fail immediately
        """
        hparams = get_hparams()
        num_evals = await self.db.count_evaluations(submission_id)
        required_evals = num_runs

        logger.info(f"Finalizing {submission_id}: {num_evals}/{required_evals} evaluations")

        # If fatal error detected, fail immediately without checking success rate
        # Fatal errors are deterministic - retrying would give the same result
        if fatal_error:
            all_evals = await self.db.get_evaluations(submission_id)
            # Get the error message from the most recent failed evaluation
            failed_evals = [e for e in all_evals if not e.success and e.error]
            error_msg = failed_evals[-1].error if failed_evals else "Fatal error in evaluation"
            await self.db.update_submission_status(
                submission_id,
                SubmissionStatus.FAILED_EVALUATION,
                error_message=f"Fatal error (deterministic failure): {error_msg}",
            )
            logger.warning(f"Submission {submission_id} failed with fatal error: {error_msg}")
            return

        if num_evals >= required_evals:
            all_evals = await self.db.get_evaluations(submission_id)
            successful_evals = [e for e in all_evals if e.success]

            # Check minimum success rate
            success_rate = len(successful_evals) / len(all_evals) if all_evals else 0
            min_success_rate = hparams.min_success_rate

            if success_rate < min_success_rate:
                await self.db.update_submission_status(
                    submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message=(
                        f"Success rate {success_rate:.1%} "
                        f"below minimum {min_success_rate:.0%}"
                    ),
                )
                logger.warning(
                    f"Submission {submission_id} failed: "
                    f"success rate {success_rate:.1%} "
                    f"< {min_success_rate:.0%}"
                )
                return

            if successful_evals:
                # MFU is the primary metric now
                mfu_scores = [e.mfu for e in successful_evals]
                # Use median_low to always return an actual run value (not average of two)
                median_mfu = statistics.median_low(mfu_scores)

                logger.info(
                    f"Final score for {submission_id}:\n"
                    f"   Successful runs: {len(mfu_scores)}\n"
                    f"   MFU scores: {[f'{s:.2f}%' for s in sorted(mfu_scores)]}\n"
                    f"   Median MFU (low): {median_mfu:.2f}%"
                )

                await self.db.update_submission_score(submission_id, median_mfu)
                await self.db.update_submission_status(
                    submission_id,
                    SubmissionStatus.FINISHED,
                )
                logger.info(f"Submission {submission_id} FINISHED with MFU={median_mfu:.2f}%")

                # Check if this submission is the new leader and update threshold immediately
                # This provides immediate feedback on website/TUI instead of waiting for
                # the next weight-setting cycle (~20 minutes)
                try:
                    await self._check_and_update_threshold_if_new_leader(submission_id, median_mfu)
                except Exception as e:
                    logger.warning(f"Failed to check/update threshold (non-fatal): {e}")
                    # Continue - weight setter will handle this on next cycle
            else:
                # All evaluations failed - mark submission as failed with score 0
                await self.db.update_submission_score(submission_id, 0.0)
                await self.db.update_submission_status(
                    submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message="All evaluations failed",
                )
                logger.warning(f"Submission {submission_id} FAILED: all evaluations failed")

    async def _check_and_update_threshold_if_new_leader(
        self, submission_id: str, score: float
    ) -> None:
        """Update adaptive threshold immediately if this submission is the new leader.

        This provides immediate feedback on the website/TUI instead of waiting
        for the next weight-setting cycle (which can be ~20 minutes).

        Shares state with weight_setter via database to avoid duplicate updates.
        """
        hparams = get_hparams()
        threshold_config = hparams.adaptive_threshold

        # Cannot update without block number - would corrupt decay state
        if self.commitment_reader is None:
            logger.debug("Skipping immediate threshold update: no commitment_reader")
            return

        current_block = self.commitment_reader.get_current_block()

        # Get current threshold (with decay applied)
        current_threshold = await self.db.get_adaptive_threshold(
            current_block=current_block,
            base_threshold=threshold_config.base_threshold,
            decay_percent=threshold_config.decay_percent,
            decay_interval_blocks=threshold_config.decay_interval_blocks,
        )

        # Get the current leaderboard winner (after this submission was marked FINISHED)
        winner = await self.db.get_leaderboard_winner(
            threshold=current_threshold,
            spec_version=crusades.COMPETITION_VERSION,
        )

        if winner is None:
            return

        # Check if THIS submission is the new leader
        if winner.submission_id != submission_id:
            return  # Not the new leader, nothing to do

        # This submission is the new leader! Update threshold immediately.
        # Load previous winner from DB (shared state with weight setter)
        previous_winner_id = await self.db.get_validator_state("previous_winner_id")
        previous_winner_score_str = await self.db.get_validator_state("previous_winner_score")

        # Safe float conversion with error handling
        previous_winner_score = 0.0
        if previous_winner_score_str:
            try:
                previous_winner_score = float(previous_winner_score_str)
            except ValueError:
                logger.warning(f"Invalid previous_winner_score in DB: {previous_winner_score_str}")
                previous_winner_score = 0.0

        # Only update if this is a NEW leader (different from previous)
        if previous_winner_id == submission_id:
            return  # Same leader, threshold already updated

        # Save winner identity FIRST to prevent duplicate threshold updates.
        # If threshold update below fails, the next cycle will see this winner
        # as "already handled" and skip it. If we saved identity after threshold,
        # a failure between them would cause the threshold to be bumped twice.
        await self.db.set_validator_state("previous_winner_id", submission_id)
        await self.db.set_validator_state("previous_winner_score", str(score))

        # Update adaptive threshold
        if previous_winner_score > 0:
            new_threshold = await self.db.update_adaptive_threshold(
                new_score=score,
                old_score=previous_winner_score,
                current_block=current_block,
                base_threshold=threshold_config.base_threshold,
            )
            improvement = (score - previous_winner_score) / previous_winner_score * 100
            logger.info(
                f"NEW LEADER (immediate)! Threshold updated:\n"
                f"  - Previous: {previous_winner_score:.2f}% MFU\n"
                f"  - New: {score:.2f}% MFU (+{improvement:.1f}%)\n"
                f"  - New threshold: {new_threshold:.1%}"
            )
        else:
            # First winner ever
            await self.db.update_adaptive_threshold(
                new_score=score,
                old_score=0.0,
                current_block=current_block,
                base_threshold=threshold_config.base_threshold,
            )
            logger.info(f"First leader established (immediate): {score:.2f}% MFU")
