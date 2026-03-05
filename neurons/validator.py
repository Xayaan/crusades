"""Crusades validator - evaluates miner submissions and sets weights.

URL-Based Architecture:
1. Reads code URL commitments from blockchain (timelock decrypted)
2. Downloads miner's train.py from the committed URL
3. Evaluates via affinetes (Docker locally or Basilica remotely)
4. Sets weights based on MFU (Model FLOPs Utilization) scores
"""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import secrets
import statistics
import time
import urllib.error
import urllib.request
from typing import Literal

import bittensor as bt
import torch

import crusades
from crusades.affinetes import AffinetesRunner
from crusades.chain.commitments import (
    CodeUrlInfo,
    CommitmentReader,
    MinerCommitment,
)
from crusades.chain.payment import (
    get_hotkey_owner,
    resolve_payment_address,
    verify_payment_direct_async,
)
from crusades.chain.weights import WeightSetter
from crusades.config import get_config, get_hparams
from crusades.core.protocols import SubmissionStatus
from crusades.logging import setup_loki_logger
from crusades.storage.database import Database, get_database
from crusades.storage.models import EvaluationModel, SubmissionModel

from .base_node import BaseNode

logger = logging.getLogger(__name__)


class Validator(BaseNode):
    """Crusades validator node (URL-Based Architecture).

    Responsibilities:
    1. Read miner code URL commitments from blockchain
    2. Download train.py from URL and evaluate via affinetes
    3. Calculate and set weights (winner-takes-all)
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
        skip_blockchain_check: bool = False,
        affinetes_mode: Literal["docker", "basilica"] = "docker",
    ):
        super().__init__(wallet=wallet, skip_blockchain_check=skip_blockchain_check)
        self.affinetes_mode = affinetes_mode

        # Components (initialized in start)
        self.db: Database | None = None
        self.weight_setter: WeightSetter | None = None
        self.commitment_reader: CommitmentReader | None = None
        self.affinetes_runner: AffinetesRunner | None = None

        # Archive subtensor for payment verification (standard nodes prune state
        # after ~256 blocks, making historical block lookups fail)
        self._archive_subtensor: bt.subtensor | None = None

        # State
        self.last_processed_block: int = 0
        # Map URL -> (reveal_block, hotkey) to track first committer
        # Pruned to max_evaluated_urls (from hparams) to prevent unbounded memory growth
        self.evaluated_code_urls: dict[str, tuple[int, str]] = {}

        # Timing
        self.last_weight_set_block: int = 0
        self.last_sync_time: float = 0

        # Memory cleanup tracking
        self._loop_count: int = 0

    @property
    def archive_subtensor(self) -> bt.subtensor:
        """Lazy-load archive subtensor for historical block lookups."""
        if self._archive_subtensor is None:
            hparams = get_hparams()
            endpoint = hparams.payment.archive_endpoint
            logger.info(f"Connecting to archive node: {endpoint}")
            self._archive_subtensor = bt.subtensor(
                network=endpoint,
            )
        return self._archive_subtensor

    async def initialize(self) -> None:
        """Initialize validator components."""
        config = get_config()
        hparams = get_hparams()

        # Setup Loki logging for Grafana dashboard
        uid_str = str(self.uid) if self.uid is not None else "unknown"
        loki_logger = setup_loki_logger(
            service="crusades-validator",
            uid=uid_str,
            version=crusades.__version__,
            environment=config.subtensor_network or "finney",
        )
        # Use the loki_logger (which is the "crusades" logger) as the parent
        # for the __main__ logger so all logs go through a single handler chain.
        logger.parent = loki_logger
        logger.propagate = True

        # Setup signal handlers within the running event loop
        self.setup_signal_handlers()

        logger.info("Initializing validator (URL-Based Architecture)")

        # Database
        self.db = await get_database()

        # Weight setter and commitment reader
        if self.chain is not None:
            self.weight_setter = WeightSetter(
                chain=self.chain,
                database=self.db,
            )

            self.commitment_reader = CommitmentReader(
                subtensor=self.chain.subtensor,
                netuid=hparams.netuid,
                network=config.subtensor_network,
            )
            self.commitment_reader.sync()

            # Sync chain's metagraph first, then initialize weight block
            # This prevents spamming weight setting attempts on restart
            await self.chain.sync_metagraph()
            await self._init_weight_block_from_chain()
        else:
            logger.warning("Running without blockchain connection")
            logger.warning("Weight setting will be disabled")
            # Still create commitment reader - will connect lazily
            self.commitment_reader = CommitmentReader(
                subtensor=None,
                netuid=hparams.netuid,
                network=config.subtensor_network,
            )

        # Affinetes runner - all config comes from validated Pydantic models
        self.affinetes_runner = AffinetesRunner(
            mode=self.affinetes_mode,
            basilica_api_key=os.getenv("BASILICA_API_TOKEN"),
            # Docker config
            docker_memory_limit=hparams.docker.memory_limit,
            docker_shm_size=hparams.docker.shm_size,
            num_gpus=hparams.docker.num_gpus,
            # Benchmark config
            model_url=hparams.benchmark_model_name,
            data_url=hparams.benchmark_dataset_name,
            timeout=hparams.eval_timeout,
            max_loss_difference=hparams.verification.max_loss_difference,
            min_params_changed_ratio=hparams.verification.min_params_changed_ratio,
            # Gradient verification
            gradient_norm_ratio_max=hparams.verification.gradient_norm_ratio_max,
            # Weight verification
            weight_relative_error_max=hparams.verification.weight_relative_error_max,
            # Timer integrity
            timer_divergence_threshold=hparams.verification.timer_divergence_threshold,
            # MFU calculation
            gpu_peak_tflops=hparams.mfu.gpu_peak_tflops,
            max_plausible_mfu=hparams.mfu.max_plausible_mfu,
            min_mfu=hparams.mfu.min_mfu,
            # Basilica config
            basilica_image=hparams.basilica.image,
            basilica_ttl_seconds=hparams.basilica.ttl_seconds,
            basilica_gpu_count=hparams.basilica.gpu_count,
            basilica_gpu_models=hparams.basilica.gpu_models,
            basilica_min_gpu_memory_gb=hparams.basilica.min_gpu_memory_gb,
            basilica_cpu=hparams.basilica.cpu,
            basilica_memory=hparams.basilica.memory,
            basilica_interconnect=hparams.basilica.interconnect,
            basilica_geo=hparams.basilica.geo,
            basilica_spot=hparams.basilica.spot,
        )

        # Load persisted state
        await self._load_state()

        logger.info(f"   Affinetes mode: {self.affinetes_mode}")
        logger.info(f"   Model: {hparams.benchmark_model_name}")
        logger.info(f"   Dataset: {hparams.benchmark_dataset_name}")
        if self.affinetes_mode == "docker":
            logger.info(f"   Docker num_gpus: {hparams.docker.num_gpus}")
            logger.info(f"   Docker memory limit: {hparams.docker.memory_limit}")
            logger.info(f"   Docker shm size: {hparams.docker.shm_size}")
        elif self.affinetes_mode == "basilica":
            logger.info(f"   Basilica image: {hparams.basilica.image}")
            logger.info(f"   Basilica TTL: {hparams.basilica.ttl_seconds}s")
            logger.info(
                f"   Basilica GPU: {hparams.basilica.gpu_count}x {hparams.basilica.gpu_models}"
            )
            logger.info(f"   Basilica min GPU memory: {hparams.basilica.min_gpu_memory_gb}GB")

    async def start(self) -> None:
        """Start the validator."""
        await self.initialize()
        await super().start()

    async def run_step(self) -> None:
        """Run one iteration of the validator loop."""
        logger.info("Starting validation loop iteration...")

        # 1. Read blockchain commitments
        logger.info("Step 1: Reading blockchain commitments...")
        await self.process_blockchain_commitments()

        # 2. Evaluate via affinetes
        logger.info("Step 2: Evaluating via affinetes...")
        await self.evaluate_submissions()

        # 3. Set weights
        logger.info("Step 3: Checking weight setting...")
        await self.maybe_set_weights()

        # 4. Sync metagraph
        logger.info("Step 4: Checking metagraph sync...")
        await self.maybe_sync()

        # Memory cleanup
        self._cleanup_memory()

        logger.info("Loop iteration complete. Sleeping 10s...")
        await asyncio.sleep(10)

    async def process_blockchain_commitments(self) -> None:
        """Read and process new URL commitments from blockchain."""
        try:
            new_commitments = self.commitment_reader.get_new_commitments_since(
                self.last_processed_block
            )

            current_block = self.commitment_reader.get_current_block()

            if new_commitments:
                logger.info(f"Found {len(new_commitments)} new commitments")

                for commitment in new_commitments:
                    logger.info(
                        f"Processing commitment from UID {commitment.uid}, hotkey: {commitment.hotkey[:16]}..."
                    )
                    logger.info(f"   Has valid URL: {commitment.has_valid_code_url()}")

                    # Skip if no valid code URL
                    if not commitment.has_valid_code_url():
                        logger.warning(
                            f"Skipping commitment without valid code URL: UID {commitment.uid}"
                        )
                        continue

                    # Use code URL as unique identifier - first committer wins
                    code_url = commitment.code_url_info.url
                    if code_url in self.evaluated_code_urls:
                        first_block, first_hotkey = self.evaluated_code_urls[code_url]
                        if commitment.reveal_block >= first_block:
                            logger.info(
                                f"Skipping duplicate URL from UID {commitment.uid} "
                                f"(first committed at block {first_block} by {first_hotkey[:16]}...)"
                            )
                            continue
                        else:
                            # This commitment is earlier - it should have priority
                            # This shouldn't happen if we process in order, but log it
                            logger.warning(
                                f"Found earlier commitment for URL from UID {commitment.uid} "
                                f"at block {commitment.reveal_block} (previously saw block {first_block})"
                            )

                    try:
                        await self._create_submission_from_commitment(commitment)
                        # Only record URL if submission was successfully saved
                        self.evaluated_code_urls[code_url] = (
                            commitment.reveal_block,
                            commitment.hotkey,
                        )
                    except Exception as e:
                        logger.exception(f"Failed to create submission for {code_url[:60]}: {e}")
                        # Don't record URL - will retry on next cycle

            self.last_processed_block = current_block
            await self._save_state()

        except Exception as e:
            logger.exception(f"Error processing commitments: {e}")

    async def _verify_submission_payment(
        self,
        commitment: MinerCommitment,
        submission_id: str,
    ) -> bool:
        """Verify that the miner has paid the submission fee via alpha transfer.

        Requires the commitment to contain a payment extrinsic reference
        (block + index) for O(1) direct on-chain lookup.  Commitments
        without the reference are rejected.

        Returns:
            True if payment verified (or payments disabled), False otherwise
        """
        hparams = get_hparams()

        if not hparams.payment.enabled:
            logger.debug("Payment verification disabled in hparams")
            return True

        if self.chain is None:
            logger.error("No chain connection — cannot verify payment")
            return False

        netuid = hparams.netuid

        # Use explicit payment_address if configured, otherwise derive from burn_uid
        if hparams.payment.payment_address:
            payment_address = hparams.payment.payment_address
            logger.debug(f"Using explicit payment_address: {payment_address[:16]}...")
        else:
            payment_address = resolve_payment_address(
                self.chain.subtensor, netuid, hparams.burn_uid
            )
            if payment_address is None:
                logger.error(
                    "Could not resolve payment address from burn_uid — cannot verify payment"
                )
                return False

        # Look up miner's coldkey. Try archive node for historical accuracy
        # first, fall back to regular subtensor if archive is unavailable.
        archive_sub = self.archive_subtensor
        miner_coldkey = None
        if commitment.payment_block:
            try:
                miner_coldkey = get_hotkey_owner(
                    archive_sub, commitment.hotkey, block=commitment.payment_block
                )
            except Exception as e:
                logger.debug(
                    f"Historical hotkey lookup failed at block {commitment.payment_block}: {e}"
                )
        if miner_coldkey is None:
            miner_coldkey = get_hotkey_owner(archive_sub, commitment.hotkey)
        if miner_coldkey is None:
            miner_coldkey = get_hotkey_owner(self.chain.subtensor, commitment.hotkey)
        if miner_coldkey is None:
            logger.error(
                f"Could not look up coldkey for hotkey {commitment.hotkey[:16]}... "
                f"- cannot verify payment"
            )
            return False

        logger.info(
            f"Verifying payment for {commitment.hotkey[:16]}... "
            f"(coldkey: {miner_coldkey[:16]}..., dest: {payment_address[:16]}...)"
        )

        # Convert fee_rao (TAO) to expected alpha using the subnet AMM rate.
        min_alpha = 0
        try:
            subnet_info = self.chain.subtensor.subnet(netuid=netuid)
            expected_alpha, _ = subnet_info.tao_to_alpha_with_slippage(
                bt.Balance.from_rao(hparams.payment.fee_rao)
            )
            min_alpha = int(expected_alpha.rao * 0.90)
            logger.debug(
                f"Payment min_alpha: {min_alpha} "
                f"(fee_rao={hparams.payment.fee_rao}, expected_alpha={expected_alpha.rao})"
            )
        except Exception as e:
            logger.warning(f"Could not query AMM rate, skipping amount check: {e}")

        # O(1) direct lookup using the extrinsic reference embedded in the
        # commitment.  Miners must include the payment ref; commitments
        # without one are rejected.
        if not commitment.has_payment_ref:
            logger.warning(
                f"Commitment from {commitment.hotkey[:16]}... has no payment "
                f"extrinsic reference — rejecting"
            )
            return False

        logger.info(
            f"Direct payment lookup: block {commitment.payment_block} "
            f"extrinsic {commitment.payment_extrinsic_index}"
        )
        payment = await verify_payment_direct_async(
            subtensor=archive_sub,
            block_number=commitment.payment_block,
            extrinsic_index=commitment.payment_extrinsic_index,
            miner_coldkey=miner_coldkey,
            payment_address=payment_address,
            netuid=netuid,
            min_amount=min_alpha,
            rpc_timeout=hparams.payment.rpc_timeout,
            rpc_retries=hparams.payment.rpc_retries,
        )

        # Fall back to regular subtensor if archive node failed (the block
        # may still be within the pruning window on the standard node).
        if payment is None:
            logger.info("Archive verification failed, retrying with standard subtensor")
            payment = await verify_payment_direct_async(
                subtensor=self.chain.subtensor,
                block_number=commitment.payment_block,
                extrinsic_index=commitment.payment_extrinsic_index,
                miner_coldkey=miner_coldkey,
                payment_address=payment_address,
                netuid=netuid,
                min_amount=min_alpha,
                rpc_timeout=hparams.payment.rpc_timeout,
                rpc_retries=hparams.payment.rpc_retries,
            )

        if payment is None:
            logger.warning(
                f"No valid payment found for {commitment.hotkey[:16]}... "
                f"(required transfer_stake to {payment_address[:16]}... on netuid {netuid})"
            )
            return False

        # Check for double-spend, but allow idempotent retries.
        already_used = await self.db.is_payment_used(payment.block_hash, payment.extrinsic_index)
        if already_used:
            existing_payment = await self.db.get_payment_for_submission(submission_id)
            if (
                existing_payment
                and existing_payment.block_hash == payment.block_hash
                and existing_payment.extrinsic_index == payment.extrinsic_index
            ):
                logger.info(
                    f"Payment for {submission_id} already recorded (idempotent retry) — reusing"
                )
                return True
            logger.warning(
                f"Payment at block {payment.block_hash[:16]}... extrinsic {payment.extrinsic_index} "
                f"already used for another submission"
            )
            return False

        # Record the payment (unique constraint guards against concurrent claims)
        try:
            await self.db.record_verified_payment(
                submission_id=submission_id,
                miner_hotkey=commitment.hotkey,
                miner_coldkey=miner_coldkey,
                block_hash=payment.block_hash,
                extrinsic_index=payment.extrinsic_index,
                amount_rao=payment.alpha_amount,
            )
        except ValueError:
            # Concurrent insert won the race — check if it was for this same submission
            existing_payment = await self.db.get_payment_for_submission(submission_id)
            if (
                existing_payment
                and existing_payment.block_hash == payment.block_hash
                and existing_payment.extrinsic_index == payment.extrinsic_index
            ):
                logger.info(f"Payment for {submission_id} recorded by concurrent task — reusing")
                return True
            logger.warning(
                f"Payment at block {payment.block_hash[:16]}... extrinsic {payment.extrinsic_index} "
                f"was concurrently claimed by another submission"
            )
            return False

        logger.info(
            f"Payment verified: {payment.alpha_amount} alpha at block "
            f"{payment.block_hash[:16]}... extrinsic {payment.extrinsic_index}"
        )
        return True

    async def _create_submission_from_commitment(
        self,
        commitment: MinerCommitment,
    ) -> None:
        """Create a submission record from a blockchain commitment."""
        hparams = get_hparams()
        version = crusades.COMPETITION_VERSION
        submission_id = f"v{version}_commit_{commitment.reveal_block}_{commitment.uid}"

        try:
            existing = await self.db.get_submission(submission_id)
            if existing:
                logger.debug(f"Submission {submission_id} already exists")
                return
        except Exception as e:
            logger.error(f"Database error checking existing submission: {e}")
            return  # Don't proceed if we can't check for duplicates

        # Rate limiting
        min_blocks = hparams.min_blocks_between_commits
        last_submission = await self.db.get_latest_submission_by_hotkey(commitment.hotkey)

        if last_submission:
            try:
                # Handle both old (commit_block_uid) and new (vN_commit_block_uid) formats
                parts = last_submission.submission_id.split("_")
                if parts[0].startswith("v"):
                    # New format: v3_commit_79639_1
                    last_block = int(parts[2])
                else:
                    # Old format: commit_79639_1
                    last_block = int(parts[1])
                blocks_since = commitment.reveal_block - last_block

                if blocks_since < min_blocks:
                    logger.warning(
                        f"Rate limit: {commitment.hotkey[:16]}... submitted too soon "
                        f"({blocks_since} blocks, min={min_blocks}). Skipping."
                    )
                    return
            except (IndexError, ValueError):
                logger.warning(
                    f"Failed to parse submission_id for rate limit check: "
                    f"{last_submission.submission_id}. Applying rate limit defensively."
                )
                return

        # Verify payment before creating the submission
        payment_verified = await self._verify_submission_payment(commitment, submission_id)

        if not payment_verified:
            # Create submission but mark it as failed
            failed_submission = SubmissionModel(
                submission_id=submission_id,
                miner_hotkey=commitment.hotkey,
                miner_uid=commitment.uid,
                code_hash=commitment.code_url_info.code_hash or "",
                bucket_path=commitment.code_url_info.url,
                status=SubmissionStatus.FAILED_EVALUATION,
                payment_verified=False,
                spec_version=crusades.COMPETITION_VERSION,
                error_message="Payment not verified: no valid transfer_stake payment found on-chain",
            )
            try:
                await self.db.save_submission(failed_submission)
                logger.warning(f"Submission {submission_id} FAILED: payment not verified")
            except Exception:
                logger.exception(f"Failed to save failed submission {submission_id}")
            return

        submission = SubmissionModel(
            submission_id=submission_id,
            miner_hotkey=commitment.hotkey,
            miner_uid=commitment.uid,
            code_hash=commitment.code_url_info.code_hash or "",
            bucket_path=commitment.code_url_info.url,
            status=SubmissionStatus.EVALUATING,
            payment_verified=True,
            spec_version=crusades.COMPETITION_VERSION,
        )

        try:
            await self.db.save_submission(submission)
            logger.info(f"Created submission: {submission_id}")
            logger.info(f"   Code URL: {commitment.code_url_info.url[:60]}...")
            logger.info(f"   UID: {commitment.uid}")
            logger.info(f"   Hotkey: {commitment.hotkey[:16]}...")
            logger.info("   Payment: verified")
        except Exception as e:
            logger.error(f"Failed to save submission: {e}")
            logger.exception("Traceback:")
            raise  # Propagate so caller doesn't mark URL as processed

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
        from urllib.parse import urlparse, urlunparse

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
            if sub_idx > 0:
                await self.maybe_set_weights()

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

            # Run evaluations — wrapped in try/finally so the Basilica deployment
            # is always deleted even if an unexpected exception occurs mid-eval.
            fatal_error = False
            try:
                for run_idx in range(runs_remaining):
                    current_run = len(my_evals) + run_idx + 1
                    seed = f"{submission.miner_uid}:{current_run}:{int(time.time())}:{secrets.token_hex(16)}"

                    logger.info(f"Evaluation run {current_run}/{num_runs} (seed: {seed})")

                    result = await self.affinetes_runner.evaluate(
                        code=miner_code,  # Pass code directly
                        seed=seed,
                        steps=hparams.eval_steps,
                        batch_size=hparams.benchmark_batch_size,
                        sequence_length=hparams.benchmark_sequence_length,
                        data_samples=hparams.benchmark_data_samples,
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
                        mfu=result.mfu,  # MFU is primary metric
                        tokens_per_second=result.tps,
                        total_tokens=result.total_tokens,
                        wall_time_seconds=result.wall_time_seconds,
                        success=result.success,
                        error=result.error,
                    )
                    await self.db.save_evaluation(evaluation)
                    self._cleanup_memory()

                    if result.is_fatal():
                        logger.warning(
                            f"Fatal error detected ({result.error_code}), skipping remaining runs"
                        )
                        fatal_error = True
                        break

                # Persist state (URL already recorded during commitment processing)
                await self._save_state()

                # Store miner's code in database
                await self.db.update_submission_code(submission.submission_id, miner_code)
                logger.info(f"Stored code for {submission.submission_id} in database")

                # Finalize submission (pass fatal_error to skip unnecessary checks)
                await self._finalize_submission(
                    submission.submission_id, num_runs, fatal_error=fatal_error
                )
            finally:
                if self.affinetes_mode == "basilica":
                    await self.affinetes_runner.delete_basilica_deployment()

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
                    error_message=f"Success rate {success_rate:.1%} below minimum {min_success_rate:.0%}",
                )
                logger.warning(
                    f"Submission {submission_id} failed: success rate {success_rate:.1%} < {min_success_rate:.0%}"
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

    def _cleanup_memory(self):
        """Clean up GPU and system memory."""
        self._loop_count += 1

        if self._loop_count % 10 == 0:
            logger.info(f"Memory cleanup (iteration {self._loop_count})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def maybe_sync(self) -> None:
        """Sync metagraph periodically."""
        now = time.time()
        if now - self.last_sync_time >= 300:
            await self.sync()
            if self.commitment_reader:
                self.commitment_reader.sync()
            self.last_sync_time = now

    async def _refresh_weight_block_from_chain(self) -> None:
        """Refresh last_weight_set_block from chain's metagraph.

        This ensures we have the latest last_update from chain before
        checking if we can set weights.
        """
        if self.chain is None:
            return

        try:
            # Sync metagraph to get latest state
            await self.chain.sync_metagraph()

            if self.chain.metagraph is None:
                return

            # Find our UID in the metagraph
            hotkey = self.chain.hotkey
            if hotkey not in self.chain.metagraph.hotkeys:
                return

            uid = self.chain.metagraph.hotkeys.index(hotkey)
            chain_last_update = int(self.chain.metagraph.last_update[uid])

            # Update if chain shows a more recent value
            if chain_last_update > self.last_weight_set_block:
                logger.debug(
                    f"Updating last_weight_set_block from chain: "
                    f"{self.last_weight_set_block} -> {chain_last_update}"
                )
                self.last_weight_set_block = chain_last_update
        except Exception as e:
            logger.debug(f"Could not refresh weight block from chain: {e}")

    async def maybe_set_weights(self) -> None:
        """Set weights if enough blocks have passed since last update."""
        if self.weight_setter is None:
            logger.debug("Skipping weight setting (test mode)")
            return

        if self.commitment_reader is None:
            logger.warning("Commitment reader not initialized - cannot set weights")
            return

        # Sync metagraph and refresh last_weight_set_block from chain
        # This ensures we don't attempt weight setting if chain shows recent update
        await self._refresh_weight_block_from_chain()

        hparams = get_hparams()
        current_block = self.commitment_reader.get_current_block()
        blocks_since_last = current_block - self.last_weight_set_block
        min_blocks = hparams.set_weights_interval_blocks

        if blocks_since_last <= min_blocks:
            # Don't log every time - only when we actually attempted
            return

        logger.info(
            f"Setting weights (block {current_block}, "
            f"{blocks_since_last} blocks since last update)..."
        )
        success, message = await self.weight_setter.set_weights()

        if success:
            self.last_weight_set_block = current_block
            logger.info(f"Weights set successfully at block {current_block}: {message}")
        else:
            # Provide detailed error info
            # Chain requires > min_blocks, so next allowed is last + min_blocks + 1
            next_allowed_block = self.last_weight_set_block + min_blocks + 1
            blocks_to_wait = max(0, next_allowed_block - current_block)
            logger.warning(
                f"Failed to set weights: {message}\n"
                f"  Current block: {current_block}\n"
                f"  Last successful: block {self.last_weight_set_block}\n"
                f"  Min interval: {min_blocks} blocks (chain requires >)\n"
                f"  Next allowed: block {next_allowed_block} ({blocks_to_wait} blocks to wait)"
            )

    async def _init_weight_block_from_chain(self) -> None:
        """Initialize last_weight_set_block from chain's metagraph.

        This prevents spamming weight setting attempts on validator restart
        by checking when the chain says we last set weights.
        """
        if self.chain is None or self.chain.metagraph is None:
            return

        try:
            # Find our UID in the metagraph
            hotkey = self.chain.hotkey
            if hotkey not in self.chain.metagraph.hotkeys:
                logger.warning("Validator hotkey not found in metagraph")
                return

            uid = self.chain.metagraph.hotkeys.index(hotkey)
            last_update = int(self.chain.metagraph.last_update[uid])

            if last_update > 0:
                self.last_weight_set_block = last_update
                logger.info(f"Initialized last_weight_set_block from chain: {last_update}")
        except Exception as e:
            logger.warning(f"Could not init weight block from chain: {e}")

    async def _load_state(self) -> None:
        """Load persisted validator state from database."""
        current_version = crusades.COMPETITION_VERSION

        try:
            # Check if version changed - if so, reset state for fresh competition
            stored_version_str = await self.db.get_validator_state("competition_version")
            stored_version = int(stored_version_str) if stored_version_str else 0

            if stored_version != current_version:
                # Get current block to start fresh from NOW (ignore old commitments)
                try:
                    current_block = self.chain.subtensor.get_current_block()
                except Exception as e:
                    # Cannot determine current block - defer version reset to avoid
                    # reprocessing all historical commitments from block 0
                    logger.warning(
                        f"Competition version changed ({stored_version} -> {current_version}), "
                        f"but cannot get current block: {e}. Deferring reset until chain is available."
                    )
                    return

                logger.info(
                    f"Competition version changed ({stored_version} -> {current_version}), "
                    f"starting fresh from block {current_block}"
                )
                # Keep last_processed_block at current block (ignore old commitments)
                self.last_processed_block = current_block
                # Clear evaluated URLs (allow same URLs in new version)
                self.evaluated_code_urls = {}
                # Save new version
                await self.db.set_validator_state("competition_version", str(current_version))
                return

            # Load last processed block
            block_str = await self.db.get_validator_state("last_processed_block")
            if block_str:
                self.last_processed_block = int(block_str)
                logger.info(f"Loaded state: last_processed_block={self.last_processed_block}")

            # Load evaluated code URLs
            urls_json = await self.db.get_validator_state("evaluated_code_urls")
            if urls_json:
                loaded = json.loads(urls_json)
                # Handle both old format (list of URLs) and new format (dict)
                if isinstance(loaded, list):
                    # Old format: convert to dict with placeholder values
                    self.evaluated_code_urls = {url: (0, "unknown") for url in loaded}
                elif isinstance(loaded, dict):
                    # New format: dict mapping URL -> [reveal_block, hotkey]
                    self.evaluated_code_urls = {url: tuple(info) for url, info in loaded.items()}
                else:
                    self.evaluated_code_urls = {}
                logger.info(f"Loaded state: {len(self.evaluated_code_urls)} evaluated URLs")
        except Exception as e:
            logger.warning(f"Failed to load state (starting fresh): {e}")
            self.last_processed_block = 0
            self.evaluated_code_urls = {}

    def _prune_evaluated_urls(self) -> None:
        """Prune evaluated_code_urls to prevent unbounded memory growth."""
        hparams = get_hparams()
        limit = hparams.max_evaluated_urls
        if len(self.evaluated_code_urls) > limit:
            sorted_urls = sorted(
                self.evaluated_code_urls.items(), key=lambda x: x[1][0], reverse=True
            )
            self.evaluated_code_urls = dict(sorted_urls[:limit])

    async def _save_state(self) -> None:
        """Persist validator state to database."""
        self._prune_evaluated_urls()
        try:
            await self.db.set_validator_state(
                "competition_version", str(crusades.COMPETITION_VERSION)
            )
            await self.db.set_validator_state(
                "last_processed_block", str(self.last_processed_block)
            )
            await self.db.set_validator_state(
                "evaluated_code_urls",
                json.dumps({url: list(info) for url, info in self.evaluated_code_urls.items()}),
            )
        except Exception:
            logger.exception("Failed to save state")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.affinetes_mode == "basilica" and self.affinetes_runner:
            await self.affinetes_runner.delete_basilica_deployment()
        await self._save_state()
        await super().cleanup()
        if self.db:
            await self.db.close()


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser(description="Crusades Validator (URL-Based)")
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="default",
        help="Wallet name",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Wallet hotkey",
    )
    parser.add_argument(
        "--skip-blockchain-check",
        action="store_true",
        help="Skip blockchain registration check",
    )
    parser.add_argument(
        "--affinetes-mode",
        type=str,
        choices=["docker", "basilica"],
        default="docker",
        help="Execution mode: docker (local) or basilica (remote GPU)",
    )

    args = parser.parse_args()

    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    validator = Validator(
        wallet=wallet,
        skip_blockchain_check=args.skip_blockchain_check,
        affinetes_mode=args.affinetes_mode,
    )

    logger.info("Starting validator")
    logger.info(f"   Affinetes mode: {args.affinetes_mode}")

    asyncio.run(validator.start())


if __name__ == "__main__":
    main()
