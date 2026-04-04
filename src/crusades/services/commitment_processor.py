"""Processes blockchain commitments into submission records."""

import logging
from collections.abc import Callable

import bittensor as bt

import crusades
from crusades.chain.commitments import CommitmentReader, MinerCommitment
from crusades.chain.payment import (
    get_hotkey_owner,
    resolve_payment_address,
    verify_payment_direct_async,
)
from crusades.config import get_hparams
from crusades.core.protocols import SubmissionStatus
from crusades.storage.database import Database
from crusades.storage.models import SubmissionModel

logger = logging.getLogger(__name__)


class CommitmentProcessor:
    """Reads blockchain commitments, verifies payments, and creates submissions.

    Responsibilities:
    - Reading blockchain commitments
    - Deduplication via evaluated_code_urls
    - Rate limiting (min_blocks_between_commits check)
    - Payment verification
    - Creating submission records
    """

    def __init__(
        self,
        db: Database,
        commitment_reader: CommitmentReader,
        chain: object | None,  # ChainManager type
        hotkey: str,
        archive_subtensor_factory: Callable[[], bt.subtensor],
    ) -> None:
        self.db = db
        self.commitment_reader = commitment_reader
        self.chain = chain
        self.hotkey = hotkey
        self._archive_subtensor_factory = archive_subtensor_factory

        # Lazy-loaded archive subtensor
        self._archive_subtensor: bt.subtensor | None = None

        # State
        self.last_processed_block: int = 0
        # Map URL -> (reveal_block, hotkey) to track first committer
        # Pruned to max_evaluated_urls (from hparams) to prevent unbounded memory growth
        self.evaluated_code_urls: dict[str, tuple[int, str]] = {}

    @property
    def archive_subtensor(self) -> bt.subtensor:
        """Lazy-load archive subtensor for historical block lookups."""
        if self._archive_subtensor is None:
            self._archive_subtensor = self._archive_subtensor_factory()
        return self._archive_subtensor

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
                        f"Processing commitment from UID {commitment.uid}, "
                        f"hotkey: {commitment.hotkey[:16]}..."
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
                                "(first committed at block "
                                f"{first_block} by {first_hotkey[:16]}...)"
                            )
                            continue
                        else:
                            # This commitment is earlier - it should have priority
                            # This shouldn't happen if we process in order, but log it
                            logger.warning(
                                f"Found earlier commitment for URL from UID {commitment.uid} "
                                f"at block {commitment.reveal_block} "
                                f"(previously saw block {first_block})"
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

        burn_uid is exempt from payment -- it is the subnet operator's own
        UID and cannot be claimed by external miners.

        Returns:
            True if payment verified (or payments disabled), False otherwise
        """
        hparams = get_hparams()

        if not hparams.payment.enabled:
            logger.debug("Payment verification disabled in hparams")
            return True

        if commitment.uid == hparams.burn_uid:
            logger.info(
                f"Skipping payment for burn_uid {hparams.burn_uid} "
                f"(hotkey {commitment.hotkey[:16]}...)"
            )
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
                f"Payment at block {payment.block_hash[:16]}... "
                f"extrinsic {payment.extrinsic_index} "
                "already used for another submission"
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
                f"Payment at block {payment.block_hash[:16]}... "
                f"extrinsic {payment.extrinsic_index} "
                "was concurrently claimed by another submission"
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
                error_message=(
                    "Payment not verified: no valid "
                    "transfer_stake payment found on-chain"
                ),
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
