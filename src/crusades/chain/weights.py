"""Weight setting logic with configurable burn rate and adaptive threshold."""

import logging

from crusades import COMPETITION_VERSION

from ..config import get_hparams
from ..storage.database import Database
from .manager import ChainManager

logger = logging.getLogger(__name__)


class WeightSetter:
    """Handles setting weights on the Bittensor network.

    Implements burn_rate distribution:
    - burn_rate portion goes to burn_uid (configurable burn destination)
    - (1 - burn_rate) portion goes to top MFU winner (leaderboard rank 1)

    If no valid winner exists, all emissions go to burn_uid.

    The leaderboard uses an adaptive threshold that:
    - Increases when big improvements happen (rewards big jumps)
    - Decays over time towards base_threshold (1%)
    """

    def __init__(
        self,
        chain: ChainManager,
        database: Database,
    ):
        self.chain = chain
        self.db = database
        self.hparams = get_hparams()

        # Burn configuration from hparams
        self.burn_rate = self.hparams.burn_rate  # Fraction of emissions to burn_uid
        self.burn_uid = self.hparams.burn_uid  # UID that receives burn portion

        # Track previous winner to detect changes (loaded from DB each cycle)
        self._previous_winner_id: str | None = None
        self._previous_winner_score: float = 0.0

    async def _load_previous_winner(self) -> None:
        """Load previous winner info from database (for restart resilience)."""
        winner_id = await self.db.get_validator_state("previous_winner_id")
        winner_score_str = await self.db.get_validator_state("previous_winner_score")
        if winner_id and winner_score_str:
            self._previous_winner_id = winner_id
            try:
                self._previous_winner_score = float(winner_score_str)
            except ValueError:
                self._previous_winner_score = 0.0
            logger.info(
                f"Loaded previous winner from DB: {winner_id} ({self._previous_winner_score:.2f}% MFU)"
            )

    async def _save_previous_winner(self, winner_id: str, winner_score: float) -> None:
        """Save previous winner info to database (for restart resilience)."""
        await self.db.set_validator_state("previous_winner_id", winner_id)
        await self.db.set_validator_state("previous_winner_score", str(winner_score))

    async def set_weights(self) -> tuple[bool, str]:
        """Set weights based on leaderboard rank 1 with burn_rate distribution.

        Distribution:
        - burn_rate portion goes to burn_uid (burn destination)
        - (1 - burn_rate) portion goes to leaderboard rank 1

        The leaderboard applies an adaptive threshold that decays over time.
        Big improvements create high thresholds, rewarding significant jumps.

        If no valid winner, 100% goes to burn_uid.

        Returns:
            Tuple of (success, message)
        """
        # Load burn mode state — takes precedence over hparams burn_rate
        burn_mode = await self.db.get_burn_mode()
        if burn_mode.enabled:
            logger.warning(
                "Burn mode ACTIVE: rate_override=%.0f%%, blocked_uids=%s, reason=%s",
                burn_mode.burn_rate_override * 100,
                burn_mode.blocked_uids,
                burn_mode.reason,
            )

        # Sync metagraph to get latest state
        await self.chain.sync_metagraph()

        # Skip weight setting if metagraph sync failed
        if self.chain.metagraph is None:
            logger.warning("Metagraph not available - cannot set weights")
            logger.warning(
                "Possible causes: subtensor not running, network issues, or netuid doesn't exist"
            )
            return False, "Metagraph sync failed - cannot set weights"

        # Get current block for adaptive threshold
        current_block = await self.chain.get_current_block()

        # Get adaptive threshold config
        threshold_config = self.hparams.adaptive_threshold

        # Get current adaptive threshold (decays over time)
        current_threshold = await self.db.get_adaptive_threshold(
            current_block=current_block,
            base_threshold=threshold_config.base_threshold,
            decay_percent=threshold_config.decay_percent,
            decay_interval_blocks=threshold_config.decay_interval_blocks,
        )

        logger.info(f"Adaptive threshold: {current_threshold:.2%} at block {current_block}")

        # Get leaderboard rank 1 (with adaptive threshold, filtered by competition version)
        winner = await self.db.get_leaderboard_winner(
            threshold=current_threshold,
            spec_version=COMPETITION_VERSION,
        )

        # If no valid winner, all emissions go to burn_uid
        if winner is None:
            logger.info("No finished submissions - 100% to burn_uid")
            return await self._set_burn_only_weights("No finished submissions")

        # Verify miner is still registered
        winner_hotkey = winner.miner_hotkey
        if not self.chain.is_registered(winner_hotkey):
            logger.warning(f"Winner {winner_hotkey} not registered - 100% to burn_uid")
            return await self._set_burn_only_weights(f"Winner {winner_hotkey} not registered")

        # Get UID for winner
        winner_uid = self.chain.get_uid_for_hotkey(winner_hotkey)
        if winner_uid is None:
            logger.error(f"Could not get UID for {winner_hotkey} - 100% to burn_uid")
            return await self._set_burn_only_weights(f"Could not get UID for {winner_hotkey}")

        winner_score = winner.final_score or 0.0

        # Always load previous winner from DB to stay in sync with
        # immediate threshold updates from validator._check_and_update_threshold_if_new_leader()
        await self._load_previous_winner()

        # Race condition fix: the immediate update path may have already identified a
        # higher-scoring winner and bumped the threshold. When that happens, the leaderboard
        # query (using the new high threshold) may fail to recognize that winner because
        # the threshold was computed FROM their improvement. Trust the DB's previous_winner
        # if it has a higher score than the leaderboard query returned.
        if self._previous_winner_id is not None and self._previous_winner_score > winner_score:
            # The immediate update identified a better winner that the threshold-based
            # leaderboard query missed. Look up that submission to use it instead.
            db_winner = await self.db.get_submission(self._previous_winner_id)
            if db_winner is not None and db_winner.miner_hotkey:
                db_hotkey = db_winner.miner_hotkey
                if self.chain.is_registered(db_hotkey):
                    db_uid = self.chain.get_uid_for_hotkey(db_hotkey)
                    if db_uid is not None:
                        logger.info(
                            f"Using immediate-update winner over leaderboard query: "
                            f"{self._previous_winner_id} ({self._previous_winner_score:.2f}% MFU) "
                            f"> leaderboard winner ({winner_score:.2f}% MFU)"
                        )
                        winner = db_winner
                        winner_hotkey = db_hotkey
                        winner_uid = db_uid
                        winner_score = self._previous_winner_score

        # If no previous winner in DB, initialize with current winner
        if self._previous_winner_id is None:
            self._previous_winner_id = winner.submission_id
            self._previous_winner_score = winner_score
            await self._save_previous_winner(winner.submission_id, winner_score)
            logger.info(
                f"Initialized winner tracking (first time): "
                f"{self._previous_winner_id} ({self._previous_winner_score:.2f}% MFU)"
            )

        # Check if winner changed - if so, update adaptive threshold
        if winner.submission_id != self._previous_winner_id:
            # Update adaptive threshold
            if self._previous_winner_score > 0:
                new_threshold = await self.db.update_adaptive_threshold(
                    new_score=winner_score,
                    old_score=self._previous_winner_score,
                    current_block=current_block,
                    base_threshold=threshold_config.base_threshold,
                )
                improvement = (
                    (winner_score - self._previous_winner_score) / self._previous_winner_score * 100
                )
                logger.info(
                    f"NEW LEADER! Threshold updated:\n"
                    f"  - Previous: {self._previous_winner_score:.2f}% MFU\n"
                    f"  - New: {winner_score:.2f}% MFU (+{improvement:.1f}%)\n"
                    f"  - New threshold: {new_threshold:.1%}"
                )
            else:
                # First winner ever - initialize with base threshold
                await self.db.update_adaptive_threshold(
                    new_score=winner_score,
                    old_score=0.0,
                    current_block=current_block,
                    base_threshold=threshold_config.base_threshold,
                )
                logger.info(f"First winner established: {winner_score:.2f}% MFU")

            # Update tracking (in memory and DB)
            self._previous_winner_id = winner.submission_id
            self._previous_winner_score = winner_score
            await self._save_previous_winner(winner.submission_id, winner_score)

        # If burn mode is active and winner is blocked, redirect to burn
        if burn_mode.enabled and winner_uid in burn_mode.blocked_uids:
            logger.warning(
                "Winner UID %d is blocked by burn mode — redirecting 100%% to burn_uid",
                winner_uid,
            )
            return await self._set_burn_only_weights(
                f"Winner UID {winner_uid} blocked by burn mode"
            )

        # Calculate weight distribution (burn mode overrides hparams burn_rate)
        effective_burn_rate = (
            burn_mode.burn_rate_override if burn_mode.enabled else self.burn_rate
        )
        winner_weight = 1.0 - effective_burn_rate
        burn_weight = effective_burn_rate

        logger.info(
            f"Setting weights with burn_rate={effective_burn_rate:.0%}:\n"
            f"  - UID {self.burn_uid} (burn): {burn_weight:.2f}\n"
            f"  - UID {winner_uid} (winner, MFU={winner_score:.2f}%): {winner_weight:.2f}"
        )

        # Set weights for both burn_uid and winner
        uids = [self.burn_uid, winner_uid]
        weights = [burn_weight, winner_weight]

        # Handle case where winner IS the burn_uid (unlikely but possible)
        if winner_uid == self.burn_uid:
            uids = [self.burn_uid]
            weights = [1.0]
            logger.info(f"Winner is burn_uid - setting 100% to UID {self.burn_uid}")

        success, message = await self.chain.set_weights(
            uids=uids,
            weights=weights,
        )

        if success:
            logger.info(f"Weights set successfully: {dict(zip(uids, weights))}")
        else:
            logger.error(f"Failed to set weights: {message}")

        return (success, message)

    async def _set_burn_only_weights(self, reason: str) -> tuple[bool, str]:
        """Set 100% weights to burn_uid when no valid winner exists.

        Args:
            reason: Why we're giving all to burn_uid (for logging)

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Setting 100% weight to burn_uid {self.burn_uid} ({reason})")

        success, message = await self.chain.set_weights(
            uids=[self.burn_uid],
            weights=[1.0],
        )

        if success:
            logger.info(f"Weights set successfully: UID {self.burn_uid} -> 100%")
        else:
            logger.error(f"Failed to set weights: {message}")

        return (success, f"Burn only ({reason}): {message}" if success else message)
