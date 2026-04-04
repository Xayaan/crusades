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
import json
import logging
import os
import time
from typing import Literal

import bittensor as bt
import torch

import crusades
from crusades.affinetes import AffinetesRunner
from crusades.chain.commitments import CommitmentReader
from crusades.chain.weights import WeightSetter
from crusades.config import get_config, get_hparams
from crusades.logging import setup_loki_logger
from crusades.services import CommitmentProcessor, SubmissionEvaluator
from crusades.storage.database import Database, get_database

from .base_node import BaseNode

logger = logging.getLogger(__name__)


class Validator(BaseNode):
    """Crusades validator node (URL-Based Architecture).

    Thin orchestrator that delegates to:
    - CommitmentProcessor: blockchain commitment reading, payment verification
    - SubmissionEvaluator: code download, evaluation, scoring
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

        # Service layer
        self.commitment_processor: CommitmentProcessor | None = None
        self.submission_evaluator: SubmissionEvaluator | None = None

        # Archive subtensor for payment verification (standard nodes prune state
        # after ~256 blocks, making historical block lookups fail)
        self._archive_subtensor: bt.subtensor | None = None

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

    def _create_archive_subtensor(self) -> bt.subtensor:
        """Factory method for creating archive subtensor instances."""
        return self.archive_subtensor

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

        # Initialize service layer
        self.commitment_processor = CommitmentProcessor(
            db=self.db,
            commitment_reader=self.commitment_reader,
            chain=self.chain,
            hotkey=self.hotkey,
            archive_subtensor_factory=self._create_archive_subtensor,
        )

        self.submission_evaluator = SubmissionEvaluator(
            db=self.db,
            affinetes_runner=self.affinetes_runner,
            hotkey=self.hotkey,
            commitment_reader=self.commitment_reader,
            weight_setter_callback=self.maybe_set_weights,
            cleanup_memory_callback=self._cleanup_memory,
            save_state_callback=self._save_state,
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
        await self.commitment_processor.process_blockchain_commitments()
        await self._save_state()

        # 2. Evaluate via affinetes
        logger.info("Step 2: Evaluating via affinetes...")
        await self.submission_evaluator.evaluate_submissions()

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
        try:
            current_block = self.commitment_reader.get_current_block()
        except (TimeoutError, OSError, ConnectionError) as e:
            logger.warning(
                f"Could not get current block for weight setting (will retry next loop): {e}"
            )
            return
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
                        f"Competition version changed "
                        f"({stored_version} -> {current_version})"
                        f", but cannot get current block: {e}. "
                        "Deferring reset until chain is available."
                    )
                    return

                logger.info(
                    f"Competition version changed ({stored_version} -> {current_version}), "
                    f"starting fresh from block {current_block}"
                )
                # Keep last_processed_block at current block (ignore old commitments)
                self.commitment_processor.last_processed_block = current_block
                # Clear evaluated URLs (allow same URLs in new version)
                self.commitment_processor.evaluated_code_urls = {}
                # Save new version
                await self.db.set_validator_state("competition_version", str(current_version))
                return

            # Load last processed block
            block_str = await self.db.get_validator_state("last_processed_block")
            if block_str:
                self.commitment_processor.last_processed_block = int(block_str)
                logger.info(
                    "Loaded state: last_processed_block="
                    f"{self.commitment_processor.last_processed_block}"
                )

            # Load evaluated code URLs
            urls_json = await self.db.get_validator_state("evaluated_code_urls")
            if urls_json:
                loaded = json.loads(urls_json)
                # Handle both old format (list of URLs) and new format (dict)
                if isinstance(loaded, list):
                    # Old format: convert to dict with placeholder values
                    self.commitment_processor.evaluated_code_urls = {
                        url: (0, "unknown") for url in loaded
                    }
                elif isinstance(loaded, dict):
                    # New format: dict mapping URL -> [reveal_block, hotkey]
                    self.commitment_processor.evaluated_code_urls = {
                        url: tuple(info)
                        for url, info in loaded.items()
                    }
                else:
                    self.commitment_processor.evaluated_code_urls = {}
                logger.info(
                    f"Loaded state: "
                    f"{len(self.commitment_processor.evaluated_code_urls)}"
                    " evaluated URLs"
                )
        except Exception as e:
            logger.warning(f"Failed to load state (starting fresh): {e}")
            self.commitment_processor.last_processed_block = 0
            self.commitment_processor.evaluated_code_urls = {}

    def _prune_evaluated_urls(self) -> None:
        """Prune evaluated_code_urls to prevent unbounded memory growth."""
        hparams = get_hparams()
        limit = hparams.max_evaluated_urls
        urls = self.commitment_processor.evaluated_code_urls
        if len(urls) > limit:
            sorted_urls = sorted(
                urls.items(), key=lambda x: x[1][0], reverse=True
            )
            self.commitment_processor.evaluated_code_urls = dict(sorted_urls[:limit])

    async def _save_state(self) -> None:
        """Persist validator state to database."""
        self._prune_evaluated_urls()
        try:
            await self.db.set_validator_state(
                "competition_version", str(crusades.COMPETITION_VERSION)
            )
            await self.db.set_validator_state(
                "last_processed_block", str(self.commitment_processor.last_processed_block)
            )
            await self.db.set_validator_state(
                "evaluated_code_urls",
                json.dumps({
                    url: list(info)
                    for url, info in
                    self.commitment_processor.evaluated_code_urls.items()
                }),
            )
        except Exception:
            logger.exception("Failed to save state")

    async def cleanup(self) -> None:
        """Cleanup resources."""
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
