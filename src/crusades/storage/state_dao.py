"""Data access object for validator state and adaptive threshold operations."""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import AdaptiveThresholdModel, ValidatorStateModel

logger = logging.getLogger(__name__)


class StateDAO:
    """Handles validator state and adaptive threshold database operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_validator_state(self, key: str) -> str | None:
        """Get a validator state value."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ValidatorStateModel.value).where(ValidatorStateModel.key == key)
            )
            return result.scalar_one_or_none()

    async def set_validator_state(self, key: str, value: str) -> None:
        """Set a validator state value (upsert)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ValidatorStateModel).where(ValidatorStateModel.key == key)
            )
            state = result.scalar_one_or_none()
            if state:
                state.value = value
            else:
                session.add(ValidatorStateModel(key=key, value=value))
            await session.commit()

    async def get_adaptive_threshold(
        self,
        current_block: int,
        base_threshold: float = 0.01,
        decay_percent: float = 0.05,
        decay_interval_blocks: int = 100,
    ) -> float:
        """Get the current adaptive threshold with decay applied.

        The threshold decays towards base_threshold over time.
        Each interval, it loses decay_percent (5%) of the excess above base.
        Formula: threshold = base + (current - base) * (1 - decay_percent)^steps

        Args:
            current_block: Current block number
            base_threshold: Minimum threshold (default 1%)
            decay_percent: Percent of excess to lose per interval (default 5%)
            decay_interval_blocks: Blocks between decay steps

        Returns:
            Current threshold value
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(AdaptiveThresholdModel).where(AdaptiveThresholdModel.id == 1)
            )
            state = result.scalar_one_or_none()

            if state is None:
                return base_threshold

            # Calculate decay based on blocks elapsed
            # Each step loses decay_percent of the excess above base
            blocks_elapsed = max(0, current_block - state.last_update_block)

            # Guard against misconfigured decay_interval_blocks (avoid division by zero)
            if decay_interval_blocks <= 0:
                return state.current_threshold

            decay_steps = blocks_elapsed / decay_interval_blocks
            decay_factor = (1.0 - decay_percent) ** decay_steps

            # Decay from current threshold towards base
            decayed = base_threshold + (state.current_threshold - base_threshold) * decay_factor
            return max(base_threshold, decayed)

    async def update_adaptive_threshold(
        self,
        new_score: float,
        old_score: float,
        current_block: int,
        base_threshold: float = 0.01,
    ) -> float:
        """Update threshold when a new leader is established.

        Threshold = improvement percentage (no multiplier).
        e.g., if new leader is 20% better, threshold becomes 20%.

        Args:
            new_score: New leader's score
            old_score: Previous leader's score
            current_block: Current block number
            base_threshold: Minimum threshold

        Returns:
            New threshold value
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(AdaptiveThresholdModel).where(AdaptiveThresholdModel.id == 1)
            )
            state = result.scalar_one_or_none()

            # Calculate improvement ratio
            if old_score > 0:
                improvement = (new_score - old_score) / old_score
            else:
                improvement = base_threshold  # First submission, use base

            # New threshold = improvement (no cap)
            new_threshold = max(base_threshold, improvement)

            if state is None:
                # Create new state
                state = AdaptiveThresholdModel(
                    id=1,
                    current_threshold=new_threshold,
                    last_improvement=improvement,
                    last_update_block=current_block,
                )
                session.add(state)
            else:
                # Update existing state
                state.current_threshold = new_threshold
                state.last_improvement = improvement
                state.last_update_block = current_block

            await session.commit()
            return new_threshold
