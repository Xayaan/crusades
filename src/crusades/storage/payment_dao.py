"""Data access object for payment verification operations."""

import logging

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import VerifiedPaymentModel

logger = logging.getLogger(__name__)


class PaymentDAO:
    """Handles all payment verification database operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def is_payment_used(self, block_hash: str, extrinsic_index: int) -> bool:
        """Check if a payment extrinsic has already been claimed by a submission."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(VerifiedPaymentModel)
                .where(
                    VerifiedPaymentModel.block_hash == block_hash,
                    VerifiedPaymentModel.extrinsic_index == extrinsic_index,
                )
            )
            return (result.scalar() or 0) > 0

    async def record_verified_payment(
        self,
        submission_id: str,
        miner_hotkey: str,
        miner_coldkey: str,
        block_hash: str,
        extrinsic_index: int,
        amount_rao: int,
    ) -> None:
        """Record a verified payment to prevent double-spend.

        Raises:
            ValueError: If the payment extrinsic was already claimed by another submission.
        """
        async with self._session_factory() as session:
            payment = VerifiedPaymentModel(
                submission_id=submission_id,
                miner_hotkey=miner_hotkey,
                miner_coldkey=miner_coldkey,
                block_hash=block_hash,
                extrinsic_index=extrinsic_index,
                amount_rao=amount_rao,
            )
            try:
                session.add(payment)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                raise ValueError(
                    f"Payment at block {block_hash[:16]}... extrinsic {extrinsic_index} "
                    f"already claimed by another submission"
                )

    async def get_payment_for_submission(self, submission_id: str) -> VerifiedPaymentModel | None:
        """Get the payment record for a submission."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(VerifiedPaymentModel).where(
                    VerifiedPaymentModel.submission_id == submission_id
                )
            )
            return result.scalar_one_or_none()
