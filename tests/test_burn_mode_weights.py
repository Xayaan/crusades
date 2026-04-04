"""Tests for burn mode integration with WeightSetter."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crusades.chain.burn_mode import BurnMode
from crusades.chain.weights import WeightSetter
from crusades.storage.database import Database


@pytest.fixture
async def db():
    """In-memory async database."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.initialize()
    yield database
    await database.close()


def _mock_chain():
    """Create a mock ChainManager with plausible defaults."""
    chain = AsyncMock()
    chain.metagraph = MagicMock()
    chain.sync_metagraph = AsyncMock()
    chain.get_current_block = AsyncMock(return_value=1000)
    chain.is_registered = MagicMock(return_value=True)
    chain.get_uid_for_hotkey = MagicMock(return_value=10)
    chain.set_weights = AsyncMock(return_value=(True, "ok"))
    return chain


def _mock_submission(submission_id="sub-1", hotkey="hk-winner", score=75.0):
    """Create a mock submission."""
    sub = MagicMock()
    sub.submission_id = submission_id
    sub.miner_hotkey = hotkey
    sub.final_score = score
    return sub


class TestWeightSetterBurnMode:
    @patch("crusades.chain.weights.get_hparams")
    async def test_burn_mode_overrides_burn_rate(self, mock_hparams, db: Database):
        """When burn mode is active, its burn_rate_override replaces hparams burn_rate."""
        hparams = MagicMock()
        hparams.burn_rate = 0.95
        hparams.burn_uid = 1
        hparams.adaptive_threshold.base_threshold = 0.01
        hparams.adaptive_threshold.decay_percent = 0.05
        hparams.adaptive_threshold.decay_interval_blocks = 100
        mock_hparams.return_value = hparams

        # Activate burn mode with 100% burn
        await db.set_burn_mode(BurnMode(
            enabled=True,
            burn_rate_override=1.0,
            reason="exploit",
            activated_at=datetime.now(UTC),
        ))

        chain = _mock_chain()
        ws = WeightSetter(chain=chain, database=db)

        # Mock leaderboard winner
        winner = _mock_submission()
        db.get_leaderboard_winner = AsyncMock(return_value=winner)

        success, msg = await ws.set_weights()
        assert success

        # With 100% burn override, winner_weight=0 and burn_weight=1
        call_args = chain.set_weights.call_args
        weights = call_args.kwargs.get("weights") or call_args[1].get("weights")
        assert weights[0] == 1.0  # burn uid gets everything

    @patch("crusades.chain.weights.get_hparams")
    async def test_blocked_uid_triggers_burn_only(self, mock_hparams, db: Database):
        """When the winner UID is in blocked_uids, all emissions go to burn_uid."""
        hparams = MagicMock()
        hparams.burn_rate = 0.95
        hparams.burn_uid = 1
        hparams.adaptive_threshold.base_threshold = 0.01
        hparams.adaptive_threshold.decay_percent = 0.05
        hparams.adaptive_threshold.decay_interval_blocks = 100
        mock_hparams.return_value = hparams

        # Block UID 10 (the winner)
        await db.set_burn_mode(BurnMode(
            enabled=True,
            burn_rate_override=0.5,
            blocked_uids=[10],
            reason="UID 10 exploit",
            activated_at=datetime.now(UTC),
        ))

        chain = _mock_chain()
        chain.get_uid_for_hotkey.return_value = 10  # winner maps to blocked UID
        ws = WeightSetter(chain=chain, database=db)

        winner = _mock_submission()
        db.get_leaderboard_winner = AsyncMock(return_value=winner)

        success, msg = await ws.set_weights()
        assert success
        assert "blocked" in msg.lower()

        # Should set 100% to burn_uid
        call_args = chain.set_weights.call_args
        uids = call_args.kwargs.get("uids") or call_args[1].get("uids")
        weights = call_args.kwargs.get("weights") or call_args[1].get("weights")
        assert uids == [1]
        assert weights == [1.0]

    @patch("crusades.chain.weights.get_hparams")
    async def test_inactive_burn_mode_uses_hparams(self, mock_hparams, db: Database):
        """When burn mode is inactive, normal hparams burn_rate is used."""
        hparams = MagicMock()
        hparams.burn_rate = 0.95
        hparams.burn_uid = 1
        hparams.adaptive_threshold.base_threshold = 0.01
        hparams.adaptive_threshold.decay_percent = 0.05
        hparams.adaptive_threshold.decay_interval_blocks = 100
        mock_hparams.return_value = hparams

        # Burn mode not set — db returns inactive by default

        chain = _mock_chain()
        ws = WeightSetter(chain=chain, database=db)

        winner = _mock_submission()
        db.get_leaderboard_winner = AsyncMock(return_value=winner)

        success, msg = await ws.set_weights()
        assert success

        call_args = chain.set_weights.call_args
        weights = call_args.kwargs.get("weights") or call_args[1].get("weights")
        # burn_rate=0.95 → burn_weight=0.95, winner_weight=0.05
        assert abs(weights[0] - 0.95) < 0.001
        assert abs(weights[1] - 0.05) < 0.001
