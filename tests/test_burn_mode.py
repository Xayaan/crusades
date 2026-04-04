"""Tests for burn mode model and database persistence."""

from datetime import UTC, datetime

import pytest

from crusades.chain.burn_mode import BURN_MODE_KEY, BurnMode
from crusades.storage.database import Database

# -- BurnMode model tests --


class TestBurnModeModel:
    def test_inactive_factory(self):
        bm = BurnMode.inactive()
        assert bm.enabled is False
        assert bm.burn_rate_override == 1.0
        assert bm.blocked_uids == []
        assert bm.reason == ""
        assert bm.activated_at is None

    def test_serialization_roundtrip(self):
        bm = BurnMode(
            enabled=True,
            burn_rate_override=0.5,
            blocked_uids=[15, 42],
            reason="exploit detected",
            activated_at=datetime(2026, 1, 1, tzinfo=UTC),
            activated_by="operator",
        )
        raw = bm.model_dump_json()
        restored = BurnMode.model_validate_json(raw)
        assert restored.enabled is True
        assert restored.burn_rate_override == 0.5
        assert restored.blocked_uids == [15, 42]
        assert restored.reason == "exploit detected"
        assert restored.activated_by == "operator"

    def test_burn_rate_bounds_low(self):
        with pytest.raises(Exception):
            BurnMode(burn_rate_override=-0.1)

    def test_burn_rate_bounds_high(self):
        with pytest.raises(Exception):
            BurnMode(burn_rate_override=1.1)

    def test_burn_rate_at_boundaries(self):
        bm_zero = BurnMode(burn_rate_override=0.0)
        assert bm_zero.burn_rate_override == 0.0
        bm_one = BurnMode(burn_rate_override=1.0)
        assert bm_one.burn_rate_override == 1.0


# -- Database persistence tests --


@pytest.fixture
async def db():
    """In-memory async database for testing."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.initialize()
    yield database
    await database.close()


class TestBurnModePersistence:
    async def test_get_returns_inactive_when_unset(self, db: Database):
        bm = await db.get_burn_mode()
        assert bm.enabled is False

    async def test_set_then_get_roundtrip(self, db: Database):
        original = BurnMode(
            enabled=True,
            burn_rate_override=1.0,
            blocked_uids=[15],
            reason="skip backward exploit",
            activated_at=datetime(2026, 4, 4, tzinfo=UTC),
            activated_by="api",
        )
        await db.set_burn_mode(original)
        loaded = await db.get_burn_mode()
        assert loaded.enabled is True
        assert loaded.burn_rate_override == 1.0
        assert loaded.blocked_uids == [15]
        assert loaded.reason == "skip backward exploit"

    async def test_overwrite_existing(self, db: Database):
        await db.set_burn_mode(BurnMode(enabled=True, blocked_uids=[1, 2]))
        await db.set_burn_mode(BurnMode.inactive())
        loaded = await db.get_burn_mode()
        assert loaded.enabled is False
        assert loaded.blocked_uids == []

    async def test_persists_via_validator_state(self, db: Database):
        """Burn mode is stored as a validator_state KV entry."""
        bm = BurnMode(enabled=True, reason="test")
        await db.set_burn_mode(bm)
        raw = await db.get_validator_state(BURN_MODE_KEY)
        assert raw is not None
        assert '"enabled":true' in raw.lower() or '"enabled": true' in raw.lower()
