"""Integration tests for burn mode API endpoints."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from crusades.api.server import burn_mode_router, verify_api_key
from crusades.chain.burn_mode import BurnMode
from crusades.storage.database import Database

API_KEY = "test-secret-key"


@pytest.fixture
async def db():
    """In-memory async database."""
    database = Database(url="sqlite+aiosqlite:///:memory:")
    await database.initialize()
    yield database
    await database.close()


def _test_app(db: Database) -> FastAPI:
    """Build a minimal FastAPI app with burn mode routes and injected DB."""

    @asynccontextmanager
    async def _noop_lifespan(app: FastAPI):
        yield

    from fastapi import Depends

    test_app = FastAPI(
        lifespan=_noop_lifespan,
        dependencies=[Depends(verify_api_key)],
    )
    test_app.state.api_key = API_KEY
    test_app.state.db = db
    test_app.include_router(burn_mode_router)
    return test_app


@pytest.fixture
async def client(db: Database):
    """Test client with authenticated API and wired database."""
    app = _test_app(db)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def auth_headers():
    return {"X-API-Key": API_KEY}


class TestBurnModeActivate:
    async def test_activate_success(self, client: AsyncClient):
        resp = await client.post(
            "/burn-mode/activate",
            json={
                "burn_rate_override": 1.0,
                "blocked_uids": [15],
                "reason": "UID 15 skip backward exploit",
            },
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["burn_rate_override"] == 1.0
        assert data["blocked_uids"] == [15]
        assert data["reason"] == "UID 15 skip backward exploit"
        assert data["activated_at"] is not None

    async def test_activate_without_api_key_rejected(self, client: AsyncClient):
        resp = await client.post(
            "/burn-mode/activate",
            json={"reason": "no auth"},
        )
        assert resp.status_code == 401

    async def test_activate_invalid_burn_rate(self, client: AsyncClient):
        resp = await client.post(
            "/burn-mode/activate",
            json={"burn_rate_override": 1.5},
            headers=auth_headers(),
        )
        assert resp.status_code == 422

    async def test_activate_defaults(self, client: AsyncClient):
        resp = await client.post(
            "/burn-mode/activate",
            json={},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["burn_rate_override"] == 1.0
        assert data["blocked_uids"] == []


class TestBurnModeDeactivate:
    async def test_deactivate_success(self, client: AsyncClient, db: Database):
        await db.set_burn_mode(
            BurnMode(enabled=True, reason="test", activated_at=datetime.now(UTC))
        )
        resp = await client.post("/burn-mode/deactivate", headers=auth_headers())
        assert resp.status_code == 200
        assert resp.json()["status"] == "deactivated"

        # Verify persisted
        bm = await db.get_burn_mode()
        assert bm.enabled is False

    async def test_deactivate_without_api_key_rejected(self, client: AsyncClient):
        resp = await client.post("/burn-mode/deactivate")
        assert resp.status_code == 401


class TestBurnModeStatus:
    async def test_status_returns_inactive_by_default(self, client: AsyncClient):
        resp = await client.get("/burn-mode/status", headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    async def test_status_reflects_activation(self, client: AsyncClient):
        await client.post(
            "/burn-mode/activate",
            json={"blocked_uids": [15, 42], "reason": "exploit"},
            headers=auth_headers(),
        )
        resp = await client.get("/burn-mode/status", headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["blocked_uids"] == [15, 42]
