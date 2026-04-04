"""Integration tests for FastAPI endpoints using MockClient."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import crusades.api.server as server_module
from crusades.tui.client import MockClient


@pytest.fixture
def client():
    """TestClient using the module-level app with MockClient injected."""
    mock = MockClient()
    app = server_module.app
    # Ensure no API key is required
    app.state.api_key = None
    with patch.object(server_module, "_db_client", mock):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def authed_client():
    """TestClient with API key required."""
    mock = MockClient()
    app = server_module.app
    app.state.api_key = "test-secret-key"
    with patch.object(server_module, "_db_client", mock):
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    """Health check returns structured status."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "db_connected" in data

    def test_health_with_mock_client(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["db_connected"] is True

    def test_health_full_response_shape(self, client):
        """Healthy response includes all expected keys."""
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["db_connected"] is True
        assert "evaluations_1h" in data
        assert "queue_depth" in data
        assert "current_evaluation" in data
        assert "validator_status" in data
        # MockClient returns evaluations_completed_1h=18 → healthy
        assert data["evaluations_1h"] == 18

    def test_health_degraded_when_zero_evaluations(self):
        """Status is 'degraded' when evaluations_completed_1h is 0."""

        class StaleClient(MockClient):
            def get_validator_status(self):
                return {
                    "status": "running",
                    "evaluations_completed_1h": 0,
                    "current_evaluation": None,
                }

        app = server_module.app
        app.state.api_key = None
        with patch.object(server_module, "_db_client", StaleClient()):
            with TestClient(app) as c:
                resp = c.get("/health")
                data = resp.json()
                assert data["status"] == "degraded"
                assert data["db_connected"] is True
                assert data["evaluations_1h"] == 0

    def test_health_unhealthy_on_exception(self):
        """Status is 'unhealthy' when client methods raise."""

        class BrokenClient(MockClient):
            def get_validator_status(self):
                raise RuntimeError("DB gone")

        app = server_module.app
        app.state.api_key = None
        with patch.object(server_module, "_db_client", BrokenClient()):
            with TestClient(app) as c:
                resp = c.get("/health")
                data = resp.json()
                assert data["status"] == "unhealthy"
                assert data["db_connected"] is False

    def test_health_db_unavailable(self):
        """When db client is None, returns healthy with db_connected=False."""
        app = server_module.app
        app.state.api_key = None
        with patch.object(server_module, "_db_client", None):
            with patch.object(server_module, "get_db_client", return_value=None):
                with TestClient(app) as c:
                    resp = c.get("/health")
                    data = resp.json()
                    assert data["status"] == "healthy"
                    assert data["db_connected"] is False


class TestStatsEndpoints:
    """Stats endpoints return expected structures."""

    def test_overview(self, client):
        resp = client.get("/api/stats/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "submissions_24h" in data
        assert "current_top_score" in data
        assert "active_miners" in data

    def test_validator_status(self, client):
        resp = client.get("/api/stats/validator")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "evaluations_completed_1h" in data

    def test_recent_submissions(self, client):
        resp = client.get("/api/stats/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_recent_submissions_limit(self, client):
        resp = client.get("/api/stats/recent?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 5

    def test_history(self, client):
        resp = client.get("/api/stats/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_history_limit(self, client):
        resp = client.get("/api/stats/history?limit=10")
        assert resp.status_code == 200

    def test_queue_stats(self, client):
        resp = client.get("/api/stats/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert "queued_count" in data
        assert "running_count" in data
        assert "finished_count" in data

    def test_threshold(self, client):
        resp = client.get("/api/stats/threshold")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_threshold" in data
        assert "decayed_threshold" in data


class TestLeaderboard:
    """Leaderboard endpoint."""

    def test_leaderboard_default(self, client):
        resp = client.get("/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_leaderboard_with_limit(self, client):
        resp = client.get("/leaderboard?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 5


class TestSubmissionEndpoints:
    """Submission detail endpoints."""

    def test_submission_detail(self, client):
        # MockClient returns data for any submission_id
        resp = client.get("/api/submissions/test-sub-1")
        assert resp.status_code == 200

    def test_submission_not_found(self):
        """When get_submission returns empty dict, endpoint returns 404."""

        class EmptyClient(MockClient):
            def get_submission(self, submission_id):
                return {}

        app = server_module.app
        app.state.api_key = None
        with patch.object(server_module, "_db_client", EmptyClient()):
            with TestClient(app) as c:
                resp = c.get("/api/submissions/nonexistent")
                assert resp.status_code == 404

    def test_submission_evaluations(self, client):
        resp = client.get("/api/submissions/test-sub-1/evaluations")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_submission_code(self, client):
        resp = client.get("/api/submissions/test-sub-1/code")
        assert resp.status_code == 200
        data = resp.json()
        assert "code" in data


class TestAuthentication:
    """API key authentication."""

    def test_no_key_returns_401(self, authed_client):
        resp = authed_client.get("/health")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, authed_client):
        resp = authed_client.get("/health", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_correct_key_returns_200(self, authed_client):
        resp = authed_client.get("/health", headers={"X-API-Key": "test-secret-key"})
        assert resp.status_code == 200
