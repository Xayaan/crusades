"""Tests for scripts/health_monitor.py."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path so we can import health_monitor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import health_monitor


class TestEvaluateHealth:
    """Table-driven tests for evaluate_health — all four return paths."""

    @pytest.mark.parametrize(
        "data, expected_healthy, expected_reason_contains",
        [
            # Error key present → unhealthy
            ({"error": "Unreachable: connection refused"}, False, "Unreachable"),
            # Status unhealthy → unhealthy
            ({"status": "unhealthy"}, False, "DB disconnected"),
            # Status degraded → unhealthy with queue info
            (
                {"status": "degraded", "queue_depth": 12},
                False,
                "0 evaluations",
            ),
            # Healthy → healthy
            (
                {"status": "healthy", "evaluations_1h": 5, "queue_depth": 0},
                True,
                "healthy",
            ),
        ],
        ids=["error", "unhealthy", "degraded", "healthy"],
    )
    def test_evaluate_health(self, data, expected_healthy, expected_reason_contains):
        is_healthy, reason = health_monitor.evaluate_health(data)
        assert is_healthy is expected_healthy
        assert expected_reason_contains in reason


class TestPostDiscord:
    """Discord webhook posting."""

    def test_noop_when_no_webhook_url(self):
        """post_discord is a no-op when DISCORD_WEBHOOK_URL is empty."""
        with patch.object(health_monitor, "DISCORD_WEBHOOK_URL", ""):
            # Should return without error and without making any network calls
            health_monitor.post_discord("test alert")

    def test_posts_embed_to_webhook(self):
        """post_discord sends a JSON embed to the configured webhook."""
        fake_url = "https://discord.com/api/webhooks/fake"
        with patch.object(health_monitor, "DISCORD_WEBHOOK_URL", fake_url):
            with patch.object(health_monitor, "urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__ = MagicMock()
                mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

                health_monitor.post_discord("validator down", color=0xFF0000)

                mock_urlopen.assert_called_once()
                req = mock_urlopen.call_args[0][0]
                assert req.full_url == fake_url
                body = json.loads(req.data.decode())
                assert body["embeds"][0]["description"] == "validator down"
                assert body["embeds"][0]["color"] == 0xFF0000


class TestProbeHealth:
    """probe_health I/O logic."""

    def test_returns_parsed_json_on_success(self):
        """Successful probe returns parsed JSON from the endpoint."""
        fake_body = json.dumps({"status": "healthy", "evaluations_1h": 5}).encode()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = fake_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(health_monitor, "urlopen", return_value=mock_resp):
            result = health_monitor.probe_health()

        assert result == {"status": "healthy", "evaluations_1h": 5}

    def test_returns_error_on_url_error(self):
        """URLError produces a dict with 'error' key."""
        from urllib.error import URLError

        with patch.object(
            health_monitor, "urlopen", side_effect=URLError("connection refused")
        ):
            result = health_monitor.probe_health()

        assert "error" in result
        assert "Unreachable" in result["error"]

    def test_returns_error_on_generic_exception(self):
        """Any other exception produces a dict with 'error' key."""
        with patch.object(
            health_monitor, "urlopen", side_effect=ValueError("bad data")
        ):
            result = health_monitor.probe_health()

        assert "error" in result
        assert "bad data" in result["error"]
