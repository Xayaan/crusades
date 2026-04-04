#!/usr/bin/env python3
"""Health monitor for crusades validator with Discord alerting.

Probes the validator's /health endpoint on a configurable interval.
Posts to a Discord webhook when the validator is unreachable, returning
non-200, or stale (zero evaluations in the staleness window).

Sends recovery notifications when the validator returns to healthy after
a failure period.

Configuration via environment variables:
    VALIDATOR_URL           - Base URL of the validator API (default: http://localhost:8080)
    DISCORD_WEBHOOK_URL     - Discord webhook for alerts (optional; omit for log-only mode)
    CHECK_INTERVAL_SECONDS  - Seconds between health checks (default: 60)
    STALE_THRESHOLD_SECONDS - Seconds of zero evaluations before alerting (default: 3600)
    ALERT_COOLDOWN_SECONDS  - Minimum seconds between repeat alerts (default: 300)
"""

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from urllib.error import URLError
from urllib.request import Request, urlopen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("health_monitor")

VALIDATOR_URL = os.getenv("VALIDATOR_URL", "http://localhost:8080").rstrip("/")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))
STALE_THRESHOLD = int(os.getenv("STALE_THRESHOLD_SECONDS", "3600"))
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))


def probe_health() -> dict:
    """Probe the validator /health endpoint.

    Returns:
        Parsed JSON response, or a dict with 'error' key on failure.
    """
    url = f"{VALIDATOR_URL}/health"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}"}
            return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": f"Unreachable: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def post_discord(content: str, color: int = 0xFF0000) -> None:
    """Post an embed to Discord webhook. Fails silently if no webhook configured."""
    if not DISCORD_WEBHOOK_URL:
        return

    payload = json.dumps({
        "embeds": [{
            "title": "Crusades Validator Health Alert",
            "description": content,
            "color": color,
            "timestamp": datetime.now(UTC).isoformat(),
        }]
    }).encode()

    try:
        req = Request(
            DISCORD_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10):
            pass
    except Exception as e:
        logger.error(f"Discord webhook failed: {e}")


def evaluate_health(data: dict) -> tuple[bool, str]:
    """Evaluate health response.

    Returns:
        (is_healthy, reason)
    """
    if "error" in data:
        return False, data["error"]

    status = data.get("status", "unknown")
    if status == "unhealthy":
        return False, "Validator reports unhealthy (DB disconnected)"

    if status == "degraded":
        return False, (
            f"Validator is degraded: 0 evaluations in last hour, "
            f"queue_depth={data.get('queue_depth', '?')}"
        )

    return True, "healthy"


def run() -> None:
    """Main monitoring loop."""
    logger.info(f"Monitoring {VALIDATOR_URL}/health every {CHECK_INTERVAL}s")
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set - running in log-only mode")

    is_alerting = False
    last_alert_time = 0.0

    while True:
        data = probe_health()
        healthy, reason = evaluate_health(data)

        if healthy:
            if is_alerting:
                # Recovery
                logger.info("Validator recovered")
                post_discord(
                    f"Validator at `{VALIDATOR_URL}` has **recovered**.\n"
                    f"Status: `{data.get('status', 'healthy')}`\n"
                    f"Evaluations/1h: `{data.get('evaluations_1h', 'N/A')}`",
                    color=0x00FF00,
                )
                is_alerting = False
            else:
                logger.info(f"OK | evals_1h={data.get('evaluations_1h', '?')} "
                            f"queue={data.get('queue_depth', '?')}")
        else:
            now = time.time()
            if not is_alerting or (now - last_alert_time) >= ALERT_COOLDOWN:
                logger.error(f"ALERT | {reason}")
                post_discord(
                    f"Validator at `{VALIDATOR_URL}` is **down or degraded**.\n"
                    f"Reason: `{reason}`\n"
                    f"Checked at: `{datetime.now(UTC).isoformat()}`",
                    color=0xFF0000,
                )
                is_alerting = True
                last_alert_time = now
            else:
                logger.warning(f"Still failing: {reason} (cooldown active)")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Monitor stopped")
        sys.exit(0)
