"""FastAPI server for crusades dashboard.

Provides HTTP endpoints for the web dashboard to fetch crusades data.
Reads from the local SQLite database (crusades.db).

Usage:
    # Run the API server
    uv run -m crusades.api --port 8080

    # Or use the CLI
    crusades-api --port 8080

Endpoints:
    GET /health              - Health check
    GET /api/stats/overview  - Dashboard overview stats
    GET /api/stats/validator - Validator status
    GET /api/stats/recent    - Recent submissions
    GET /api/stats/history   - TPS history for charts
    GET /api/stats/queue     - Queue statistics
    GET /api/stats/threshold - Adaptive threshold stats
    GET /leaderboard         - Leaderboard entries
    GET /api/submissions/{id}            - Submission details
    GET /api/submissions/{id}/evaluations - Submission evaluations
    GET /api/submissions/{id}/code       - Submission code
"""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crusades import COMPETITION_VERSION
from crusades.chain.burn_mode import BurnMode
from crusades.storage.database import Database
from crusades.tui.client import DatabaseClient, MockClient

logger = logging.getLogger(__name__)

# Global database client
_db_client = None


def get_db_client():
    """Get or create database client (filtered by current competition version)."""
    global _db_client
    if _db_client is None:
        db_path = os.getenv("CRUSADES_DB_PATH", "crusades.db")
        competition_version = COMPETITION_VERSION
        if Path(db_path).exists():
            logger.info(f"Using database: {db_path} (version={competition_version})")
            _db_client = DatabaseClient(db_path, spec_version=competition_version)
        else:
            logger.warning(f"Database not found at {db_path}, using mock data")
            _db_client = MockClient()
    return _db_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    # Startup
    logger.info("Crusades API starting...")
    get_db_client()

    # Initialize async database for burn mode operations
    db_path = os.getenv("CRUSADES_DB_PATH", "crusades.db")
    db = Database(url=f"sqlite+aiosqlite:///{db_path}")
    await db.initialize()
    app.state.db = db

    yield

    # Shutdown
    await db.close()
    global _db_client
    if _db_client:
        _db_client.close()
        _db_client = None
    logger.info("Crusades API shutdown")


def verify_api_key(request: Request, x_api_key: str | None = Header(None)):
    """Verify API key if configured. Used as a global FastAPI dependency."""
    configured_key = request.app.state.api_key
    if configured_key:
        if not x_api_key or not hmac.compare_digest(x_api_key, configured_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ============================================================
# Burn Mode Router (defined before create_app so it can be included)
# ============================================================


class BurnModeActivateRequest(BaseModel):
    """Request body for burn mode activation."""

    burn_rate_override: float = Field(default=1.0, ge=0.0, le=1.0)
    blocked_uids: list[int] = Field(default_factory=list)
    reason: str = ""


burn_mode_router = APIRouter(prefix="/burn-mode", tags=["burn-mode"])


@burn_mode_router.post("/activate")
async def activate_burn_mode(
    request: Request,
    body: BurnModeActivateRequest,
) -> dict[str, Any]:
    """Activate burn mode. Overrides hparams burn_rate and optionally blocks UIDs."""
    db: Database = request.app.state.db
    burn_mode = BurnMode(
        enabled=True,
        burn_rate_override=body.burn_rate_override,
        blocked_uids=body.blocked_uids,
        reason=body.reason,
        activated_at=datetime.now(UTC),
        activated_by="api",
    )
    await db.set_burn_mode(burn_mode)
    logger.warning(
        "Burn mode ACTIVATED: rate=%.0f%%, blocked=%s, reason=%s",
        burn_mode.burn_rate_override * 100,
        burn_mode.blocked_uids,
        burn_mode.reason,
    )
    return burn_mode.model_dump(mode="json")


@burn_mode_router.post("/deactivate")
async def deactivate_burn_mode(request: Request) -> dict[str, Any]:
    """Deactivate burn mode. Restores normal hparams-driven weight distribution."""
    db: Database = request.app.state.db
    burn_mode = BurnMode.inactive()
    await db.set_burn_mode(burn_mode)
    logger.warning("Burn mode DEACTIVATED")
    return {"status": "deactivated"}


@burn_mode_router.get("/status")
async def get_burn_mode_status(request: Request) -> dict[str, Any]:
    """Get current burn mode state."""
    db: Database = request.app.state.db
    burn_mode = await db.get_burn_mode()
    return burn_mode.model_dump(mode="json")


def create_app(api_key: str | None = None) -> FastAPI:
    """Create and configure the FastAPI app.

    Args:
        api_key: Optional API key for authentication. If set, all requests
                 must include X-API-Key header.
    """
    new_app = FastAPI(
        title="Templar Crusades API",
        description="API for the Templar Crusades web dashboard",
        version="1.0.0",
        lifespan=lifespan,
        dependencies=[Depends(verify_api_key)],
    )

    # CORS - allow website to call API
    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to your domain
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Store API key for authentication
    new_app.state.api_key = api_key or os.getenv("DASHBOARD_API_KEY")

    # Include burn mode router
    new_app.include_router(burn_mode_router)

    return new_app


# Create default app instance
app = create_app()


# ============================================================
# Health Check
# ============================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================
# Stats Endpoints
# ============================================================


@app.get("/api/stats/overview")
async def get_overview() -> dict[str, Any]:
    """Get dashboard overview statistics.

    Returns:
        - submissions_24h: Submissions in last 24 hours
        - current_top_score: Current top TPS score
        - score_improvement_24h: Score improvement percentage
        - total_submissions: Total submissions all time
        - active_miners: Active miners in last 24h
    """
    client = get_db_client()
    return client.get_overview()


@app.get("/api/stats/validator")
async def get_validator_status() -> dict[str, Any]:
    """Get validator status.

    Returns:
        - status: "running" | "idle" | "unknown"
        - evaluations_completed_1h: Evaluations in last hour
        - current_evaluation: Current submission being evaluated
        - uptime: Validator uptime
        - queued_count, running_count, finished_count, failed_count
        - success_rate: Percentage of successful evaluations
    """
    client = get_db_client()
    return client.get_validator_status()


@app.get("/api/stats/recent")
async def get_recent_submissions(
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get recent submissions.

    Returns list of recent submissions with status and scores.
    """
    client = get_db_client()
    return client.get_recent_submissions()[:limit]


@app.get("/api/stats/history")
async def get_history(
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """Get TPS history for charts.

    Returns list of TPS scores over time for chart visualization.
    Returns the most recent `limit` data points (newest submissions).
    """
    client = get_db_client()
    history = client.get_history()
    # Return the most recent N data points (end of the list)
    return history[-limit:] if len(history) > limit else history


@app.get("/api/stats/queue")
async def get_queue_stats() -> dict[str, Any]:
    """Get queue statistics.

    Returns:
        - queued_count: Submissions waiting
        - running_count: Currently evaluating
        - finished_count: Completed successfully
        - failed_count: Failed submissions
        - avg_wait_time_seconds: Average wait time
        - avg_score: Average TPS score
    """
    client = get_db_client()
    return client.get_queue_stats()


@app.get("/api/stats/threshold")
async def get_threshold() -> dict[str, Any]:
    """Get adaptive threshold statistics.

    Returns:
        - current_threshold: Current threshold value
        - last_improvement: Last improvement ratio
        - last_update_block: Block when threshold was last updated
        - decayed_threshold: Current threshold after time-based decay
    """
    client = get_db_client()
    return client.get_adaptive_threshold()


# ============================================================
# Leaderboard Endpoint
# ============================================================


@app.get("/leaderboard")
async def get_leaderboard(
    limit: int = Query(50, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get leaderboard entries.

    Returns ranked list of top submissions by TPS score.
    """
    client = get_db_client()
    return client.get_leaderboard(limit=limit)


# ============================================================
# Submission Detail Endpoints
# ============================================================


@app.get("/api/submissions/{submission_id}")
async def get_submission(
    submission_id: str,
) -> dict[str, Any]:
    """Get submission details.

    Returns full details for a specific submission.
    """
    client = get_db_client()
    submission = client.get_submission(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    return submission


@app.get("/api/submissions/{submission_id}/evaluations")
async def get_submission_evaluations(
    submission_id: str,
) -> list[dict[str, Any]]:
    """Get evaluations for a submission.

    Returns list of all evaluation runs for the submission.
    """
    client = get_db_client()
    return client.get_submission_evaluations(submission_id)


class CodeResponse(BaseModel):
    """Response model for code endpoint."""

    code: str | None
    code_hash: str | None = None


@app.get("/api/submissions/{submission_id}/code", response_model=CodeResponse)
async def get_submission_code(
    submission_id: str,
) -> CodeResponse:
    """Get code for a submission.

    Returns the miner's train.py code for the submission.
    """
    client = get_db_client()
    code = client.get_submission_code(submission_id)

    # Get submission for code_hash
    submission = client.get_submission(submission_id)
    code_hash = submission.get("code_hash") if submission else None

    return CodeResponse(code=code, code_hash=code_hash)


# ============================================================
# CLI Entry Point
# ============================================================


def main():
    """Run the API server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Crusades API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--db", default="crusades.db", help="Path to database")
    parser.add_argument("--api-key", help="API key for authentication (optional)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Set environment variables
    os.environ["CRUSADES_DB_PATH"] = args.db
    # Set API key directly on the app instance (not via env var indirection)
    # to avoid the race where app is created at import-time before env var is set
    if args.api_key:
        app.state.api_key = args.api_key

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info(f"Starting Crusades API on {args.host}:{args.port}")
    logger.info(f"Database: {args.db}")
    logger.info(f"API Key: {'configured' if args.api_key else 'not configured (open access)'}")

    uvicorn.run(
        "crusades.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
