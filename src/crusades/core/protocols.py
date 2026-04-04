"""Protocol definitions for templar-crusades.

These protocols define contracts between components, enabling loose coupling
and easy testing/mocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from crusades.storage.models import EvaluationModel, SubmissionModel


class SubmissionStatus(StrEnum):
    """Status of a code submission."""

    # In-progress statuses
    PENDING = "pending"  # Just submitted, awaiting validation
    VALIDATING = "validating"  # Being validated (syntax, imports, functions)
    EVALUATING = "evaluating"  # Passed validation, being evaluated in sandbox

    # Final statuses (shown in recent submissions)
    FINISHED = "finished"  # Evaluation complete, has final MFU score
    FAILED_VALIDATION = (
        "failed_validation"  #  Code validation failed (syntax, imports, missing function)
    )
    FAILED_EVALUATION = (
        "failed_evaluation"  #  Sandbox ran but verification failed (logits mismatch, timeout)
    )
    ERROR = "error"  #  Unexpected system error


@runtime_checkable
class Submission(Protocol):
    """Protocol for submission data."""

    @property
    def submission_id(self) -> str: ...

    @property
    def miner_hotkey(self) -> str: ...

    @property
    def miner_uid(self) -> int: ...

    @property
    def code_hash(self) -> str: ...

    @property
    def status(self) -> SubmissionStatus: ...

    @property
    def created_at(self) -> datetime: ...

    @property
    def final_score(self) -> float | None: ...


@runtime_checkable
class EvaluationResult(Protocol):
    """Protocol for evaluation result data."""

    @property
    def submission_id(self) -> str: ...

    @property
    def evaluator_hotkey(self) -> str: ...

    @property
    def tokens_per_second(self) -> float: ...

    @property
    def total_tokens(self) -> int: ...

    @property
    def wall_time_seconds(self) -> float: ...

    @property
    def success(self) -> bool: ...

    @property
    def error(self) -> str | None: ...

    @property
    def timestamp(self) -> datetime: ...


@runtime_checkable
class SandboxRuntime(Protocol):
    """Protocol for sandbox execution environments."""

    async def initialize(self) -> None:
        """Initialize the sandbox runtime (build images, create networks, etc.)."""
        ...

    async def run(
        self,
        code_path: str,
        timeout_seconds: int,
        env_vars: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run code in sandbox and return results."""
        ...

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        ...


@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    success: bool
    tokens_per_second: float
    total_tokens: int
    wall_time_seconds: float
    exit_code: int
    stdout: str
    stderr: str
    error: str | None = None
    # Verification fields
    final_loss: float | None = None
    final_logits: Any = None
    final_logits_path: str | None = None


@runtime_checkable
class CodeValidator(Protocol):
    """Protocol for code validation."""

    def validate(self, code: str) -> ValidationResult:
        """Validate code before execution."""
        ...


@dataclass
class ValidationResult:
    """Result from code validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for database operations."""

    async def save_submission(self, submission: SubmissionModel) -> None: ...

    async def get_submission(self, submission_id: str) -> SubmissionModel | None: ...

    async def update_submission_status(
        self, submission_id: str, status: SubmissionStatus
    ) -> None: ...

    async def save_evaluation(self, evaluation: EvaluationModel) -> None: ...

    async def get_evaluations(self, submission_id: str) -> list[EvaluationModel]: ...

    async def get_leaderboard(self, limit: int = 100) -> list[SubmissionModel]: ...

    async def get_top_submission(self) -> SubmissionModel | None: ...

    async def get_pending_submissions(self) -> list[SubmissionModel]: ...
