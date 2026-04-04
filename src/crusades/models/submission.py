"""Domain models for submission scoring and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubmissionScore:
    """Domain model for a submission's score calculation."""

    mfu_scores: list[float] = field(default_factory=list)
    median_mfu: float = 0.0
    success_rate: float = 0.0
    total_runs: int = 0
    successful_runs: int = 0
