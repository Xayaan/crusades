"""Domain models for evaluation results and verification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MFUResult:
    """Result from MFU (Model FLOPs Utilization) calculation."""

    mfu_percent: float = 0.0
    total_system_flops: float = 0.0
    total_system_peak: float = 0.0
    tokens_per_second: float = 0.0
    wall_time_seconds: float = 0.0
    total_unique_tokens: int = 0
    model_params: int = 0
    num_gpus: int = 1
    gpu_peak_tflops: float = 312.0


@dataclass
class VerificationResult:
    """Result from output verification checks."""

    passed: bool = False
    error: str | None = None
    error_code: str | None = None
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[dict[str, str]] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)
