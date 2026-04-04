"""Base executor protocol and shared configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from crusades.affinetes.runner import EvaluationResult


@dataclass
class EvalConfig:
    """Shared evaluation configuration used by all executor strategies."""

    timeout: int = 600
    model_url: str | None = None
    data_url: str | None = None
    max_loss_difference: float = 0.3
    min_params_changed_ratio: float = 0.8
    weight_relative_error_max: float = 0.008
    timer_divergence_threshold: float = 0.005
    gpu_peak_tflops: float = 312.0
    max_plausible_mfu: float = 75.0
    min_mfu: float = 50.0


class ExecutorProtocol(Protocol):
    """Protocol that all executor strategies must implement."""

    async def evaluate(
        self,
        code: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult: ...
