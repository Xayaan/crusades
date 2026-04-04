"""Affinetes integration for Templar Crusades.

URL-Based Architecture:
- Validator owns the evaluation image (templar-eval)
- Miner's code is downloaded from their committed URL
- Supports docker (local) and basilica (remote) modes
"""

from .executors import BasilicaExecutor, DockerExecutor, EvalConfig, ExecutorProtocol
from .runner import AffinetesRunner, BasilicaDeploymentContext, EvaluationResult

__all__ = [
    "AffinetesRunner",
    "BasilicaDeploymentContext",
    "EvaluationResult",
    "ExecutorProtocol",
    "EvalConfig",
    "DockerExecutor",
    "BasilicaExecutor",
]
