from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as distribution_version

from hl.config import LLMConfig, RunConfig
from hl.continuous_learning import (
    ContinuousLearningConfig,
    ContinuousLearningResult,
    DriftConfig,
    run_continuous_learning,
)
from hl.model import BatchPredictFunction, PredictFunction, load_batch_model, load_model
from hl.orchestrator import run_heuristic_learning


try:
    __version__ = distribution_version("medical-heuristic-learning")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = [
    "ContinuousLearningConfig",
    "ContinuousLearningResult",
    "DriftConfig",
    "BatchPredictFunction",
    "LLMConfig",
    "PredictFunction",
    "RunConfig",
    "__version__",
    "load_batch_model",
    "load_model",
    "run_continuous_learning",
    "run_heuristic_learning",
]
