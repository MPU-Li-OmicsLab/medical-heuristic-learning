from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DriftConfig:
    dropped_cols: tuple[str, ...] = ()
    added_cols: tuple[str, ...] = ()
    renamed_cols: tuple[tuple[str, str], ...] = ()
    change_note: str = ""
    prev_hl_out_dir: Path | None = None


@dataclass(frozen=True)
class ContinuousLearningConfig:
    drift: DriftConfig = field(default_factory=DriftConfig)
    output_dir: Path | None = None
    iterations: int = 10
    metric_priority: tuple[str, ...] = ("F1", "ACC")
    run_univariate_probe: bool = True
    run_knowledge_probe: bool = True
    run_v0_generation: bool = True
    run_iterations: bool = True
    max_error_samples: int = 100
    max_error_details: int = 40
    degradation_max_examples: int = 30
    max_llm_attempts: int = 4
    task_description: str = ""
    univariate_top_k: int = 30
    random_seed: int = 42
    llm_enabled: bool = True


@dataclass(frozen=True)
class ContinuousLearningResult:
    out_dir: Path
    heuristic_path: Path
    final_model_path: Path
