from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    model_name: str = "deepseek-v4-pro"
    temperature: float = 0.3
    extra_body: dict | None = None


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path | None = None
    iterations: int = 10
    metric_priority: tuple[str, ...] = ("F1", "ACC", "Sensitivity","Specificity")
    train_baselines: bool = False
    run_univariate_probe: bool = True
    run_knowledge_probe: bool = True
    run_v0_generation: bool = True
    run_iterations: bool = True
    max_error_samples: int = 100
    max_error_details: int = 40
    degradation_threshold: int = 10
    degradation_rate: float = 0.05
    degradation_max_examples: int = 30
    max_llm_attempts: int = 4
    task_description: str = ""
    enable_auto_patch: bool = False
    max_specificity_drop: float = 1.0
    max_acc_drop: float = 1.0
    univariate_top_k: int = 30
    knowledge_top_k: int = 20
    random_seed: int = 42
    llm_enabled: bool = True
