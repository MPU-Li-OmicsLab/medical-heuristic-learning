from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.3


@dataclass(frozen=True)
class RunConfig:
    data_csv_path: Path
    label_col: str
    output_dir: Path
    iterations: int = 10
    metric_priority: tuple[str, ...] = ("F1", "ACC")
    max_error_samples: int = 200
    degradation_threshold: int = 10
    univariate_top_k: int = 30
    knowledge_top_k: int = 20
    test_size: float = 0.2
    random_seed: int = 42
    llm_enabled: bool = True

