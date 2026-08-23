from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib.metadata import version
from pathlib import Path

import pytest

import hl
import hl.orchestrator as orchestrator_api
from hl.config import THINKING_STRENGTH_LEVELS, LLMConfig, RunConfig
from hl.continuous_learning import ContinuousLearningConfig, ContinuousLearningResult, DriftConfig
from hl.result import RunResult


@pytest.mark.parametrize(
    "export_name",
    [name for name in hl.__all__ if name != "__version__"],
)
def test_every_declared_root_export_exists(export_name: str) -> None:
    assert hasattr(hl, export_name)


def test_package_version_matches_installed_distribution() -> None:
    assert "__version__" in hl.__all__
    assert hl.__version__ == version("medical-heuristic-learning")


def test_run_result_has_one_clear_public_owner() -> None:
    assert not hasattr(hl, "RunResult")
    assert not hasattr(orchestrator_api, "RunResult")
    assert issubclass(ContinuousLearningResult, RunResult)


def test_configuration_objects_are_frozen_and_keep_safe_defaults(tmp_path: Path) -> None:
    llm_cfg = LLMConfig()
    run_cfg = RunConfig(output_dir=tmp_path)
    continuous_cfg = ContinuousLearningConfig(drift=DriftConfig(added_cols=("marker",)))

    assert llm_cfg.api_key is None
    assert run_cfg.output_dir == tmp_path
    assert continuous_cfg.drift.added_cols == ("marker",)

    with pytest.raises(FrozenInstanceError):
        run_cfg.iterations = 3  # type: ignore[misc]


def test_critical_configuration_defaults() -> None:
    llm_cfg = LLMConfig()
    run_cfg = RunConfig()
    continuous_cfg = ContinuousLearningConfig()

    assert llm_cfg.model_name == "deepseek-v4-pro"
    assert llm_cfg.thinking_mode is None
    assert llm_cfg.thinking_strength is None
    assert THINKING_STRENGTH_LEVELS == ("low", "medium", "high", "xhigh", "max")
    assert run_cfg.iterations == 10
    assert run_cfg.metric_priority == ("F1", "ACC", "Sensitivity", "Specificity")
    assert run_cfg.llm_enabled is True
    assert run_cfg.run_univariate_probe is True
    assert run_cfg.run_knowledge_probe is True
    assert run_cfg.run_v0_generation is True
    assert run_cfg.run_iterations is True
    assert continuous_cfg.drift == DriftConfig()
