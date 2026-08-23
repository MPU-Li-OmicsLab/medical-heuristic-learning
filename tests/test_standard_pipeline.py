from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import hl.orchestrator.main_orchestrator as standard_main
from hl.config import LLMConfig, RunConfig
from hl.model import load_model
from hl.orchestrator.iteration_step import IterationRecord
from hl.result import RunResult
from tests.support import (
    ScriptedLLM,
    install_fake_client,
    make_knowledge_table,
    make_rule_proposal,
    prompt_text,
)


def test_standard_pipeline_runs_all_stages_with_synthetic_data_and_fake_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binary_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df = binary_frames
    fake = ScriptedLLM(
        text_responses=[make_knowledge_table(["risk_score", "age", "ward", "binary_marker"])],
        json_responses=[
            "{invalid v0 json",
            make_rule_proposal("v0", threshold=0.75),
            make_rule_proposal("wrong_version", threshold=0.0),
            make_rule_proposal("v1", threshold=0.0),
        ],
    )
    captured = install_fake_client(monkeypatch, standard_main, fake)
    out_dir = tmp_path / "standard_run"

    result = standard_main.run_heuristic_learning(
        train_df=train_df,
        val_df=val_df,
        label_col="target",
        run_cfg=RunConfig(
            output_dir=out_dir,
            iterations=1,
            max_llm_attempts=2,
            task_description="Synthetic risk classification",
            random_seed=17,
        ),
        llm_cfg=LLMConfig(
            base_url="https://fake.invalid/v1",
            api_key="fake-key",
            model_name="fake-model",
            thinking_mode=True,
            thinking_strength="low",
        ),
    )

    assert isinstance(result, RunResult)
    assert result.out_dir == out_dir
    assert result.heuristic_path == out_dir / "heuristic_system.py"
    assert result.final_model_path == out_dir / "final_heuristic_model.py"
    for artifact in (
        "probe_univariate_results.csv",
        "probe_knowledge.md",
        "heuristic_system.py",
        "evolution_results.txt",
        "iteration_log.json",
        "final_comparison.txt",
        "final_heuristic_model.py",
    ):
        assert (out_dir / artifact).is_file(), artifact

    final_source = result.final_model_path.read_text(encoding="utf-8")
    assert 'FINAL_VERSION = "v1"' in final_source
    predict = load_model(result.final_model_path)
    predictions = [
        predict({column: row[column] for column in val_df.columns if column != "target"})
        for _, row in val_df.iterrows()
    ]
    assert predictions == val_df["target"].tolist()

    iteration_log = json.loads((out_dir / "iteration_log.json").read_text(encoding="utf-8"))
    assert iteration_log == [
        {
            "version": "v1",
            "accepted": True,
            "attempt_logs": [
                {
                    "attempt": 1,
                    "status": "version_mismatch",
                    "expected": "v1",
                    "got": "wrong_version",
                }
            ],
            "last_accepted_version": "v1",
            "last_regressed_indices": [],
        }
    ]

    assert captured["model_name"] == "fake-model"
    assert captured["thinking_strength"] == "low"
    assert len(fake.text_calls) == 1
    assert len(fake.json_calls) == 4
    assert "Synthetic risk classification" in prompt_text(fake.text_calls)
    assert "predict_v0" in prompt_text(fake.json_calls, 2)
    comparison = (out_dir / "final_comparison.txt").read_text(encoding="utf-8")
    assert "V0=" in comparison
    assert "FINAL(v1)=" in comparison
    assert "LAST(v1)=" in comparison
    fake.assert_exhausted()


def test_standard_pipeline_can_run_fully_offline_from_cached_heuristic(
    tmp_path: Path,
    binary_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df = binary_frames
    out_dir = tmp_path / "offline_run"
    out_dir.mkdir()
    heuristic_path = out_dir / "heuristic_system.py"
    heuristic_path.write_text(
        "CURRENT_VERSION = 'v0'\n\n"
        "def predict_v0(features: dict) -> int:\n"
        "    return int(float(features.get('risk_score', 0.0)) >= 0.0)\n\n"
        "ERROR_ANALYSIS_predict_v0 = 'cached synthetic rule'\n",
        encoding="utf-8",
    )

    result = standard_main.run_heuristic_learning(
        train_df=train_df,
        val_df=val_df,
        label_col="target",
        run_cfg=RunConfig(
            output_dir=out_dir,
            llm_enabled=False,
            run_univariate_probe=False,
            run_knowledge_probe=False,
            run_v0_generation=False,
            run_iterations=True,
        ),
        llm_cfg=LLMConfig(),
    )

    assert result.final_model_path.is_file()
    assert load_model(result.final_model_path)({"risk_score": 1.0}) == 1
    assert json.loads((out_dir / "iteration_log.json").read_text(encoding="utf-8")) == []
    assert not (out_dir / "probe_univariate_results.csv").exists()
    assert not (out_dir / "probe_knowledge.md").exists()


def test_best_record_respects_metric_priority_lexicographically() -> None:
    records = [
        IterationRecord("v0", "", {"F1": 0.7, "ACC": 0.9}),
        IterationRecord("v1", "", {"F1": 0.8, "ACC": 0.7}),
        IterationRecord("v2", "", {"F1": 0.8, "ACC": 0.8}),
    ]
    assert standard_main._pick_best_record(records, ("F1", "ACC")).version == "v2"
    assert standard_main._pick_best_record(records, ("ACC", "F1")).version == "v0"


@pytest.mark.parametrize("problem", ["missing_label", "mismatched_features"])
def test_standard_pipeline_validates_input_schema_before_llm_use(
    tmp_path: Path,
    binary_frames: tuple[pd.DataFrame, pd.DataFrame],
    problem: str,
) -> None:
    train_df, val_df = (frame.copy() for frame in binary_frames)
    if problem == "missing_label":
        val_df = val_df.drop(columns=["target"])
        expected = "must exist in both"
    else:
        val_df = val_df.rename(columns={"age": "renamed_age"})
        expected = "same set of feature columns"

    with pytest.raises(ValueError, match=expected):
        standard_main.run_heuristic_learning(
            train_df=train_df,
            val_df=val_df,
            label_col="target",
            run_cfg=RunConfig(output_dir=tmp_path / problem, llm_enabled=False),
            llm_cfg=LLMConfig(),
        )
