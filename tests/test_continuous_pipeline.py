from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import hl.continuous_learning.main_orchestrator as continuous_main
import hl.orchestrator.main_orchestrator as standard_main
from hl.config import LLMConfig, RunConfig
from hl.continuous_learning import ContinuousLearningConfig, ContinuousLearningResult, DriftConfig
from hl.model import load_model
from hl.result import RunResult
from tests.support import (
    ScriptedLLM,
    install_fake_client,
    make_knowledge_table,
    make_rule_proposal,
    prompt_text,
)


def _drifted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    drifted = frame.rename(columns={"age": "patient_age"}).drop(columns=["binary_marker"])
    drifted.insert(
        drifted.columns.get_loc("target"),
        "new_marker",
        (drifted["risk_score"] >= 0.0).astype(int),
    )
    return drifted


def test_standard_output_feeds_complete_continuous_learning_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binary_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df = binary_frames
    standard_fake = ScriptedLLM(
        text_responses=[make_knowledge_table(["risk_score", "age", "ward", "binary_marker"])],
        json_responses=[
            make_rule_proposal("v0", threshold=0.75),
            make_rule_proposal("v1", threshold=0.0),
        ],
    )
    install_fake_client(monkeypatch, standard_main, standard_fake)
    previous_out = tmp_path / "previous_standard"
    previous_result = standard_main.run_heuristic_learning(
        train_df=train_df,
        val_df=val_df,
        label_col="target",
        run_cfg=RunConfig(
            output_dir=previous_out,
            iterations=1,
            task_description="Previous synthetic schema",
        ),
        llm_cfg=LLMConfig(api_key="fake-key", model_name="fake-standard"),
    )
    standard_fake.assert_exhausted()

    continuous_fake = ScriptedLLM(
        text_responses=[make_knowledge_table(["new_marker"])],
        json_responses=[
            make_rule_proposal("v0", threshold=0.6),
            make_rule_proposal("v1", threshold=0.0),
        ],
    )
    captured = install_fake_client(monkeypatch, continuous_main, continuous_fake)
    current_out = tmp_path / "continuous"
    drift = DriftConfig(
        dropped_cols=("binary_marker",),
        added_cols=("new_marker",),
        renamed_cols=(("age", "patient_age"),),
        change_note="Synthetic schema version two",
        prev_hl_out_dir=previous_result.out_dir,
    )

    result = continuous_main.run_continuous_learning(
        train_df=_drifted_frame(train_df),
        val_df=_drifted_frame(val_df),
        label_col="target",
        llm_cfg=LLMConfig(
            base_url="https://fake.invalid/v1",
            api_key="fake-key",
            model_name="fake-continuous",
        ),
        continuous_cfg=ContinuousLearningConfig(
            drift=drift,
            output_dir=current_out,
            iterations=1,
            max_llm_attempts=1,
            task_description="Adapt the synthetic rule to schema version two",
        ),
    )

    assert isinstance(result, ContinuousLearningResult)
    assert isinstance(result, RunResult)
    assert result.out_dir == current_out
    for artifact in (
        "continuous_learning_context.json",
        "probe_univariate_results_prev.csv",
        "probe_univariate_results.csv",
        "probe_knowledge_prev.md",
        "probe_knowledge.md",
        "heuristic_system.py",
        "evolution_results.txt",
        "iteration_log.json",
        "final_comparison.txt",
        "final_heuristic_model.py",
    ):
        assert (current_out / artifact).is_file(), artifact

    context = json.loads((current_out / "continuous_learning_context.json").read_text(encoding="utf-8"))
    assert context["label_col"] == "target"
    assert context["drift"] == {
        "dropped_cols": ["binary_marker"],
        "added_cols": ["new_marker"],
        "renamed_cols": [["age", "patient_age"]],
        "change_note": "Synthetic schema version two",
        "prev_hl_out_dir": str(previous_out),
    }

    univariate = pd.read_csv(current_out / "probe_univariate_results.csv")
    assert set(univariate["feature"]) == {"risk_score", "patient_age", "ward", "new_marker"}
    knowledge = (current_out / "probe_knowledge.md").read_text(encoding="utf-8")
    assert "binary_marker" not in knowledge
    assert "patient_age" in knowledge
    assert "new_marker" in knowledge

    v0_prompt = prompt_text(continuous_fake.json_calls, 0)
    assert "Previous Final Model Blueprint" in v0_prompt
    assert 'FINAL_VERSION = "v1"' in v0_prompt
    assert "binary_marker" in v0_prompt
    assert "new_marker" in v0_prompt
    assert "patient_age" in v0_prompt
    assert "Synthetic schema version two" in v0_prompt

    final_source = result.final_model_path.read_text(encoding="utf-8")
    assert 'FINAL_VERSION = "v1"' in final_source
    predict = load_model(result.final_model_path)
    drifted_val = _drifted_frame(val_df)
    predictions = [
        predict({column: row[column] for column in drifted_val.columns if column != "target"})
        for _, row in drifted_val.iterrows()
    ]
    assert predictions == drifted_val["target"].tolist()

    comparison = (current_out / "final_comparison.txt").read_text(encoding="utf-8")
    assert "V0=" in comparison
    assert "FINAL(v1)=" in comparison
    assert "LAST(v1)=" in comparison
    assert captured["model_name"] == "fake-continuous"
    assert len(continuous_fake.text_calls) == 1
    assert len(continuous_fake.json_calls) == 2
    continuous_fake.assert_exhausted()


@pytest.mark.parametrize("problem", ["missing_label", "mismatched_features"])
def test_continuous_pipeline_validates_input_schema_before_llm_use(
    tmp_path: Path,
    binary_frames: tuple[pd.DataFrame, pd.DataFrame],
    problem: str,
) -> None:
    train_df, val_df = (frame.copy() for frame in binary_frames)
    if problem == "missing_label":
        val_df = val_df.drop(columns=["target"])
        expected = "must exist in both"
    else:
        val_df = val_df.rename(columns={"age": "patient_age"})
        expected = "same set of feature columns"

    with pytest.raises(ValueError, match=expected):
        continuous_main.run_continuous_learning(
            train_df=train_df,
            val_df=val_df,
            label_col="target",
            llm_cfg=LLMConfig(),
            continuous_cfg=ContinuousLearningConfig(
                output_dir=tmp_path / problem,
                llm_enabled=False,
            ),
        )
