from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl.config import RunConfig
from hl.continuous_learning.config import ContinuousLearningConfig, DriftConfig
from hl.continuous_learning.knowledge_probe_step import (
    _filter_previous_rows,
    _parse_markdown_table,
    _render_markdown_table,
    run_knowledge_probe_task,
)
from hl.continuous_learning.main_orchestrator import _build_run_config
from hl.continuous_learning.univariate_probe_step import (
    _filter_previous_probe,
    run_univariate_probe_task,
)
from hl.continuous_learning.v0_generation_step import _read_blueprint_final
from tests.support import ScriptedLLM, make_knowledge_table, prompt_text


def test_markdown_table_parser_pads_short_rows_and_ignores_non_table_text() -> None:
    markdown = """
Introductory prose.
| Feature | Signal | Rationale |
| --- | --- | --- |
| risk_score | strong |
| age | weak | synthetic |
Trailing prose.
"""

    header, rows = _parse_markdown_table(markdown)

    assert header == ["Feature", "Signal", "Rationale"]
    assert rows == [
        ["risk_score", "strong", ""],
        ["age", "weak", "synthetic"],
    ]
    assert _parse_markdown_table("plain text only") == ([], [])


def test_markdown_table_render_round_trip() -> None:
    header = ["Feature", "Signal"]
    rows = [["risk_score", "strong"], ["age"]]

    rendered = _render_markdown_table(header, rows)

    assert _parse_markdown_table(rendered) == (
        header,
        [["risk_score", "strong"], ["age", ""]],
    )
    assert _render_markdown_table([], rows) == ""


def test_filter_previous_knowledge_rows_applies_drop_and_rename() -> None:
    header = ["Feature", "Signal"]
    rows = [["old_lab", "drop"], ["age", "keep"], ["risk_score", "keep"]]
    drift = DriftConfig(
        dropped_cols=("old_lab",),
        renamed_cols=(("age", "patient_age"),),
    )

    assert _filter_previous_rows(header, rows, drift) == [
        ["patient_age", "keep"],
        ["risk_score", "keep"],
    ]


def test_filter_previous_univariate_probe_applies_drop_and_rename() -> None:
    previous = pd.DataFrame(
        {
            "feature": ["old_lab", "age", "risk_score"],
            "p_value": [0.1, 0.2, 0.3],
            "missing_rate": [0.0, 0.0, 0.0],
        }
    )
    drift = DriftConfig(
        dropped_cols=("old_lab",),
        renamed_cols=(("age", "patient_age"),),
    )

    filtered = _filter_previous_probe(previous, drift)

    assert filtered["feature"].tolist() == ["patient_age", "risk_score"]
    assert previous["feature"].tolist() == ["old_lab", "age", "risk_score"]


def test_blueprint_reader_handles_missing_and_truncates_large_model(tmp_path: Path) -> None:
    assert _read_blueprint_final(None) == ""
    assert _read_blueprint_final(tmp_path / "missing") == ""

    previous = tmp_path / "previous"
    previous.mkdir()
    source = "0123456789" * 10
    (previous / "final_heuristic_model.py").write_text(source, encoding="utf-8")

    truncated = _read_blueprint_final(previous, max_chars=20)

    assert truncated.startswith(source[:10])
    assert truncated.endswith(source[-10:])
    assert "[...TRUNCATED...]" in truncated


def test_continuous_config_maps_to_standard_run_config(tmp_path: Path) -> None:
    cfg = ContinuousLearningConfig(
        output_dir=tmp_path,
        iterations=3,
        metric_priority=("ACC", "F1"),
        run_univariate_probe=False,
        run_knowledge_probe=False,
        run_v0_generation=False,
        run_iterations=False,
        max_error_samples=7,
        max_error_details=4,
        degradation_max_examples=2,
        max_llm_attempts=6,
        task_description="Synthetic drift task",
        univariate_top_k=5,
        random_seed=99,
        llm_enabled=False,
    )

    run_cfg = _build_run_config(cfg, tmp_path)

    assert isinstance(run_cfg, RunConfig)
    assert run_cfg.output_dir == tmp_path
    assert run_cfg.iterations == 3
    assert run_cfg.metric_priority == ("ACC", "F1")
    assert run_cfg.run_univariate_probe is False
    assert run_cfg.run_knowledge_probe is False
    assert run_cfg.run_v0_generation is False
    assert run_cfg.run_iterations is False
    assert run_cfg.max_error_samples == 7
    assert run_cfg.max_error_details == 4
    assert run_cfg.degradation_max_examples == 2
    assert run_cfg.max_llm_attempts == 6
    assert run_cfg.task_description == "Synthetic drift task"
    assert run_cfg.univariate_top_k == 5
    assert run_cfg.random_seed == 99
    assert run_cfg.llm_enabled is False


def test_continuous_univariate_step_merges_previous_and_added_features(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    previous.mkdir()
    current.mkdir()
    pd.DataFrame(
        {
            "feature": ["old_lab", "age", "risk_score"],
            "p_value": [0.3, 0.2, 0.1],
            "missing_rate": [0.0, 0.0, 0.0],
        }
    ).to_csv(previous / "probe_univariate_results.csv", index=False)
    train_df = pd.DataFrame(
        {
            "patient_age": [30, 40, 50, 60],
            "risk_score": [-1.0, -0.5, 0.5, 1.0],
            "new_marker": [0, 0, 1, 1],
            "target": [0, 0, 1, 1],
        }
    )
    drift = DriftConfig(
        dropped_cols=("old_lab",),
        added_cols=("new_marker",),
        renamed_cols=(("age", "patient_age"),),
        prev_hl_out_dir=previous,
    )

    top_features, report_features, summary = run_univariate_probe_task(
        train_df=train_df,
        label_col="target",
        run_cfg=RunConfig(univariate_top_k=10),
        univariate_path=current / "probe_univariate_results.csv",
        feature_cols=["patient_age", "risk_score", "new_marker"],
        drift=drift,
    )

    saved = pd.read_csv(current / "probe_univariate_results.csv")
    assert set(saved["feature"]) == {"patient_age", "risk_score", "new_marker"}
    assert "old_lab" not in summary
    assert top_features == report_features
    assert (current / "probe_univariate_results_prev.csv").is_file()


def test_continuous_knowledge_step_filters_previous_and_queries_only_added_features(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    previous.mkdir()
    current.mkdir()
    (previous / "probe_knowledge.md").write_text(
        make_knowledge_table(["old_lab", "age", "risk_score"]),
        encoding="utf-8",
    )
    fake = ScriptedLLM(text_responses=[make_knowledge_table(["new_marker"])])
    drift = DriftConfig(
        dropped_cols=("old_lab",),
        added_cols=("new_marker",),
        renamed_cols=(("age", "patient_age"),),
        prev_hl_out_dir=previous,
    )

    merged = run_knowledge_probe_task(
        client=fake,  # type: ignore[arg-type]
        feature_cols=["patient_age", "risk_score", "new_marker"],
        label_col="target",
        run_cfg=RunConfig(task_description="Synthetic drift"),
        knowledge_path=current / "probe_knowledge.md",
        drift=drift,
    )

    assert "old_lab" not in merged
    assert "patient_age" in merged
    assert "risk_score" in merged
    assert "new_marker" in merged
    sent_prompt = prompt_text(fake.text_calls)
    assert '"new_marker"' in sent_prompt
    assert '"risk_score"' not in sent_prompt
    assert (current / "probe_knowledge_prev.md").is_file()
    fake.assert_exhausted()
