from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hl.config import RunConfig
from hl.orchestrator.iteration_step import run_iterations_task
from hl.orchestrator.knowledge_probe_step import run_knowledge_probe_task
from hl.orchestrator.univariate_probe_step import run_univariate_probe_task
from hl.orchestrator.v0_generation_step import generate_v0_task
from tests.support import ScriptedLLM, make_knowledge_table, make_rule_proposal


def _proposal(version: str, code: str) -> str:
    return json.dumps(
        {
            "version": version,
            "error_analysis": "synthetic proposal",
            "new_policy_code": code,
        }
    )


def test_v0_generation_retries_static_validation_failure(tmp_path: Path) -> None:
    fake = ScriptedLLM(
        json_responses=[
            _proposal("v0", "def predict_v0(features: dict) -> int:\n    return int(bare_feature > 0)"),
            make_rule_proposal("v0", threshold=0.0),
        ]
    )
    heuristic_path = tmp_path / "heuristic_system.py"

    generate_v0_task(
        client=fake,  # type: ignore[arg-type]
        run_cfg=RunConfig(max_llm_attempts=2),
        heuristic_path=heuristic_path,
        univariate_summary="summary",
        knowledge_table="knowledge",
        metric_desc="F1 first",
    )

    source = heuristic_path.read_text(encoding="utf-8")
    assert "CURRENT_VERSION = 'v0'" in source
    assert "def predict_v0" in source
    assert "ERROR_ANALYSIS_predict_v0" in source
    assert len(fake.json_calls) == 2
    fake.assert_exhausted()


def test_v0_generation_reports_last_failure_without_creating_artifact(tmp_path: Path) -> None:
    fake = ScriptedLLM(json_responses=[make_rule_proposal("v9")])
    heuristic_path = tmp_path / "heuristic_system.py"

    with pytest.raises(RuntimeError, match="version mismatch.*resp_preview"):
        generate_v0_task(
            client=fake,  # type: ignore[arg-type]
            run_cfg=RunConfig(max_llm_attempts=1),
            heuristic_path=heuristic_path,
            univariate_summary="",
            knowledge_table="",
            metric_desc="F1 first",
        )
    assert not heuristic_path.exists()
    fake.assert_exhausted()


def test_v0_generation_reuses_existing_artifact_without_llm_call(tmp_path: Path) -> None:
    heuristic_path = tmp_path / "heuristic_system.py"
    original = "def predict_v0(features: dict) -> int:\n    return 1\n"
    heuristic_path.write_text(original, encoding="utf-8")
    fake = ScriptedLLM()

    generate_v0_task(
        client=fake,  # type: ignore[arg-type]
        run_cfg=RunConfig(),
        heuristic_path=heuristic_path,
        univariate_summary="",
        knowledge_table="",
        metric_desc="",
    )

    assert heuristic_path.read_text(encoding="utf-8") == original
    assert fake.json_calls == []


def test_v0_generation_disabled_without_artifact_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="run_v0_generation=False"):
        generate_v0_task(
            client=ScriptedLLM(),  # type: ignore[arg-type]
            run_cfg=RunConfig(run_v0_generation=False),
            heuristic_path=tmp_path / "heuristic_system.py",
            univariate_summary="",
            knowledge_table="",
            metric_desc="",
        )


def test_v0_generation_requires_client_when_artifact_is_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="llm_enabled=False"):
        generate_v0_task(
            client=None,
            run_cfg=RunConfig(llm_enabled=False),
            heuristic_path=tmp_path / "heuristic_system.py",
            univariate_summary="",
            knowledge_table="",
            metric_desc="",
        )


def test_iteration_stage_records_each_rejection_reason(
    tmp_path: Path,
    binary_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df = binary_frames
    heuristic_path = tmp_path / "heuristic_system.py"
    heuristic_path.write_text(
        "CURRENT_VERSION = 'v0'\n\n"
        "def predict_v0(features: dict) -> int:\n"
        "    return int(float(features.get('risk_score', 0.0)) >= 0.0)\n",
        encoding="utf-8",
    )
    fake = ScriptedLLM(
        json_responses=[
            "not-json",
            make_rule_proposal("v8"),
            _proposal("v1", "def predict_v1(:\n    return 0"),
            _proposal("v1", "def predict_v1(features: dict) -> int:\n    return int(bare_name > 0)"),
            _proposal("v1", "def predict_other(features: dict) -> int:\n    return 0"),
        ]
    )

    records, iteration_log = run_iterations_task(
        client=fake,  # type: ignore[arg-type]
        train_df=train_df,
        val_df=val_df,
        label_col="target",
        run_cfg=RunConfig(iterations=2, max_llm_attempts=5),
        heuristic_path=heuristic_path,
        evolution_results_path=tmp_path / "evolution_results.txt",
        metric_desc="F1 first",
        report_features=["risk_score"],
    )

    assert [record.version for record in records] == ["v0"]
    assert len(iteration_log) == 1
    assert iteration_log[0]["accepted"] is False
    assert [entry["status"] for entry in iteration_log[0]["attempt_logs"]] == [
        "json_parse_failed",
        "version_mismatch",
        "syntax_invalid",
        "undefined_name",
        "function_name_mismatch",
    ]
    assert "predict_other" not in heuristic_path.read_text(encoding="utf-8")
    fake.assert_exhausted()


def test_univariate_step_can_compute_then_reuse_cached_csv(
    tmp_path: Path,
    binary_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _ = binary_frames
    feature_cols = [column for column in train_df.columns if column != "target"]
    path = tmp_path / "probe.csv"
    computed = run_univariate_probe_task(
        train_df=train_df,
        label_col="target",
        run_cfg=RunConfig(run_univariate_probe=True, univariate_top_k=2),
        univariate_path=path,
        feature_cols=feature_cols,
    )
    reused = run_univariate_probe_task(
        train_df=train_df,
        label_col="target",
        run_cfg=RunConfig(run_univariate_probe=False, univariate_top_k=2),
        univariate_path=path,
        feature_cols=feature_cols,
    )

    assert path.is_file()
    assert reused[0] == computed[0]
    assert reused[1] == computed[1]
    assert reused[2]


def test_knowledge_step_can_query_then_reuse_cached_markdown(tmp_path: Path) -> None:
    table = make_knowledge_table(["risk_score"])
    fake = ScriptedLLM(text_responses=[table])
    path = tmp_path / "knowledge.md"

    queried = run_knowledge_probe_task(
        client=fake,  # type: ignore[arg-type]
        feature_cols=["risk_score"],
        label_col="target",
        run_cfg=RunConfig(run_knowledge_probe=True),
        knowledge_path=path,
    )
    reused = run_knowledge_probe_task(
        client=None,
        feature_cols=["risk_score"],
        label_col="target",
        run_cfg=RunConfig(run_knowledge_probe=False),
        knowledge_path=path,
    )

    assert queried == table
    assert reused == table
    fake.assert_exhausted()
