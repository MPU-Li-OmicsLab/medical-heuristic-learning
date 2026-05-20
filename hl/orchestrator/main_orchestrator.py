from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from hl.agent.client import LLMClient
from hl.config import LLMConfig, RunConfig
from hl.metrics import generate_metric_description
from hl.orchestrator.iteration_step import IterationRecord, run_iterations_task
from hl.orchestrator.knowledge_probe_step import run_knowledge_probe_task
from hl.orchestrator.univariate_probe_step import run_univariate_probe_task
from hl.orchestrator.v0_generation_step import generate_v0_task
from hl.utils.io import ensure_dir, write_json, write_text


def _pick_best_record(records: list[IterationRecord], metric_priority: tuple[str, ...]) -> IterationRecord:
    def key_fn(r: IterationRecord) -> tuple:
        return tuple(float(r.metrics.get(m, float("-inf"))) for m in metric_priority)

    return max(records, key=key_fn)


def _export_final_model(out_dir: Path, heuristic_path: Path, final_version: str) -> None:
    code_all = heuristic_path.read_text(encoding="utf-8")
    exported = (
        f"FINAL_VERSION = {json.dumps(final_version)}\n\n"
        + code_all
        + "\n\n"
        + "def predict(features: dict) -> int:\n"
        + f"    fn = globals().get('predict_{final_version}')\n"
        + "    if fn is None:\n"
        + "        raise RuntimeError('final predictor not found')\n"
        + "    return int(fn(features))\n"
    )
    write_text(out_dir / "final_heuristic_model.py", exported)


def run_heuristic_learning(
    train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str, run_cfg: RunConfig, llm_cfg: LLMConfig
) -> None:
    if run_cfg.output_dir is None:
        base = Path.cwd() / "out"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base / stamp
    else:
        out_dir = run_cfg.output_dir

    ensure_dir(out_dir)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"label_col={label_col} must exist in both train_df and test_df.")

    train_cols = [c for c in train_df.columns if c != label_col]
    test_cols = [c for c in test_df.columns if c != label_col]
    if set(train_cols) != set(test_cols):
        raise ValueError("train_df and test_df must have the same set of feature columns.")
    feature_cols = train_cols

    heuristic_path = out_dir / "heuristic_system.py"
    evolution_results_path = out_dir / "evolution_results.txt"
    iteration_log_path = out_dir / "iteration_log.json"
    final_comparison_path = out_dir / "final_comparison.txt"
    knowledge_path = out_dir / "probe_knowledge.md"
    univariate_path = out_dir / "probe_univariate_results.csv"

    metric_desc = generate_metric_description(run_cfg.metric_priority)

    client: LLMClient | None = None
    if run_cfg.llm_enabled:
        client = LLMClient(
            base_url=llm_cfg.base_url,
            api_key_env=llm_cfg.api_key_env,
            model_name=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            api_key=llm_cfg.api_key,
        )

    _top_features, report_features, univariate_summary = run_univariate_probe_task(
        train_df=train_df,
        label_col=label_col,
        run_cfg=run_cfg,
        univariate_path=univariate_path,
        feature_cols=feature_cols,
    )
    knowledge_table = run_knowledge_probe_task(
        client=client,
        feature_cols=feature_cols,
        label_col=label_col,
        run_cfg=run_cfg,
        knowledge_path=knowledge_path,
    )
    generate_v0_task(
        client=client,
        run_cfg=run_cfg,
        heuristic_path=heuristic_path,
        univariate_summary=univariate_summary,
        knowledge_table=knowledge_table,
        metric_desc=metric_desc,
    )
    records, iteration_log = run_iterations_task(
        client=client,
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        run_cfg=run_cfg,
        heuristic_path=heuristic_path,
        evolution_results_path=evolution_results_path,
        metric_desc=metric_desc,
        report_features=report_features,
    )

    write_json(iteration_log_path, iteration_log)

    if not records:
        raise RuntimeError("No version records were generated.")

    best = _pick_best_record(records, run_cfg.metric_priority)
    last = records[-1]
    v0 = records[0]

    _export_final_model(out_dir, heuristic_path, best.version)

    comparison = (
        f"METRIC_PRIORITY={run_cfg.metric_priority}\n"
        f"V0={v0.metrics}\n"
        f"FINAL({best.version})={best.metrics}\n"
        f"LAST({last.version})={last.metrics}\n"
    )
    write_text(final_comparison_path, comparison)
