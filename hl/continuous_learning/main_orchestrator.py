from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from hl.agent.client import LLMClient
from hl.config import LLMConfig, RunConfig
from hl.continuous_learning.config import ContinuousLearningConfig, ContinuousLearningResult
from hl.continuous_learning.knowledge_probe_step import run_knowledge_probe_task
from hl.continuous_learning.univariate_probe_step import run_univariate_probe_task
from hl.continuous_learning.v0_generation_step import generate_v0_task
from hl.metrics import generate_metric_description
from hl.orchestrator.iteration_step import IterationRecord, run_iterations_task
from hl.utils.io import ensure_dir, write_json, write_text
from hl.utils.progress import log_progress


def _pick_best_record(records: list[IterationRecord], metric_priority: tuple[str, ...]) -> IterationRecord:
    def key_fn(record: IterationRecord) -> tuple:
        return tuple(float(record.metrics.get(metric, float("-inf"))) for metric in metric_priority)

    return max(records, key=key_fn)


def _export_final_model(out_dir: Path, heuristic_path: Path, final_version: str) -> Path:
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
    final_model_path = out_dir / "final_heuristic_model.py"
    write_text(final_model_path, exported)
    return final_model_path


def _resolve_output_dir(cfg: ContinuousLearningConfig) -> Path:
    if cfg.output_dir is not None:
        return cfg.output_dir
    base = Path.cwd() / "out"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"{stamp}_continuous_learning"


def _build_run_config(cfg: ContinuousLearningConfig, out_dir: Path) -> RunConfig:
    return RunConfig(
        output_dir=out_dir,
        iterations=cfg.iterations,
        metric_priority=cfg.metric_priority,
        run_univariate_probe=cfg.run_univariate_probe,
        run_knowledge_probe=cfg.run_knowledge_probe,
        run_v0_generation=cfg.run_v0_generation,
        run_iterations=cfg.run_iterations,
        max_error_samples=cfg.max_error_samples,
        max_error_details=cfg.max_error_details,
        degradation_max_examples=cfg.degradation_max_examples,
        max_llm_attempts=cfg.max_llm_attempts,
        task_description=cfg.task_description,
        univariate_top_k=cfg.univariate_top_k,
        random_seed=cfg.random_seed,
        llm_enabled=cfg.llm_enabled,
    )


def run_continuous_learning(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    llm_cfg: LLMConfig,
    continuous_cfg: ContinuousLearningConfig,
) -> ContinuousLearningResult:
    log_progress("HL-CL", "Starting continuous learning run.")
    out_dir = _resolve_output_dir(continuous_cfg)
    ensure_dir(out_dir)
    log_progress("HL-CL", f"Using output directory: {out_dir}")
    run_cfg = _build_run_config(continuous_cfg, out_dir)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"label_col={label_col} must exist in both train_df and test_df.")

    train_cols = [col for col in train_df.columns if col != label_col]
    test_cols = [col for col in test_df.columns if col != label_col]
    if set(train_cols) != set(test_cols):
        raise ValueError("train_df and test_df must have the same set of feature columns.")

    heuristic_path = out_dir / "heuristic_system.py"
    evolution_results_path = out_dir / "evolution_results.txt"
    iteration_log_path = out_dir / "iteration_log.json"
    final_comparison_path = out_dir / "final_comparison.txt"
    knowledge_path = out_dir / "probe_knowledge.md"
    univariate_path = out_dir / "probe_univariate_results.csv"

    write_json(
        out_dir / "continuous_learning_context.json",
        {
            "task_description": continuous_cfg.task_description,
            "output_dir": str(out_dir),
            "label_col": label_col,
            "metric_priority": list(continuous_cfg.metric_priority),
            "iterations": int(continuous_cfg.iterations),
            "random_seed": int(continuous_cfg.random_seed),
            "drift": {
                "dropped_cols": list(continuous_cfg.drift.dropped_cols),
                "added_cols": list(continuous_cfg.drift.added_cols),
                "renamed_cols": [[old_name, new_name] for old_name, new_name in continuous_cfg.drift.renamed_cols],
                "change_note": continuous_cfg.drift.change_note,
                "prev_hl_out_dir": (
                    str(continuous_cfg.drift.prev_hl_out_dir) if continuous_cfg.drift.prev_hl_out_dir is not None else ""
                ),
            },
        },
    )

    metric_desc = generate_metric_description(run_cfg.metric_priority)
    client: LLMClient | None = None
    if continuous_cfg.llm_enabled:
        log_progress("HL-CL", f"Initializing LLM client with model={llm_cfg.model_name}.")
        client = LLMClient(
            base_url=llm_cfg.base_url,
            api_key_env=llm_cfg.api_key_env,
            model_name=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            api_key=llm_cfg.api_key,
            extra_body=llm_cfg.extra_body,
        )
    else:
        log_progress("HL-CL", "LLM is disabled; continuous run will reuse existing artifacts where possible.")

    log_progress("HL-CL", "Step 1/4: Running univariate probe under drift.")
    _top_features, report_features, univariate_summary = run_univariate_probe_task(
        train_df=train_df,
        label_col=label_col,
        run_cfg=run_cfg,
        univariate_path=univariate_path,
        feature_cols=train_cols,
        drift=continuous_cfg.drift,
    )
    log_progress("HL-CL", "Step 1/4 completed.")
    log_progress("HL-CL", "Step 2/4: Running knowledge probe under drift.")
    knowledge_table = run_knowledge_probe_task(
        client=client,
        feature_cols=train_cols,
        label_col=label_col,
        run_cfg=run_cfg,
        knowledge_path=knowledge_path,
        drift=continuous_cfg.drift,
    )
    log_progress("HL-CL", "Step 2/4 completed.")
    log_progress("HL-CL", "Step 3/4: Generating v0 heuristic under drift.")
    generate_v0_task(
        client=client,
        run_cfg=run_cfg,
        drift=continuous_cfg.drift,
        heuristic_path=heuristic_path,
        univariate_summary=univariate_summary,
        knowledge_table=knowledge_table,
        metric_desc=metric_desc,
    )
    log_progress("HL-CL", "Step 3/4 completed.")

    log_progress("HL-CL", "Step 4/4: Running iterative optimization.")
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
    log_progress("HL-CL", "Step 4/4 completed.")
    write_json(iteration_log_path, iteration_log)

    if not records:
        raise RuntimeError("No version records were generated.")

    best = _pick_best_record(records, run_cfg.metric_priority)
    last = records[-1]
    v0 = records[0]
    final_model_path = _export_final_model(out_dir, heuristic_path, best.version)
    log_progress("HL-CL", f"Exported final model with version={best.version}.")

    comparison = (
        f"METRIC_PRIORITY={run_cfg.metric_priority}\n"
        f"V0={v0.metrics}\n"
        f"FINAL({best.version})={best.metrics}\n"
        f"LAST({last.version})={last.metrics}\n"
    )
    write_text(final_comparison_path, comparison)
    log_progress("HL-CL", "Continuous learning run finished.")

    return ContinuousLearningResult(
        out_dir=out_dir,
        heuristic_path=heuristic_path,
        final_model_path=final_model_path,
    )
