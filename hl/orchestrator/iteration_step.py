from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from hl.agent.client import ChatMessage, LLMClient
from hl.agent.prompts import get_iteration_prompt
from hl.config import RunConfig
from hl.evolution.degradation import collect_degradation_examples, detect_degradation, format_degradation_warning
from hl.evolution.error_analysis import collect_errors, format_error_report
from hl.evolution.rule_utils import ParsedProposal, extract_function_name, strip_code_fences, validate_python_syntax
from hl.metrics import compute_metrics
from hl.utils.io import append_text
from hl.utils.progress import log_progress


@dataclass(frozen=True)
class IterationRecord:
    version: str
    error_analysis: str
    metrics: dict[str, float | int]


def _parse_proposal(text: str) -> ParsedProposal:
    raw = strip_code_fences(text)
    data = json.loads(raw)
    return ParsedProposal(
        version=str(data.get("version", "")),
        error_analysis=str(data.get("error_analysis", "")),
        new_policy_code=str(data.get("new_policy_code", "")),
    )


def _predict_with_function(fn: Callable[[dict], int], df: pd.DataFrame, label_col: str) -> np.ndarray:
    feature_cols = [c for c in df.columns if c != label_col]
    preds: list[int] = []
    for _, row in df.iterrows():
        feats = {c: row[c] for c in feature_cols}
        p = fn(feats)
        preds.append(int(p))
    return np.asarray(preds, dtype=int)


def _load_heuristic_module(path: Path) -> dict:
    ns: dict = {}
    code = path.read_text(encoding="utf-8") if path.exists() else ""
    exec(compile(code, str(path), "exec"), ns, ns)
    return ns


def _append_new_version(path: Path, version_code: str, error_analysis: str) -> None:
    fn_name = extract_function_name(version_code) or ""
    version = fn_name.replace("predict_", "") if fn_name.startswith("predict_") else ""
    block = "\n\n" + (f"CURRENT_VERSION = {json.dumps(version)}\n\n" if version else "") + version_code.strip() + "\n"
    if error_analysis:
        block += f"\nERROR_ANALYSIS_{fn_name} = {json.dumps(error_analysis, ensure_ascii=False)}\n"
    append_text(path, block)


def run_iterations_task(
    *,
    client: LLMClient | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    run_cfg: RunConfig,
    heuristic_path: Path,
    evolution_results_path: Path,
    metric_desc: str,
    report_features: list[str],
) -> tuple[list[IterationRecord], list[dict]]:
    log_progress("HL-ITER", f"Loading heuristic module from {heuristic_path}.")
    ns = _load_heuristic_module(heuristic_path)
    fn_v0 = ns.get("predict_v0")
    if fn_v0 is None:
        raise RuntimeError("predict_v0 not found in heuristic_system.py")

    y_true_train = train_df[label_col].astype(int).to_numpy()
    y_true_test = test_df[label_col].astype(int).to_numpy()

    current_version = "v0"
    current_fn = fn_v0

    records: list[IterationRecord] = []
    iteration_log: list[dict] = []
    trajectory_lines: list[str] = []

    last_regressed_indices: list[int] = []
    last_regressed_examples: list[dict] = []

    y_pred_test_v0 = _predict_with_function(fn_v0, test_df, label_col)
    metrics_v0 = compute_metrics(y_true_test, y_pred_test_v0)
    v0_analysis = str(ns.get("ERROR_ANALYSIS_predict_v0") or "v0")
    records.append(IterationRecord(version="v0", error_analysis=v0_analysis, metrics=metrics_v0))
    append_text(evolution_results_path, f"v0\t{metrics_v0}\n")
    log_progress("HL-ITER", f"Baseline v0 metrics: {metrics_v0}.")

    if not run_cfg.run_iterations:
        log_progress("HL-ITER", "Iteration stage is disabled; returning v0 only.")
        return records, iteration_log
    if client is None:
        log_progress("HL-ITER", "LLM client is unavailable; skipping iterative optimization.")
        return records, iteration_log

    for i in range(1, max(0, run_cfg.iterations) + 1):
        next_version = f"v{i}"
        log_progress("HL-ITER", f"Starting iteration {i}/{max(0, run_cfg.iterations)} for {next_version}.")
        current_code = heuristic_path.read_text(encoding="utf-8")

        y_pred_train = _predict_with_function(current_fn, train_df, label_col)
        errors = collect_errors(
            df=train_df,
            label_col=label_col,
            y_pred=y_pred_train,
            max_error_samples=run_cfg.max_error_samples,
            random_seed=run_cfg.random_seed + i,
            feature_cols=report_features,
        )
        error_report = format_error_report(errors, max_details=run_cfg.max_error_details)

        degradation_warning = format_degradation_warning(last_regressed_indices)
        if last_regressed_examples:
            degradation_warning += "\nRegressed examples (json):\n" + json.dumps(
                last_regressed_examples, ensure_ascii=False, indent=2
            )

        prompt = get_iteration_prompt(
            current_code=current_code,
            error_report=error_report,
            trajectory="\n".join(trajectory_lines) if trajectory_lines else "None",
            degradation_warning=degradation_warning,
            metric_desc=metric_desc,
            task_description=run_cfg.task_description,
            next_version=next_version,
        )

        accepted = False
        attempt_logs: list[dict] = []
        for attempt in range(1, max(1, run_cfg.max_llm_attempts) + 1):
            log_progress(
                "HL-ITER",
                f"Requesting proposal for {next_version} (attempt {attempt}/{max(1, run_cfg.max_llm_attempts)}).",
            )
            resp = client.chat_json([ChatMessage(role="user", content=prompt)])
            try:
                proposal = _parse_proposal(resp)
            except Exception as e:
                attempt_logs.append({"attempt": attempt, "status": "json_parse_failed", "error": str(e)})
                log_progress("HL-ITER", f"{next_version} attempt {attempt} failed: json_parse_failed ({e}).")
                continue

            if proposal.version != next_version:
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "status": "version_mismatch",
                        "expected": next_version,
                        "got": proposal.version,
                    }
                )
                log_progress("HL-ITER", f"{next_version} attempt {attempt} failed: version_mismatch.")
                continue

            new_code = proposal.new_policy_code.strip()
            try:
                validate_python_syntax(new_code)
            except Exception as e:
                attempt_logs.append({"attempt": attempt, "status": "syntax_invalid", "error": str(e)})
                log_progress("HL-ITER", f"{next_version} attempt {attempt} failed: syntax_invalid ({e}).")
                continue
            
            fn_name = extract_function_name(new_code)
            if fn_name != f"predict_{next_version}":
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "status": "function_name_mismatch",
                        "expected": f"predict_{next_version}",
                        "got": fn_name,
                    }
                )
                log_progress("HL-ITER", f"{next_version} attempt {attempt} failed: function_name_mismatch.")
                continue

            _append_new_version(heuristic_path, version_code=new_code, error_analysis=proposal.error_analysis)
            ns = _load_heuristic_module(heuristic_path)
            current_fn = ns.get(f"predict_{next_version}")
            if current_fn is None:
                attempt_logs.append({"attempt": attempt, "status": "load_failed"})
                log_progress("HL-ITER", f"{next_version} attempt {attempt} failed: load_failed.")
                continue

            current_version = next_version
            trajectory_lines.append(f"{next_version}: {proposal.error_analysis}")
            accepted = True

            y_pred_test = _predict_with_function(current_fn, test_df, label_col)
            m = compute_metrics(y_true_test, y_pred_test)
            records.append(IterationRecord(version=current_version, error_analysis=proposal.error_analysis, metrics=m))
            append_text(evolution_results_path, f"{current_version}\t{m}\n")
            log_progress("HL-ITER", f"Accepted {current_version} with metrics: {m}.")

            y_pred_train_new = _predict_with_function(current_fn, train_df, label_col)
            degr = detect_degradation(y_true_train, y_pred_train, y_pred_train_new)
            examples = collect_degradation_examples(
                df=train_df,
                label_col=label_col,
                degraded_indices=degr.degraded_indices,
                y_pred_old=y_pred_train,
                y_pred_new=y_pred_train_new,
                feature_cols=report_features,
                max_samples=run_cfg.degradation_max_examples,
                random_seed=run_cfg.random_seed + i,
            )
            last_regressed_indices = list(degr.degraded_indices)
            last_regressed_examples = list(examples)
            if last_regressed_indices:
                log_progress(
                    "HL-ITER",
                    f"Detected {len(last_regressed_indices)} regressed training examples after accepting {current_version}.",
                )
            break

        iteration_log.append(
            {
                "version": next_version,
                "accepted": accepted,
                "attempt_logs": attempt_logs,
                "last_accepted_version": current_version,
                "last_regressed_indices": last_regressed_indices,
            }
        )
        if not accepted:
            log_progress("HL-ITER", f"Stopping iterations because {next_version} was not accepted.")
            break

    log_progress("HL-ITER", f"Iteration stage finished with {len(records)} recorded versions.")
    return records, iteration_log
