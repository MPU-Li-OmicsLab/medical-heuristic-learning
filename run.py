from __future__ import annotations

import ast
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from agent.client import ChatMessage, LLMClient
from agent.prompts import get_iteration_prompt, get_rule_generation_prompt
from config import LLMConfig, RunConfig
from evolution.degradation import collect_degradation_examples, detect_degradation, format_degradation_warning
from evolution.error_analysis import collect_errors, format_error_report
from evolution.rule_utils import ParsedProposal, extract_function_name, strip_code_fences, validate_python_syntax
from metrics import compute_metrics, generate_metric_description
from probes.knowledge import run_knowledge_probe
from probes.univariate import run_univariate_probe
from utils.data_loader import load_csv
from utils.io import append_text, ensure_dir, write_json, write_text


@dataclass(frozen=True)
class IterationRecord:
    version: str
    error_analysis: str
    metrics: dict[str, float]


def _predict_with_function(fn: Callable[[dict], int], df: pd.DataFrame, label_col: str) -> np.ndarray:
    feature_cols = [c for c in df.columns if c != label_col]
    preds: list[int] = []
    for _, row in df.iterrows():
        feats = {c: row[c] for c in feature_cols}
        try:
            p = int(fn(feats))
        except Exception:
            p = 0
        preds.append(int(p))
    return np.asarray(preds, dtype=int)




def _extract_function_source(code: str, fn_name: str) -> str | None:
    try:
        module = ast.parse(code)
    except Exception:
        return None
    target: ast.FunctionDef | None = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            target = node
            break
    if target is None or target.lineno is None or target.end_lineno is None:
        return None
    lines = code.splitlines()
    return "\n".join(lines[target.lineno - 1 : target.end_lineno])


def _build_score_function(fn_source: str, score_fn_name: str) -> Callable[[dict], float] | None:
    try:
        mod = ast.parse(fn_source)
    except Exception:
        return None

    fndef: ast.FunctionDef | None = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            fndef = n
            break
    if fndef is None:
        return None

    fndef.name = score_fn_name

    def _replace_return(node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Return):
            return ast.copy_location(ast.Return(value=ast.Name(id="score", ctx=ast.Load())), node)
        return node

    new_body: list[ast.stmt] = []
    for stmt in fndef.body:
        if isinstance(stmt, ast.Return):
            new_body.append(_replace_return(stmt))  # type: ignore[arg-type]
        else:
            new_body.append(stmt)
    fndef.body = new_body

    tmp_mod = ast.Module(body=[fndef], type_ignores=[])
    ast.fix_missing_locations(tmp_mod)
    try:
        code_out = ast.unparse(tmp_mod)
    except Exception:
        return None

    ns: dict = {}
    try:
        exec(compile(code_out, "<score_fn>", "exec"), ns, ns)
        fn = ns.get(score_fn_name)
        if callable(fn):
            return fn
    except Exception:
        return None
    return None


def _extract_threshold_from_source(fn_source: str) -> float | None:
    try:
        mod = ast.parse(fn_source)
    except Exception:
        return None
    fndef: ast.FunctionDef | None = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            fndef = n
            break
    if fndef is None or not fndef.body:
        return None
    last = fndef.body[-1]
    if not isinstance(last, ast.Return) or last.value is None:
        return None
    if isinstance(last.value, ast.IfExp) and isinstance(last.value.test, ast.Compare):
        cmp = last.value.test
        if (
            isinstance(cmp.left, ast.Name)
            and cmp.left.id == "score"
            and len(cmp.ops) == 1
            and isinstance(cmp.ops[0], ast.GtE)
            and len(cmp.comparators) == 1
            and isinstance(cmp.comparators[0], ast.Constant)
            and isinstance(cmp.comparators[0].value, (int, float))
        ):
            return float(cmp.comparators[0].value)
    return None


def _patch_threshold_in_source(fn_source: str, new_threshold: float, new_name: str) -> str | None:
    try:
        mod = ast.parse(fn_source)
    except Exception:
        return None
    fndef: ast.FunctionDef | None = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            fndef = n
            break
    if fndef is None:
        return None
    fndef.name = new_name
    if not fndef.body:
        return None
    last = fndef.body[-1]
    if not isinstance(last, ast.Return) or last.value is None:
        return None
    if isinstance(last.value, ast.IfExp) and isinstance(last.value.test, ast.Compare):
        cmp = last.value.test
        if (
            isinstance(cmp.left, ast.Name)
            and cmp.left.id == "score"
            and len(cmp.ops) == 1
            and isinstance(cmp.ops[0], ast.GtE)
            and len(cmp.comparators) == 1
            and isinstance(cmp.comparators[0], ast.Constant)
            and isinstance(cmp.comparators[0].value, (int, float))
        ):
            cmp.comparators[0] = ast.Constant(value=float(new_threshold))
        else:
            return None
    else:
        return None

    tmp_mod = ast.Module(body=[fndef], type_ignores=[])
    ast.fix_missing_locations(tmp_mod)
    try:
        return ast.unparse(tmp_mod)
    except Exception:
        return None


def _find_best_threshold_for_f1(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    uniq = np.unique(scores)
    if uniq.size == 0:
        return 0.0, -1.0

    best_t = float(uniq[0])
    best_f1 = -1.0
    for t in uniq:
        y_pred = (scores >= t).astype(int)
        f1 = float(compute_metrics(y_true, y_pred, y_score=scores)["F1"])
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def _insert_score_rule(fn_source: str, feature: str, delta: int) -> str | None:
    try:
        mod = ast.parse(fn_source)
    except Exception:
        return None
    fndef: ast.FunctionDef | None = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            fndef = n
            break
    if fndef is None or not fndef.body:
        return None

    insert_at = len(fndef.body)
    for idx in range(len(fndef.body) - 1, -1, -1):
        if isinstance(fndef.body[idx], ast.Return):
            insert_at = idx
            break

    snippet = f'if _get_int("{feature}") == 1:\n    score += {int(delta)}\n'
    try:
        stmts = ast.parse(snippet).body
    except Exception:
        return None

    fndef.body[insert_at:insert_at] = stmts
    tmp_mod = ast.Module(body=[fndef], type_ignores=[])
    ast.fix_missing_locations(tmp_mod)
    try:
        return ast.unparse(tmp_mod)
    except Exception:
        return None


def _tweak_score_penalty(fn_source: str, from_value: int, to_value: int) -> str | None:
    try:
        mod = ast.parse(fn_source)
    except Exception:
        return None
    fndef: ast.FunctionDef | None = None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            fndef = n
            break
    if fndef is None:
        return None

    changed = 0
    for node in ast.walk(fndef):
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "score"
            and isinstance(node.op, ast.Sub)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
            and float(node.value.value) == float(from_value)
        ):
            node.value = ast.Constant(value=int(to_value))
            changed += 1
    if changed == 0:
        return None

    tmp_mod = ast.Module(body=[fndef], type_ignores=[])
    ast.fix_missing_locations(tmp_mod)
    try:
        return ast.unparse(tmp_mod)
    except Exception:
        return None


def _rename_first_function(fn_source: str, new_name: str) -> str | None:
    try:
        mod = ast.parse(fn_source)
    except Exception:
        return None
    for n in mod.body:
        if isinstance(n, ast.FunctionDef):
            n.name = new_name
            tmp_mod = ast.Module(body=[n], type_ignores=[])
            ast.fix_missing_locations(tmp_mod)
            try:
                return ast.unparse(tmp_mod)
            except Exception:
                return None
    return None


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
        block += f'\nERROR_ANALYSIS_{fn_name} = {json.dumps(error_analysis, ensure_ascii=False)}\n'
    append_text(path, block)


def _parse_proposal(text: str) -> ParsedProposal:
    raw = strip_code_fences(text)
    data = json.loads(raw)
    return ParsedProposal(
        version=str(data.get("version", "")),
        error_analysis=str(data.get("error_analysis", "")),
        new_policy_code=str(data.get("new_policy_code", "")),
    )


def _predict_scores_from_code(code: str, fn_name: str, df: pd.DataFrame, label_col: str) -> np.ndarray | None:
    src = _extract_function_source(code, fn_name)
    if src is None:
        return None
    score_fn = _build_score_function(src, score_fn_name="__score_fn")
    if score_fn is None:
        return None
    feature_cols = [c for c in df.columns if c != label_col]
    scores: list[float] = []
    for _, row in df.iterrows():
        feats = {c: row[c] for c in feature_cols}
        try:
            scores.append(float(score_fn(feats)))
        except Exception:
            scores.append(float("nan"))
    arr = np.asarray(scores, dtype=float)
    if np.isnan(arr).all():
        return None
    return arr


def _train_and_eval_baselines(
    train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str, out_dir: Path, random_seed: int
) -> dict[str, dict[str, float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    feature_cols = [c for c in train_df.columns if c != label_col]
    x_train = train_df[feature_cols].astype(float)
    x_test = test_df[feature_cols].astype(float)
    fill = x_train.median(numeric_only=True)
    x_train = x_train.fillna(fill)
    x_test = x_test.fillna(fill)
    y_train = train_df[label_col].astype(int).to_numpy()
    y_test = test_df[label_col].astype(int).to_numpy()

    results: dict[str, dict[str, float]] = {}

    lr = LogisticRegression(max_iter=4000)
    lr.fit(x_train, y_train)
    y_score = lr.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    results["baseline_lr"] = compute_metrics(y_test, y_pred, y_score=y_score)
    (out_dir / "baseline_lr.pkl").write_bytes(pickle.dumps(lr))

    dt = DecisionTreeClassifier(random_state=random_seed, max_depth=4)
    dt.fit(x_train, y_train)
    y_score = dt.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    results["baseline_dt"] = compute_metrics(y_test, y_pred, y_score=y_score)
    (out_dir / "baseline_dt.pkl").write_bytes(pickle.dumps(dt))

    try:
        import xgboost as xgb  # type: ignore

        xgbm = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_seed,
            eval_metric="logloss",
        )
        xgbm.fit(x_train, y_train)
        y_score = xgbm.predict_proba(x_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        results["baseline_xgb"] = compute_metrics(y_test, y_pred, y_score=y_score)
        (out_dir / "baseline_xgb.pkl").write_bytes(pickle.dumps(xgbm))
    except Exception:
        pass

    return results


def run_univariate_probe_task(
    *,
    train_df: pd.DataFrame,
    label_col: str,
    run_cfg: RunConfig,
    univariate_path: Path,
    feature_cols: list[str],
) -> tuple[list[str], list[str], str]:
    univariate_df: pd.DataFrame | None = None
    if run_cfg.run_univariate_probe:
        univariate_df = run_univariate_probe(train_df=train_df, label_col=label_col)
        univariate_df.to_csv(univariate_path, index=False)
    elif univariate_path.exists():
        try:
            univariate_df = pd.read_csv(univariate_path)
        except Exception:
            univariate_df = None

    if univariate_df is not None and len(univariate_df) > 0:
        topk = min(run_cfg.univariate_top_k, len(univariate_df))
        top_features = univariate_df.head(topk)["feature"].tolist()
        report_features = list(top_features)
        univariate_summary = univariate_df.head(topk)[
            ["rank", "feature", "feature_type", "method", "p_value", "missing_rate", "pointbiserial_r", "mwu_p", "binned_or_q4_rel_to_q1"]
        ].to_string(index=False)
        return top_features, report_features, univariate_summary

    return list(feature_cols), list(feature_cols), ""


def run_knowledge_probe_task(
    *,
    client: LLMClient | None,
    top_features: list[str],
    label_col: str,
    run_cfg: RunConfig,
    knowledge_path: Path,
) -> str:
    if client is not None and run_cfg.run_knowledge_probe:
        knowledge = run_knowledge_probe(
            client=client,
            features=top_features[: run_cfg.knowledge_top_k],
            target=label_col,
            task_description=run_cfg.task_description,
        )
        knowledge_table = knowledge.markdown_table
        write_text(knowledge_path, knowledge_table)
        return knowledge_table

    if knowledge_path.exists():
        try:
            return knowledge_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


def generate_v0_task(
    *,
    client: LLMClient | None,
    run_cfg: RunConfig,
    heuristic_path: Path,
    univariate_summary: str,
    knowledge_table: str,
    metric_desc: str,
) -> None:
    if heuristic_path.exists():
        return
    if not run_cfg.run_v0_generation:
        raise RuntimeError("heuristic_system.py not found and run_v0_generation=False; cannot continue.")
    if client is None:
        raise RuntimeError("llm_enabled=False and heuristic_system.py is missing; cannot generate v0.")

    prompt = get_rule_generation_prompt(
        univariate_summary=univariate_summary,
        knowledge_table=knowledge_table,
        metric_desc=metric_desc,
        task_description=run_cfg.task_description,
    )
    resp = client.chat_json([ChatMessage(role="user", content=prompt)])
    p = _parse_proposal(resp)
    if p.version != "v0":
        raise RuntimeError(f"v0 generation failed: version mismatch (expected v0, got {p.version})")
    validate_python_syntax(p.new_policy_code)
    fn_name = extract_function_name(p.new_policy_code)
    if fn_name != "predict_v0":
        raise RuntimeError(f"v0 generation failed: function name mismatch (expected predict_v0, got {fn_name})")

    header = "CURRENT_VERSION = 'v0'\n\n"
    write_text(heuristic_path, header + p.new_policy_code.strip() + "\n")
    v0_error_analysis = p.error_analysis or "v0"
    append_text(heuristic_path, f"\n\nERROR_ANALYSIS_predict_v0 = {json.dumps(v0_error_analysis, ensure_ascii=False)}\n")


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

    code_all = heuristic_path.read_text(encoding="utf-8")
    y_pred_test_v0 = _predict_with_function(fn_v0, test_df, label_col)
    y_score_test_v0 = _predict_scores_from_code(code_all, "predict_v0", test_df, label_col)
    if y_score_test_v0 is None:
        y_score_test_v0 = y_pred_test_v0
    metrics_v0 = compute_metrics(y_true_test, y_pred_test_v0, y_score=y_score_test_v0)
    v0_analysis = str(ns.get("ERROR_ANALYSIS_predict_v0") or "v0")
    records.append(IterationRecord(version="v0", error_analysis=v0_analysis, metrics=metrics_v0))
    append_text(evolution_results_path, f"v0\t{metrics_v0}\n")
    trajectory_lines.append(f"V0: {v0_analysis}")

    if run_cfg.run_iterations and client is not None:
        for i in range(1, run_cfg.iterations + 1):
            next_version = f"v{i}"
            code_all = heuristic_path.read_text(encoding="utf-8")
            y_pred_train = _predict_with_function(current_fn, train_df, label_col)
            y_score_train = _predict_scores_from_code(code_all, f"predict_{current_version}", train_df, label_col)
            if y_score_train is None:
                y_score_train = y_pred_train
            train_old = compute_metrics(y_true_train, y_pred_train, y_score=y_score_train)
            primary = run_cfg.metric_priority[0] if run_cfg.metric_priority else "F1"
            allowed_degradation = max(run_cfg.degradation_threshold, int(len(train_df) * run_cfg.degradation_rate))
            samples = collect_errors(
                df=train_df,
                label_col=label_col,
                y_pred=y_pred_train,
                max_error_samples=run_cfg.max_error_samples,
                random_seed=run_cfg.random_seed + i,
                feature_cols=report_features,
            )
            error_report = format_error_report(samples, max_details=run_cfg.max_error_details)

            prev_regression_info = (
                format_degradation_warning(last_regressed_indices)
                + "\nregressed_examples_json=\n"
                + json.dumps(last_regressed_examples, ensure_ascii=False)
            )
            degradation_warning = (
                f"Constraints: allowed_regressions={allowed_degradation} "
                f"(min={run_cfg.degradation_threshold}, rate={run_cfg.degradation_rate}); "
                f"current_train_metrics={train_old}; primary_metric={primary}.\n\n"
                f"Last change regressions (previously correct, now wrong):\n{prev_regression_info}"
            )

            proposal: ParsedProposal | None = None
            accepted = False
            attempt = 0
            last_proposal: ParsedProposal | None = None
            while attempt < run_cfg.max_llm_attempts and not accepted:
                attempt += 1

                prompt = get_iteration_prompt(
                    current_code=code_all,
                    error_report=error_report,
                    trajectory="\n".join(trajectory_lines),
                    degradation_warning=degradation_warning,
                    metric_desc=metric_desc,
                    task_description=run_cfg.task_description,
                    next_version=next_version,
                )
                resp = client.chat_json([ChatMessage(role="user", content=prompt)])
                try:
                    proposal = _parse_proposal(resp)
                    last_proposal = proposal
                except Exception as e:
                    degradation_warning = f"JSON parse failed: {e}"
                    continue

                if proposal.version != next_version:
                    degradation_warning = f"version mismatch: expected {next_version}, got {proposal.version}"
                    continue

                new_code = proposal.new_policy_code
                validate_python_syntax(new_code)
                fn_name = extract_function_name(new_code)
                if fn_name != f"predict_{next_version}":
                    degradation_warning = f"function name mismatch: expected predict_{next_version}, got {fn_name}"
                    continue

                tmp_ns: dict = {}
                exec(compile(code_all + "\n\n" + new_code, "<heuristic>", "exec"), tmp_ns, tmp_ns)
                new_fn = tmp_ns.get(f"predict_{next_version}")
                if new_fn is None:
                    degradation_warning = "New function was not defined successfully."
                    continue

                y_pred_old = y_pred_train
                y_pred_new = _predict_with_function(new_fn, train_df, label_col)
                degr = detect_degradation(y_true=y_true_train, y_pred_old=y_pred_old, y_pred_new=y_pred_new)
                examples = collect_degradation_examples(
                    df=train_df,
                    label_col=label_col,
                    degraded_indices=degr.degraded_indices,
                    y_pred_old=y_pred_old,
                    y_pred_new=y_pred_new,
                    feature_cols=report_features,
                    max_samples=run_cfg.degradation_max_examples,
                    random_seed=run_cfg.random_seed + 1000 + i + attempt,
                )

                if len(degr.degraded_indices) > allowed_degradation:
                    degradation_warning = (
                        format_degradation_warning(degr.degraded_indices)
                        + f"\nallowed_regressions={allowed_degradation}\n"
                        + "regressed_examples_json=\n"
                        + json.dumps(examples, ensure_ascii=False)
                    )
                    continue

                y_score_new_train = _predict_scores_from_code(code_all + "\n\n" + new_code, f"predict_{next_version}", train_df, label_col)
                if y_score_new_train is None:
                    y_score_new_train = y_pred_new
                train_new = compute_metrics(y_true_train, y_pred_new, y_score=y_score_new_train)
                old_primary = float(train_old.get(primary, float("nan")))
                new_primary = float(train_new.get(primary, float("nan")))
                old_err = int(np.sum(y_pred_old != y_true_train))
                new_err = int(np.sum(y_pred_new != y_true_train))
                if np.isfinite(old_primary) and np.isfinite(new_primary):
                    if not (new_primary > old_primary + 1e-4 or (new_primary >= old_primary - 1e-6 and new_err < old_err)):
                        degradation_warning = (
                            f"Train metric/errors did not improve: {primary} {old_primary:.6f}->{new_primary:.6f}, "
                            f"errors {old_err}->{new_err}. Please make a minimal change and improve."
                        )
                        continue

                _append_new_version(heuristic_path, version_code=new_code, error_analysis=proposal.error_analysis)
                ns = _load_heuristic_module(heuristic_path)
                current_fn = ns.get(f"predict_{next_version}")
                if current_fn is None:
                    degradation_warning = "Failed to load the new version after writing."
                    continue

                current_version = next_version
                last_regressed_indices = list(degr.degraded_indices)
                last_regressed_examples = list(examples)
                trajectory_lines.append(f"V{i}: {proposal.error_analysis}")
                accepted = True

                code_all = heuristic_path.read_text(encoding="utf-8")
                y_pred_test = _predict_with_function(current_fn, test_df, label_col)
                y_score_test = _predict_scores_from_code(code_all, f"predict_{current_version}", test_df, label_col)
                if y_score_test is None:
                    y_score_test = y_pred_test
                m = compute_metrics(y_true_test, y_pred_test, y_score=y_score_test)
                records.append(IterationRecord(version=current_version, error_analysis=proposal.error_analysis, metrics=m))
                append_text(evolution_results_path, f"{current_version}\t{m}\n")

                iteration_log.append(
                    {
                        "version": current_version,
                        "attempt": attempt,
                        "error_report": error_report,
                        "degradation_warning": degradation_warning,
                        "proposal": asdict(proposal),
                        "train_metrics_old": train_old,
                        "train_metrics_new": train_new,
                        "test_metrics": m,
                    }
                )

            if not accepted:
                iteration_log.append(
                    {
                        "version": next_version,
                        "attempt": attempt,
                        "error_report": error_report,
                        "degradation_warning": degradation_warning,
                        "proposal": asdict(last_proposal) if last_proposal is not None else None,
                        "train_metrics_old": train_old,
                        "test_metrics": None,
                    }
                )
                break

    if run_cfg.run_iterations and client is None:
        iteration_log.append(
            {
                "version": "v1",
                "attempt": 0,
                "error_report": "",
                "degradation_warning": "llm_enabled=False; cannot run iterations.",
                "proposal": None,
                "train_metrics_old": {},
                "test_metrics": None,
            }
        )

    return records, iteration_log


def run_heuristic_learning(
    train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str, run_cfg: RunConfig, llm_cfg: LLMConfig
) -> None:
    ensure_dir(run_cfg.output_dir)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"label_col={label_col} must exist in both train_df and test_df.")

    train_cols = [c for c in train_df.columns if c != label_col]
    test_cols = [c for c in test_df.columns if c != label_col]
    if set(train_cols) != set(test_cols):
        raise ValueError("train_df and test_df must have the same set of feature columns.")

    feature_cols = train_cols

    heuristic_path = run_cfg.output_dir / "heuristic_system.py"
    evolution_results_path = run_cfg.output_dir / "evolution_results.txt"
    iteration_log_path = run_cfg.output_dir / "iteration_log.json"
    final_model_path = run_cfg.output_dir / "final_heuristic_model.py"
    final_comparison_path = run_cfg.output_dir / "final_comparison.txt"
    knowledge_path = run_cfg.output_dir / "probe_knowledge.md"
    univariate_path = run_cfg.output_dir / "probe_univariate_results.csv"

    metric_desc = generate_metric_description(run_cfg.metric_priority)

    client: LLMClient | None = None
    if run_cfg.llm_enabled:
        client = LLMClient(
            base_url=llm_cfg.base_url,
            api_key_env=llm_cfg.api_key_env,
            model_name=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
        )

    top_features, report_features, univariate_summary = run_univariate_probe_task(
        train_df=train_df,
        label_col=label_col,
        run_cfg=run_cfg,
        univariate_path=univariate_path,
        feature_cols=feature_cols,
    )
    knowledge_table = run_knowledge_probe_task(
        client=client,
        top_features=top_features,
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

    def _record_key(r: IterationRecord) -> tuple[float, ...]:
        vals: list[float] = []
        for name in run_cfg.metric_priority:
            v = r.metrics.get(name)
            vals.append(float(v) if v is not None else float("-inf"))
        return tuple(vals)

    final_record = max(records, key=_record_key)
    final_version = final_record.version

    export_code = heuristic_path.read_text(encoding="utf-8")
    export_code = export_code.rstrip() + "\n\n" + f"FINAL_VERSION = {json.dumps(final_version)}\n\n" + (
        "def predict(features: dict) -> int:\n"
        f"    return predict_{final_version}(features)\n"
    ) + "\n\n" + (
        "if __name__ == '__main__':\n"
        "    assert isinstance(predict({}), int)\n"
    )
    write_text(final_model_path, export_code + "\n")

    v0 = records[0]
    v_last = records[-1]
    content = (
        f"METRIC_PRIORITY={run_cfg.metric_priority}\n"
        f"V0={v0.metrics}\n"
        f"FINAL({final_record.version})={final_record.metrics}\n"
        f"LAST({v_last.version})={v_last.metrics}\n"
    )
    write_text(final_comparison_path, content)


def main() -> None:
    df = load_csv(Path("/data/yk/HL/data/YHD_bicarbonate.csv"))
    label_col = "hospital_expire_flag"
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    cfg = RunConfig(output_dir=Path("/data/yk/HL/out3"))
    llm = LLMConfig()
    run_heuristic_learning(train_df=train_df, test_df=test_df, label_col=label_col, run_cfg=cfg, llm_cfg=llm)


if __name__ == "__main__":
    main()
