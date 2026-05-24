from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hl.agent.client import ChatMessage, LLMClient
from hl.config import LLMConfig, RunConfig
from hl.evolution.rule_utils import extract_function_name, strip_code_fences, validate_python_syntax
from hl.metrics import compute_metrics, generate_metric_description
from hl.orchestrator import run_heuristic_learning
from hl.probes.knowledge import run_knowledge_probe
from hl.probes.univariate import run_univariate_probe
from hl.utils.io import write_json, write_text


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    label_col: str


@dataclass(frozen=True)
class SplitSpec:
    val_total: int
    test_total: int
    pos_value: int = 1
    neg_value: int = 0


@dataclass(frozen=True)
class DriftSpec:
    dropped_cols: tuple[str, ...]
    added_cols: tuple[str, ...]
    renamed_cols: tuple[tuple[str, str], ...]
    change_note: str
    prev_hl_out_dir: Path | None


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    train_total: int


@dataclass(frozen=True)
class ModelStageResult:
    model: str
    dataset: str
    seed: int
    stage: str
    acc: str
    f1: str
    sensitivity: str
    specificity: str
    status: str
    error: str
    out_dir: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_csv_list(s: str) -> tuple[str, ...]:
    s = (s or "").strip()
    if not s:
        return ()
    parts = [p.strip() for p in s.split(",")]
    return tuple([p for p in parts if p])


def _parse_renames(s: str) -> tuple[tuple[str, str], ...]:
    s = (s or "").strip()
    if not s:
        return ()
    items: list[tuple[str, str]] = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if ":" not in part:
            raise ValueError(f"Invalid rename spec: {part}. Expected old:new")
        old, new = part.split(":", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Invalid rename spec: {part}. Expected old:new")
        items.append((old, new))
    return tuple(items)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _split_val_test_balanced(
    df: pd.DataFrame,
    label_col: str,
    spec: SplitSpec,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[label_col].astype(int)
    pos_df = df.loc[y == spec.pos_value].copy()
    neg_df = df.loc[y == spec.neg_value].copy()

    n_val_each = spec.val_total // 2
    n_test_each = spec.test_total // 2
    need_each = n_val_each + n_test_each
    if len(pos_df) < need_each or len(neg_df) < need_each:
        raise ValueError(
            f"Not enough samples for balanced splits. Need pos>= {need_each}, neg>= {need_each}, got pos={len(pos_df)}, neg={len(neg_df)}"
        )

    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(pos_df.index.to_numpy(dtype=int))
    neg_idx = rng.permutation(neg_df.index.to_numpy(dtype=int))

    def take(idxs: np.ndarray, start: int, n: int) -> np.ndarray:
        return idxs[start : start + n]

    pos_test = take(pos_idx, 0, n_test_each)
    neg_test = take(neg_idx, 0, n_test_each)
    pos_val = take(pos_idx, n_test_each, n_val_each)
    neg_val = take(neg_idx, n_test_each, n_val_each)

    test_df = pd.concat([pos_df.loc[pos_test], neg_df.loc[neg_test]], axis=0).sample(frac=1.0, random_state=seed)
    val_df = pd.concat([pos_df.loc[pos_val], neg_df.loc[neg_val]], axis=0).sample(frac=1.0, random_state=seed + 1)

    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    used_idx = set(pos_test.tolist()) | set(neg_test.tolist()) | set(pos_val.tolist()) | set(neg_val.tolist())
    train_pool = df.loc[~df.index.isin(list(used_idx))].copy().reset_index(drop=True)
    return train_pool, val_df, test_df


def _sample_train_balanced(
    train_pool: pd.DataFrame,
    *,
    label_col: str,
    train_total: int,
    seed: int,
    spec: SplitSpec,
) -> tuple[pd.DataFrame, dict]:
    if train_total <= 0:
        raise ValueError("train_total must be positive")
    if train_total % 2 != 0:
        raise ValueError("train_total must be even for 1:1 balanced sampling")

    pos_each = train_total // 2
    neg_each = train_total // 2

    y = train_pool[label_col].astype(int)
    pos_pool = train_pool.loc[y == spec.pos_value].copy()
    neg_pool = train_pool.loc[y == spec.neg_value].copy()
    if len(pos_pool) == 0 or len(neg_pool) == 0:
        raise ValueError(f"train_pool has no samples for one class. pos={len(pos_pool)}, neg={len(neg_pool)}")

    pos_replace = len(pos_pool) < pos_each
    neg_replace = len(neg_pool) < neg_each

    pos_df = pos_pool.sample(n=pos_each, replace=pos_replace, random_state=seed + 11)
    neg_df = neg_pool.sample(n=neg_each, replace=neg_replace, random_state=seed + 23)
    train_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=seed + 97).reset_index(drop=True)

    meta = {
        "train_total": int(train_total),
        "pos_target": int(pos_each),
        "neg_target": int(neg_each),
        "pos_available": int(len(pos_pool)),
        "neg_available": int(len(neg_pool)),
        "pos_replace": bool(pos_replace),
        "neg_replace": bool(neg_replace),
    }
    return train_df, meta


def _apply_feature_drift(df: pd.DataFrame, *, label_col: str, drift: DriftSpec) -> tuple[pd.DataFrame, dict]:
    df2 = df.copy()
    rename_map = {a: b for a, b in drift.renamed_cols}
    if rename_map:
        df2 = df2.rename(columns=rename_map)
    dropped_present = [c for c in drift.dropped_cols if c in df2.columns and c != label_col]
    if dropped_present:
        df2 = df2.drop(columns=dropped_present)

    added_missing = [c for c in drift.added_cols if c not in df2.columns and c != label_col]
    if added_missing:
        for c in added_missing:
            df2[c] = np.nan

    if label_col not in df2.columns:
        raise ValueError(f"label_col={label_col} missing after feature drift application")

    meta = {
        "dropped_cols": list(drift.dropped_cols),
        "dropped_present": dropped_present,
        "added_cols": list(drift.added_cols),
        "added_missing_filled_nan": added_missing,
        "renamed_cols": [{"from": a, "to": b} for a, b in drift.renamed_cols],
        "change_note": drift.change_note,
        "prev_hl_out_dir": str(drift.prev_hl_out_dir) if drift.prev_hl_out_dir is not None else "",
    }
    return df2, meta


def _read_text_if_exists(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _format_univariate_summary(df: pd.DataFrame, top_k: int = 30) -> str:
    if df is None or df.empty:
        return ""
    k = min(int(top_k), len(df))
    try:
        return df.head(k).to_string(index=False)
    except Exception:
        return ""


def _parse_markdown_table(md: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.startswith("|") and ln.endswith("|")]
    if len(table_lines) < 2:
        return [], []
    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for ln in table_lines[2:]:
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) < len(header):
            cols = cols + [""] * (len(header) - len(cols))
        rows.append(cols[: len(header)])
    return header, rows


def _render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    if not header:
        return ""
    sep = ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        rr = list(r)
        if len(rr) < len(header):
            rr = rr + [""] * (len(header) - len(rr))
        out.append("| " + " | ".join(rr[: len(header)]) + " |")
    return "\n".join(out).strip()


def _update_probe_univariate(
    *,
    prev_out_dir: Path | None,
    out_dir: Path,
    train_df: pd.DataFrame,
    label_col: str,
    drift: DriftSpec,
) -> tuple[Path, str]:
    prev_path = prev_out_dir / "probe_univariate_results.csv" if prev_out_dir is not None else Path("__missing__")
    prev_df = _read_csv_if_exists(prev_path)
    if prev_df is not None:
        write_text(out_dir / "probe_univariate_results_prev.csv", prev_df.to_csv(index=False))
    else:
        write_text(out_dir / "probe_univariate_results_prev.csv", "")

    new_df = run_univariate_probe(train_df=train_df, label_col=label_col)
    new_df = new_df.copy()

    filtered_prev = prev_df.copy() if prev_df is not None else pd.DataFrame(columns=new_df.columns)
    if not filtered_prev.empty and "feature" in filtered_prev.columns:
        filtered_prev = filtered_prev.loc[~filtered_prev["feature"].astype(str).isin(list(drift.dropped_cols))].copy()
        for a, b in drift.renamed_cols:
            if "feature" in filtered_prev.columns:
                filtered_prev.loc[filtered_prev["feature"].astype(str) == str(a), "feature"] = str(b)

    if "feature" in new_df.columns:
        added_rows = new_df.loc[new_df["feature"].astype(str).isin(list(drift.added_cols))].copy()
    else:
        added_rows = pd.DataFrame(columns=new_df.columns)

    combined = pd.concat([filtered_prev, added_rows], axis=0, ignore_index=True)
    if "p_value" in combined.columns and "missing_rate" in combined.columns:
        combined = combined.sort_values(by=["p_value", "missing_rate"], ascending=[True, True], na_position="last")

    out_path = out_dir / "probe_univariate_results.csv"
    combined.to_csv(out_path, index=False)
    return out_path, _format_univariate_summary(combined)


def _update_probe_knowledge(
    *,
    client: LLMClient,
    prev_out_dir: Path | None,
    out_dir: Path,
    label_col: str,
    drift: DriftSpec,
    task_description: str,
) -> tuple[Path, str]:
    prev_path = prev_out_dir / "probe_knowledge.md" if prev_out_dir is not None else Path("__missing__")
    prev_md = _read_text_if_exists(prev_path).strip()
    write_text(out_dir / "probe_knowledge_prev.md", prev_md + ("\n" if prev_md else ""))

    header, rows = _parse_markdown_table(prev_md)
    feature_idx = None
    for i, h in enumerate(header):
        if h.lower() in {"feature", "features", "变量", "特征"}:
            feature_idx = i
            break
    if feature_idx is None and header:
        feature_idx = 0

    kept_rows: list[list[str]] = []
    if header and rows:
        for r in rows:
            feat = str(r[feature_idx]).strip() if feature_idx is not None else ""
            if feat in drift.dropped_cols:
                continue
            for a, b in drift.renamed_cols:
                if feat == a:
                    r = list(r)
                    r[feature_idx] = b
                    break
            kept_rows.append(r)

    add_features = [c for c in drift.added_cols if c]
    add_md = ""
    add_rows: list[list[str]] = []
    if add_features:
        add_md = run_knowledge_probe(client=client, feature_cols=list(add_features), target=label_col, task_description=task_description).markdown_table
        h2, r2 = _parse_markdown_table(add_md)
        if not header and h2:
            header = h2
            feature_idx = 0
        if header and h2 and [x.strip().lower() for x in header] == [x.strip().lower() for x in h2]:
            add_rows = r2
        elif header and r2:
            for rr in r2:
                add_rows.append(rr[: len(header)] + [""] * max(0, len(header) - len(rr)))

    merged_rows = kept_rows + add_rows
    out_md = _render_markdown_table(header, merged_rows) if header else (prev_md.strip() if prev_md else add_md.strip())
    out_path = out_dir / "probe_knowledge.md"
    write_text(out_path, out_md + ("\n" if out_md else ""))
    return out_path, out_md


def _read_blueprint_final(prev_out_dir: Path | None, max_chars: int = 20000) -> str:
    if prev_out_dir is None:
        return ""
    code = _read_text_if_exists(prev_out_dir / "final_heuristic_model.py")
    code = (code or "").strip()
    if not code:
        return ""
    if len(code) <= max_chars:
        return code
    head = code[: max_chars // 2]
    tail = code[-max_chars // 2 :]
    return head + "\n\n" + "[...TRUNCATED...]\n\n" + tail


def _build_continuous_v0_prompt(
    *,
    univariate_summary: str,
    knowledge_table: str,
    metric_desc: str,
    task_description: str,
    drift: DriftSpec,
    blueprint_code: str,
) -> str:
    parts = [
        "You are updating an existing medical heuristic rule system for binary classification under feature drift.",
        "",
        "You MUST output a single JSON object with keys:",
        '- "version": must be exactly "v0"',
        '- "error_analysis": a short explanation of what changed and why',
        '- "new_policy_code": valid Python code defining ONLY `def predict_v0(features: dict) -> int:`',
        "",
        "Constraints:",
        "- new_policy_code must be valid Python syntax (no code fences).",
        "- Do not reference any dropped features.",
        "- You may reference added features if useful.",
        "- Use features.get('<col>', default) access only.",
        "- Return 0/1 only.",
        "",
        "Feature drift changes:",
        f"- Dropped columns: {list(drift.dropped_cols)}",
        f"- Added columns: {list(drift.added_cols)}",
        f"- Renamed columns: {[{a: b} for a, b in drift.renamed_cols]}",
        f"- Change note: {drift.change_note}",
        "",
        "Task description:",
        task_description.strip(),
        "",
        "Optimization metrics:",
        metric_desc.strip(),
        "",
        "Updated probe 1 (univariate) summary:",
        univariate_summary.strip() if univariate_summary else "(empty)",
        "",
        "Updated probe 2 (knowledge) table:",
        knowledge_table.strip() if knowledge_table else "(empty)",
        "",
        "Blueprint code from previous final model (use it as the base style/logic):",
        blueprint_code.strip() if blueprint_code else "(missing blueprint)",
        "",
        "Now produce the JSON.",
    ]
    return "\n".join(parts).strip()


def _parse_proposal(text: str) -> tuple[str, str, str]:
    raw = strip_code_fences(text)
    data = json.loads(raw)
    version = str(data.get("version", ""))
    error_analysis = str(data.get("error_analysis", ""))
    new_policy_code = str(data.get("new_policy_code", ""))
    return version, error_analysis, new_policy_code


def _generate_continuous_v0(
    *,
    client: LLMClient,
    out_dir: Path,
    drift: DriftSpec,
    univariate_summary: str,
    knowledge_table: str,
    task_description: str,
    metric_priority: tuple[str, ...],
    max_llm_attempts: int,
) -> Path:
    metric_desc = generate_metric_description(metric_priority)
    blueprint_code = _read_blueprint_final(drift.prev_hl_out_dir)
    prompt = _build_continuous_v0_prompt(
        univariate_summary=univariate_summary,
        knowledge_table=knowledge_table,
        metric_desc=metric_desc,
        task_description=task_description,
        drift=drift,
        blueprint_code=blueprint_code,
    )

    last_error: Exception | None = None
    last_resp: str = ""
    last_preview: str = ""
    for attempt in range(1, max(1, int(max_llm_attempts)) + 1):
        resp = client.chat_json([ChatMessage(role="user", content=prompt)])
        last_resp = resp
        try:
            version, error_analysis, new_policy_code = _parse_proposal(resp)
            if version != "v0":
                raise RuntimeError(f"version mismatch (expected v0, got {version})")
            new_policy_code = new_policy_code.strip()
            validate_python_syntax(new_policy_code)
            fn_name = extract_function_name(new_policy_code)
            if fn_name != "predict_v0":
                raise RuntimeError(f"function name mismatch (expected predict_v0, got {fn_name})")
            header = "CURRENT_VERSION = 'v0'\n\n"
            write_text(out_dir / "heuristic_system.py", header + new_policy_code + "\n")
            write_text(out_dir / "v0_error_analysis.txt", (error_analysis or "v0") + "\n")
            write_text(out_dir / "v0_prompt.txt", prompt + "\n")
            return out_dir / "heuristic_system.py"
        except Exception as e:
            last_error = e
            last_preview = (last_resp or "").strip().replace("\n", "\\n")
            if len(last_preview) > 800:
                last_preview = last_preview[:800]
            write_text(out_dir / f"v0_attempt_{attempt}_raw.txt", (last_resp or "") + ("\n" if last_resp else ""))
            continue

    raise RuntimeError(f"v0 generation failed after retries: {last_error}; resp_preview={last_preview}")


def _load_predict_fn(model_path: Path):
    spec = importlib.util.spec_from_file_location("final_heuristic_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict_fn = getattr(module, "predict", None)
    if predict_fn is None:
        raise RuntimeError(f"`predict(features)` not found in {model_path}")
    return predict_fn


def _predict_labels(predict_fn, df: pd.DataFrame, label_col: str) -> np.ndarray:
    feature_cols = [c for c in df.columns if c != label_col]
    preds: list[int] = []
    for _, row in df.iterrows():
        feats = {c: row[c] for c in feature_cols}
        preds.append(int(predict_fn(feats)))
    return np.asarray(preds, dtype=int)


def _fit_baselines_two_stage(
    *,
    out_dir: Path,
    dataset: str,
    train_stage1: pd.DataFrame,
    val_stage1: pd.DataFrame,
    test_stage1: pd.DataFrame,
    train_stage2: pd.DataFrame,
    val_stage2: pd.DataFrame,
    test_stage2: pd.DataFrame,
    label_col: str,
    seed: int,
) -> list[ModelStageResult]:
    results: list[ModelStageResult] = []

    def add_result(model: str, stage: str, metrics: dict | None, status: str, error: str, subdir: Path):
        if metrics is None:
            results.append(
                ModelStageResult(
                    model=model,
                    dataset=dataset,
                    seed=seed,
                    stage=stage,
                    acc="",
                    f1="",
                    sensitivity="",
                    specificity="",
                    status=status,
                    error=error,
                    out_dir=str(subdir),
                )
            )
            return
        results.append(
            ModelStageResult(
                model=model,
                dataset=dataset,
                seed=seed,
                stage=stage,
                acc=f"{float(metrics.get('ACC')):.3f}" if metrics.get("ACC") is not None else "",
                f1=f"{float(metrics.get('F1')):.3f}" if metrics.get("F1") is not None else "",
                sensitivity=f"{float(metrics.get('Sensitivity')):.3f}" if metrics.get("Sensitivity") is not None else "",
                specificity=f"{float(metrics.get('Specificity')):.3f}" if metrics.get("Specificity") is not None else "",
                status=status,
                error=error,
                out_dir=str(subdir),
            )
        )

    def as_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
        xdf = df.drop(columns=[label_col]).copy()
        cols = list(xdf.columns)
        x_mat = []
        maps: dict[str, dict] = {}
        medians: dict[str, float] = {}
        for c in cols:
            s = xdf[c]
            if pd.api.types.is_numeric_dtype(s):
                v = pd.to_numeric(s, errors="coerce")
                med = float(np.nanmedian(v.to_numpy(dtype=float))) if np.isfinite(np.nanmedian(v.to_numpy(dtype=float))) else 0.0
                v = v.fillna(med)
                medians[c] = med
                x_mat.append(v.to_numpy(dtype=float))
            else:
                codes, uniques = pd.factorize(s.astype(str), sort=True)
                maps[c] = {str(u): int(i) for i, u in enumerate(list(uniques))}
                x_mat.append(codes.astype(float))
        x = np.stack(x_mat, axis=1) if x_mat else np.zeros((len(df), 0), dtype=float)
        y = df[label_col].astype(int).to_numpy()
        meta = {"cols": cols, "maps": maps, "medians": medians}
        return x, y, meta

    def transform_with_meta(df: pd.DataFrame, meta: dict) -> np.ndarray:
        cols = list(meta.get("cols") or [])
        maps = dict(meta.get("maps") or {})
        medians = dict(meta.get("medians") or {})
        xdf = df.copy()
        x_mat = []
        for c in cols:
            s = xdf.get(c)
            if s is None:
                x_mat.append(np.zeros((len(df),), dtype=float))
                continue
            if c in medians:
                v = pd.to_numeric(s, errors="coerce")
                v = v.fillna(float(medians.get(c, 0.0)))
                x_mat.append(v.to_numpy(dtype=float))
                continue
            m = maps.get(c) or {}
            arr = []
            for v in s.astype(str).tolist():
                arr.append(float(m.get(v, -1)))
            x_mat.append(np.asarray(arr, dtype=float))
        return np.stack(x_mat, axis=1) if x_mat else np.zeros((len(df), 0), dtype=float)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
    except Exception as e:
        err = str(e)
        for m in ["LogisticRegression", "MLP", "DecisionTree", "XGBoost", "LightGBM", "FT-Transformer"]:
            add_result(m, "stage1", None, "missing_dep", err, out_dir / m / "stage1")
            add_result(m, "stage2", None, "missing_dep", err, out_dir / m / "stage2")
        return results

    def align(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in feature_cols:
            if c not in out.columns:
                out[c] = np.nan
        keep_cols = list(feature_cols) + [label_col]
        out = out[[c for c in keep_cols if c in out.columns]].copy()
        return out

    all_feature_cols = set()
    for d in [train_stage1, val_stage1, test_stage1, train_stage2, val_stage2, test_stage2]:
        all_feature_cols |= set([c for c in d.columns if c != label_col])
    feature_cols = sorted(list(all_feature_cols))

    train1 = align(train_stage1, feature_cols)
    val1 = align(val_stage1, feature_cols)
    test1 = align(test_stage1, feature_cols)
    train2 = align(train_stage2, feature_cols)
    val2 = align(val_stage2, feature_cols)
    test2 = align(test_stage2, feature_cols)

    x1, y1, meta = as_xy(train1)
    xv1 = transform_with_meta(val1.drop(columns=[label_col]), meta)
    yv1 = val1[label_col].astype(int).to_numpy()
    xt1 = transform_with_meta(test1.drop(columns=[label_col]), meta)
    yt1 = test1[label_col].astype(int).to_numpy()

    x2 = transform_with_meta(train2.drop(columns=[label_col]), meta)
    y2 = train2[label_col].astype(int).to_numpy()
    xv2 = transform_with_meta(val2.drop(columns=[label_col]), meta)
    yv2 = val2[label_col].astype(int).to_numpy()
    xt2 = transform_with_meta(test2.drop(columns=[label_col]), meta)
    yt2 = test2[label_col].astype(int).to_numpy()

    baseline_dir = out_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    write_json(baseline_dir / "feature_meta.json", meta)

    def eval_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return compute_metrics(y_true, y_pred)

    lr_s1_dir = baseline_dir / "LogisticRegression" / "stage1"
    lr_s2_dir = baseline_dir / "LogisticRegression" / "stage2"
    lr_s1_dir.mkdir(parents=True, exist_ok=True)
    lr_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        lr = LogisticRegression(max_iter=2000, solver="lbfgs", warm_start=True, random_state=seed)
        lr.fit(x1, y1)
        m1 = eval_pred(yt1, lr.predict(xt1))
        add_result("LogisticRegression", "stage1", m1, "ok", "", lr_s1_dir)
        lr.fit(x2, y2)
        m2 = eval_pred(yt2, lr.predict(xt2))
        add_result("LogisticRegression", "stage2", m2, "ok", "", lr_s2_dir)
    except Exception as e:
        add_result("LogisticRegression", "stage1", None, "error", str(e), lr_s1_dir)
        add_result("LogisticRegression", "stage2", None, "error", str(e), lr_s2_dir)

    mlp_s1_dir = baseline_dir / "MLP" / "stage1"
    mlp_s2_dir = baseline_dir / "MLP" / "stage2"
    mlp_s1_dir.mkdir(parents=True, exist_ok=True)
    mlp_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=min(64, max(2, int(x1.shape[0]))),
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=False,
            warm_start=True,
            random_state=seed,
        )
        mlp.fit(x1, y1)
        m1 = eval_pred(yt1, mlp.predict(xt1))
        add_result("MLP", "stage1", m1, "ok", "", mlp_s1_dir)
        mlp.fit(x2, y2)
        m2 = eval_pred(yt2, mlp.predict(xt2))
        add_result("MLP", "stage2", m2, "ok", "", mlp_s2_dir)
    except Exception as e:
        add_result("MLP", "stage1", None, "error", str(e), mlp_s1_dir)
        add_result("MLP", "stage2", None, "error", str(e), mlp_s2_dir)

    dt_s1_dir = baseline_dir / "DecisionTree" / "stage1"
    dt_s2_dir = baseline_dir / "DecisionTree" / "stage2"
    dt_s1_dir.mkdir(parents=True, exist_ok=True)
    dt_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        dt1 = DecisionTreeClassifier(random_state=seed, max_depth=None)
        dt1.fit(x1, y1)
        m1 = eval_pred(yt1, dt1.predict(xt1))
        add_result("DecisionTree", "stage1", m1, "ok", "", dt_s1_dir)
        dt2 = DecisionTreeClassifier(random_state=seed + 1, max_depth=None)
        dt2.fit(x2, y2)
        m2 = eval_pred(yt2, dt2.predict(xt2))
        add_result("DecisionTree", "stage2", m2, "retrain", "", dt_s2_dir)
    except Exception as e:
        add_result("DecisionTree", "stage1", None, "error", str(e), dt_s1_dir)
        add_result("DecisionTree", "stage2", None, "error", str(e), dt_s2_dir)

    xgb_s1_dir = baseline_dir / "XGBoost" / "stage1"
    xgb_s2_dir = baseline_dir / "XGBoost" / "stage2"
    xgb_s1_dir.mkdir(parents=True, exist_ok=True)
    xgb_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        import xgboost as xgb

        xgb1 = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
        )
        xgb1.fit(x1, y1)
        m1 = eval_pred(yt1, xgb1.predict(xt1))
        add_result("XGBoost", "stage1", m1, "ok", "", xgb_s1_dir)
        xgb2 = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed + 1,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
        )
        xgb2.fit(x2, y2, xgb_model=xgb1.get_booster())
        m2 = eval_pred(yt2, xgb2.predict(xt2))
        add_result("XGBoost", "stage2", m2, "ok", "", xgb_s2_dir)
    except Exception as e:
        add_result("XGBoost", "stage1", None, "missing_dep", str(e), xgb_s1_dir)
        add_result("XGBoost", "stage2", None, "missing_dep", str(e), xgb_s2_dir)

    lgb_s1_dir = baseline_dir / "LightGBM" / "stage1"
    lgb_s2_dir = baseline_dir / "LightGBM" / "stage2"
    lgb_s1_dir.mkdir(parents=True, exist_ok=True)
    lgb_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        import lightgbm as lgb

        lgb1 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.0,
            random_state=seed,
            n_jobs=1,
        )
        lgb1.fit(x1, y1)
        m1 = eval_pred(yt1, lgb1.predict(xt1))
        add_result("LightGBM", "stage1", m1, "ok", "", lgb_s1_dir)
        lgb2 = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.0,
            random_state=seed + 1,
            n_jobs=1,
        )
        init_model = getattr(lgb1, "booster_", None)
        lgb2.fit(x2, y2, init_model=init_model)
        m2 = eval_pred(yt2, lgb2.predict(xt2))
        add_result("LightGBM", "stage2", m2, "ok", "", lgb_s2_dir)
    except Exception as e:
        add_result("LightGBM", "stage1", None, "missing_dep", str(e), lgb_s1_dir)
        add_result("LightGBM", "stage2", None, "missing_dep", str(e), lgb_s2_dir)

    ft_s1_dir = baseline_dir / "FT-Transformer" / "stage1"
    ft_s2_dir = baseline_dir / "FT-Transformer" / "stage2"
    ft_s1_dir.mkdir(parents=True, exist_ok=True)
    ft_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        class _FT(nn.Module):
            def __init__(self, n_features: int, d_token: int = 64, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1) -> None:
                super().__init__()
                self.n_features = int(n_features)
                self.d_token = int(d_token)
                self.value_proj = nn.Linear(1, self.d_token)
                self.cls = nn.Parameter(torch.zeros(1, 1, self.d_token))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_token,
                    nhead=max(1, min(int(n_heads), self.d_token)),
                    dim_feedforward=self.d_token * 4,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
                self.head = nn.Sequential(nn.LayerNorm(self.d_token), nn.Linear(self.d_token, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.float()
                tokens = self.value_proj(x.unsqueeze(-1))
                cls = self.cls.expand(x.shape[0], -1, -1)
                z = torch.cat([cls, tokens], dim=1)
                z = self.encoder(z)
                return self.head(z[:, 0, :]).squeeze(-1)

        def _to_tensors(x: np.ndarray, y: np.ndarray):
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y.astype(np.float32), dtype=torch.float32)

        def _metrics_from_logits(y_true_np: np.ndarray, logits_np: np.ndarray) -> dict:
            probs = 1.0 / (1.0 + np.exp(-logits_np))
            y_pred = (probs >= 0.5).astype(int)
            return compute_metrics(y_true_np, y_pred)

        def _train_stage(
            model: nn.Module,
            x_tr: np.ndarray,
            y_tr: np.ndarray,
            x_val: np.ndarray,
            y_val: np.ndarray,
            *,
            lr: float,
            max_epochs: int,
            patience: int,
            batch_size: int,
        ) -> tuple[nn.Module, dict, int]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            x_tr_t, y_tr_t = _to_tensors(x_tr, y_tr)
            x_val_t, _ = _to_tensors(x_val, y_val)
            opt = optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss()

            best_state = None
            best_metrics = None
            best_epoch = 0
            bad = 0
            n = int(x_tr_t.shape[0])
            bs = max(2, min(int(batch_size), n))

            for epoch in range(1, int(max_epochs) + 1):
                model.train()
                idx = torch.randperm(n)
                for start in range(0, n, bs):
                    b = idx[start : start + bs]
                    xb = x_tr_t[b].to(device)
                    yb = y_tr_t[b].to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(x_val_t.to(device)).detach().cpu().numpy()
                m = _metrics_from_logits(y_val, val_logits)

                key = (float(m.get("F1", float("-inf"))), float(m.get("ACC", float("-inf"))))
                if best_metrics is None:
                    best_metrics = m
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    bad = 0
                else:
                    best_key = (float(best_metrics.get("F1", float("-inf"))), float(best_metrics.get("ACC", float("-inf"))))
                    if key > best_key:
                        best_metrics = m
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        best_epoch = epoch
                        bad = 0
                    else:
                        bad += 1
                        if bad >= int(patience):
                            break

            if best_state is not None:
                model.load_state_dict(best_state, strict=True)
            return model, (best_metrics or {}), int(best_epoch)

        n_features = int(x1.shape[1])
        ft1 = _FT(n_features=n_features, d_token=64, n_layers=2, n_heads=8, dropout=0.1)
        ft1, best_val_m1, best_epoch1 = _train_stage(
            ft1,
            x1,
            y1,
            xv1,
            yv1,
            lr=1e-3,
            max_epochs=80,
            patience=10,
            batch_size=64,
        )
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ft1 = ft1.to(device)
            logits_t = ft1(torch.tensor(xt1, dtype=torch.float32).to(device)).detach().cpu().numpy()
        m1 = _metrics_from_logits(yt1, logits_t)
        add_result("FT-Transformer", "stage1", m1, "ok", "", ft_s1_dir)
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in ft1.state_dict().items()}, "best_epoch": int(best_epoch1), "best_val_metrics": best_val_m1},
            ft_s1_dir / f"seed{seed}_best.pt",
        )

        ft2 = _FT(n_features=n_features, d_token=64, n_layers=2, n_heads=8, dropout=0.1)
        ft2.load_state_dict({k: v.detach().cpu() for k, v in ft1.state_dict().items()}, strict=True)
        ft2, best_val_m2, best_epoch2 = _train_stage(
            ft2,
            x2,
            y2,
            xv2,
            yv2,
            lr=5e-4,
            max_epochs=80,
            patience=10,
            batch_size=16,
        )
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ft2 = ft2.to(device)
            logits_t2 = ft2(torch.tensor(xt2, dtype=torch.float32).to(device)).detach().cpu().numpy()
        m2 = _metrics_from_logits(yt2, logits_t2)
        add_result("FT-Transformer", "stage2", m2, "ok", "", ft_s2_dir)
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in ft2.state_dict().items()}, "best_epoch": int(best_epoch2), "best_val_metrics": best_val_m2},
            ft_s2_dir / f"seed{seed}_best.pt",
        )
    except Exception as e:
        add_result("FT-Transformer", "stage1", None, "missing_dep", str(e), ft_s1_dir)
        add_result("FT-Transformer", "stage2", None, "missing_dep", str(e), ft_s2_dir)

    return results


def _run_hl_stage(
    *,
    ds: DatasetSpec,
    drift: DriftSpec,
    stage: StageSpec,
    seed: int,
    split_spec: SplitSpec,
    llm_cfg: LLMConfig,
    output_root: Path,
) -> tuple[ModelStageResult, Path, dict]:
    df = _load_csv(ds.csv_path)
    if ds.label_col not in df.columns:
        raise ValueError(f"{ds.name}: label_col={ds.label_col} not found")
    df = df.copy()
    df[ds.label_col] = df[ds.label_col].astype(int)
    df, drift_meta = _apply_feature_drift(df, label_col=ds.label_col, drift=drift)

    train_pool, val_df, test_df = _split_val_test_balanced(df, ds.label_col, split_spec, seed=seed)
    train_df, train_meta = _sample_train_balanced(train_pool, label_col=ds.label_col, train_total=stage.train_total, seed=seed + 1000, spec=split_spec)

    base_out_dir = output_root / ds.name / f"seed{seed}" / stage.stage_name / "HL" / _timestamp()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        base_out_dir / "adaptation_spec.json",
        {
            "dataset": ds.name,
            "seed": int(seed),
            "stage": stage.stage_name,
            "train_total": int(stage.train_total),
            "split_spec": {"val_total": split_spec.val_total, "test_total": split_spec.test_total},
            "train_sampling": train_meta,
            "drift": drift_meta,
        },
    )

    client = LLMClient(
        base_url=llm_cfg.base_url,
        api_key_env=llm_cfg.api_key_env,
        model_name=llm_cfg.model_name,
        temperature=llm_cfg.temperature,
        api_key=llm_cfg.api_key,
    )

    prev_dir_text = str(drift.prev_hl_out_dir) if drift.prev_hl_out_dir is not None else "(none)"
    task_description = (
        f"Continuous learning. Dataset={ds.name}. Stage={stage.stage_name}. "
        f"TrainTotal={stage.train_total} balanced 1:1. "
        f"ValTotal={split_spec.val_total} balanced 1:1. "
        f"TestTotal={split_spec.test_total} balanced 1:1. "
        f"PrevHLDir={prev_dir_text}. "
        f"Dropped={list(drift.dropped_cols)}. Added={list(drift.added_cols)}. Renamed={list(drift.renamed_cols)}. "
        f"Note={drift.change_note}"
    )

    uni_path, uni_summary = _update_probe_univariate(
        prev_out_dir=drift.prev_hl_out_dir,
        out_dir=base_out_dir,
        train_df=train_df,
        label_col=ds.label_col,
        drift=drift,
    )
    kn_path, kn_table = _update_probe_knowledge(
        client=client,
        prev_out_dir=drift.prev_hl_out_dir,
        out_dir=base_out_dir,
        label_col=ds.label_col,
        drift=drift,
        task_description=task_description,
    )
    _ = uni_path
    _ = kn_path

    _generate_continuous_v0(
        client=client,
        out_dir=base_out_dir,
        drift=drift,
        univariate_summary=uni_summary,
        knowledge_table=kn_table,
        task_description=task_description,
        metric_priority=("F1", "ACC"),
        max_llm_attempts=4,
    )

    run_cfg = RunConfig(
        output_dir=base_out_dir,
        run_univariate_probe=False,
        run_knowledge_probe=False,
        run_v0_generation=False,
        run_iterations=True,
        task_description=task_description,
        random_seed=int(seed),
        llm_enabled=True,
    )
    run_heuristic_learning(train_df=train_df, test_df=val_df, label_col=ds.label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)

    model_path = base_out_dir / "final_heuristic_model.py"
    predict_fn = _load_predict_fn(model_path)
    y_true = test_df[ds.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, test_df, label_col=ds.label_col)
    metrics = compute_metrics(y_true, y_pred)

    write_json(
        base_out_dir / "heldout_test_summary.json",
        {
            "dataset": ds.name,
            "seed": int(seed),
            "stage": stage.stage_name,
            "train_total": int(stage.train_total),
            "split_spec": {"val_total": split_spec.val_total, "test_total": split_spec.test_total},
            "drift": drift_meta,
            "train_sampling": train_meta,
            "heldout_test_metrics": metrics,
            "llm": {"base_url": llm_cfg.base_url, "model_name": llm_cfg.model_name, "api_key_env": llm_cfg.api_key_env},
        },
    )
    write_text(
        base_out_dir / "heldout_test_summary.txt",
        "\n".join(
            [
                f"dataset={ds.name}",
                f"seed={seed}",
                f"stage={stage.stage_name}",
                f"train_total={stage.train_total}",
                f"prev_hl_out_dir={prev_dir_text}",
                f"dropped_cols={list(drift.dropped_cols)}",
                f"added_cols={list(drift.added_cols)}",
                f"renamed_cols={list(drift.renamed_cols)}",
                f"change_note={drift.change_note}",
                f"train_sampling={train_meta}",
                f"heldout_test_metrics={metrics}",
            ]
        )
        + "\n",
    )

    r = ModelStageResult(
        model="HL",
        dataset=ds.name,
        seed=seed,
        stage=stage.stage_name,
        acc=f"{float(metrics.get('ACC')):.3f}" if metrics.get("ACC") is not None else "",
        f1=f"{float(metrics.get('F1')):.3f}" if metrics.get("F1") is not None else "",
        sensitivity=f"{float(metrics.get('Sensitivity')):.3f}" if metrics.get("Sensitivity") is not None else "",
        specificity=f"{float(metrics.get('Specificity')):.3f}" if metrics.get("Specificity") is not None else "",
        status="ok",
        error="",
        out_dir=str(base_out_dir),
    )
    return r, base_out_dir, {"train_df": train_df, "val_df": val_df, "test_df": test_df}


def _run_dataset(
    *,
    ds: DatasetSpec,
    drift_stage1: DriftSpec,
    drift_stage2_template: DriftSpec,
    seeds: list[int],
    stages: list[StageSpec],
    split_spec: SplitSpec,
    llm_cfg: LLMConfig,
    output_root: Path,
) -> list[ModelStageResult]:
    all_results: list[ModelStageResult] = []
    for seed in seeds:
        stage1, stage2 = stages
        hl_s1, hl_s1_dir, data_s1 = _run_hl_stage(
            ds=ds,
            drift=drift_stage1,
            stage=stage1,
            seed=seed,
            split_spec=split_spec,
            llm_cfg=llm_cfg,
            output_root=output_root,
        )
        all_results.append(hl_s1)

        drift_stage2 = DriftSpec(
            dropped_cols=drift_stage2_template.dropped_cols,
            added_cols=drift_stage2_template.added_cols,
            renamed_cols=drift_stage2_template.renamed_cols,
            change_note=drift_stage2_template.change_note,
            prev_hl_out_dir=hl_s1_dir,
        )
        hl_s2, hl_s2_dir, data_s2 = _run_hl_stage(
            ds=ds,
            drift=drift_stage2,
            stage=stage2,
            seed=seed,
            split_spec=split_spec,
            llm_cfg=llm_cfg,
            output_root=output_root,
        )
        all_results.append(hl_s2)

        baseline_root = output_root / ds.name / f"seed{seed}" / "baselines" / _timestamp()
        baseline_root.mkdir(parents=True, exist_ok=True)
        b_results = _fit_baselines_two_stage(
            out_dir=baseline_root,
            dataset=ds.name,
            train_stage1=data_s1["train_df"],
            val_stage1=data_s1["val_df"],
            test_stage1=data_s1["test_df"],
            train_stage2=data_s2["train_df"],
            val_stage2=data_s2["val_df"],
            test_stage2=data_s2["test_df"],
            label_col=ds.label_col,
            seed=seed,
        )
        for br in b_results:
            all_results.append(
                ModelStageResult(
                    model=br.model,
                    dataset=ds.name,
                    seed=seed,
                    stage=br.stage,
                    acc=br.acc,
                    f1=br.f1,
                    sensitivity=br.sensitivity,
                    specificity=br.specificity,
                    status=br.status,
                    error=br.error,
                    out_dir=br.out_dir,
                )
            )

        write_json(
            baseline_root / "hl_stage_dirs.json",
            {"stage1_hl_out_dir": str(hl_s1_dir), "stage2_hl_out_dir": str(hl_s2_dir)},
        )

    return all_results


def _write_results_csv(path: Path, results: list[ModelStageResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "模型",
                "数据集",
                "seed",
                "阶段",
                "ACC",
                "F1",
                "Sensitivity",
                "Specificity",
                "status",
                "error",
                "out_dir",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "模型": r.model,
                    "数据集": r.dataset,
                    "seed": str(r.seed),
                    "阶段": r.stage,
                    "ACC": r.acc,
                    "F1": r.f1,
                    "Sensitivity": r.sensitivity,
                    "Specificity": r.specificity,
                    "status": r.status,
                    "error": r.error,
                    "out_dir": r.out_dir,
                }
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default="MIMIC")
    p.add_argument("--ukb-csv", type=str, default="./data/UKB.csv")
    p.add_argument("--yhd-csv", type=str, default="./data/YHD_bicarbonate.csv")
    p.add_argument("--mimic-csv", type=str, default="./data/MIMIC.csv")
    p.add_argument("--ukb-prev-hl-outdir", type=str, default="")
    p.add_argument("--yhd-prev-hl-outdir", type=str, default="")
    p.add_argument("--mimic-prev-hl-outdir", type=str, default="")
    p.add_argument("--ukb-label-col", type=str, default="label")
    p.add_argument("--yhd-label-col", type=str, default="hospital_expire_flag")
    p.add_argument("--mimic-label-col", type=str, default="death_within_hosp_28days")
    p.add_argument("--seeds", type=str, default="36,40,42")
    p.add_argument("--drop-cols", type=str, default="")
    p.add_argument("--add-cols", type=str, default="")
    p.add_argument("--rename-cols", type=str, default="")
    p.add_argument("--change-note", type=str, default="")
    p.add_argument("--stage1-drop-cols", type=str, default="")
    p.add_argument("--stage2-drop-cols", type=str, default="")
    p.add_argument("--stage1-add-cols", type=str, default="")
    p.add_argument("--stage2-add-cols", type=str, default="")
    p.add_argument("--stage1-rename-cols", type=str, default="")
    p.add_argument("--stage2-rename-cols", type=str, default="")
    p.add_argument("--stage1-change-note", type=str, default="")
    p.add_argument("--stage2-change-note", type=str, default="")
    p.add_argument("--output-root", type=str, default="./continuous_learning/outputs")
    p.add_argument("--llm-base-url", type=str, default=os.getenv("CONTINUOUS_LLM_BASE_URL", "https://api.deepseek.com/v1"))
    p.add_argument("--llm-key-env", type=str, default=os.getenv("CONTINUOUS_LLM_KEY_ENV", "DEEPSEEK_API_KEY"))
    p.add_argument("--llm-model", type=str, default=os.getenv("CONTINUOUS_LLM_MODEL", "deepseek-v4-pro"))
    p.add_argument("--llm-temperature", type=float, default=float(os.getenv("CONTINUOUS_LLM_TEMPERATURE", "0.0")))
    args = p.parse_args()

    seeds = [int(s.strip()) for s in (args.seeds or "").split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    stage1_dropped_cols = _parse_csv_list(args.stage1_drop_cols or args.drop_cols)
    stage2_dropped_cols = _parse_csv_list(args.stage2_drop_cols)
    stage1_added_cols = _parse_csv_list(args.stage1_add_cols or args.add_cols)
    stage2_added_explicit = _parse_csv_list(args.stage2_add_cols)
    stage1_renamed_cols = _parse_renames(args.stage1_rename_cols or args.rename_cols)
    stage2_renamed_cols = _parse_renames(args.stage2_rename_cols or args.rename_cols)

    stage1_change_note = str(args.stage1_change_note or args.change_note or "").strip()
    stage2_change_note = str(args.stage2_change_note or "").strip()
    if not stage1_change_note:
        raise ValueError("stage1 change note is required. Use --stage1-change-note (or --change-note).")
    if not stage2_change_note:
        raise ValueError("stage2 change note is required. Use --stage2-change-note.")

    restored_cols = tuple([c for c in stage1_dropped_cols if c and c not in stage2_dropped_cols])
    stage2_added_cols = tuple(dict.fromkeys(list(stage2_added_explicit) + list(restored_cols)).keys())

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    llm_cfg = LLMConfig(
        base_url=str(args.llm_base_url),
        api_key_env=str(args.llm_key_env),
        model_name=str(args.llm_model),
        temperature=float(args.llm_temperature),
    )

    split_spec = SplitSpec(val_total=500, test_total=500)
    stages = [StageSpec("stage1_train1000", 1000), StageSpec("stage2_train10", 10)]

    requested = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    all_results: list[ModelStageResult] = []

    def mk_prev(prev_dir: str, ds_name: str) -> Path | None:
        prev_dir = str(prev_dir or "").strip()
        if not prev_dir:
            return None
        prev_out_dir = Path(prev_dir)
        if not prev_out_dir.exists():
            raise FileNotFoundError(f"{ds_name}: prev_hl_out_dir not found: {prev_out_dir}")
        return prev_out_dir

    def mk_drift_stage1(prev_out_dir: Path | None, ds_name: str) -> DriftSpec:
        note = stage1_change_note
        if prev_out_dir is None and "start from scratch" not in note.lower():
            note = note + " (start from scratch: no previous HL output dir provided)"
        return DriftSpec(
            dropped_cols=stage1_dropped_cols,
            added_cols=stage1_added_cols,
            renamed_cols=stage1_renamed_cols,
            change_note=note,
            prev_hl_out_dir=prev_out_dir,
        )

    def mk_drift_stage2_template(ds_name: str) -> DriftSpec:
        return DriftSpec(
            dropped_cols=stage2_dropped_cols,
            added_cols=stage2_added_cols,
            renamed_cols=stage2_renamed_cols,
            change_note=stage2_change_note,
            prev_hl_out_dir=None,
        )

    if "UKB" in requested:
        if not str(args.ukb_prev_hl_outdir).strip():
            raise ValueError("UKB requested but --ukb-prev-hl-outdir is empty")
        ds = DatasetSpec("UKB", Path(args.ukb_csv), str(args.ukb_label_col))
        prev = mk_prev(args.ukb_prev_hl_outdir, "UKB")
        drift1 = mk_drift_stage1(prev, "UKB")
        drift2 = mk_drift_stage2_template("UKB")
        all_results.extend(
            _run_dataset(
                ds=ds,
                drift_stage1=drift1,
                drift_stage2_template=drift2,
                seeds=seeds,
                stages=stages,
                split_spec=split_spec,
                llm_cfg=llm_cfg,
                output_root=output_root,
            )
        )
    if "YHD" in requested:
        if not str(args.yhd_prev_hl_outdir).strip():
            raise ValueError("YHD requested but --yhd-prev-hl-outdir is empty")
        ds = DatasetSpec("YHD", Path(args.yhd_csv), str(args.yhd_label_col))
        prev = mk_prev(args.yhd_prev_hl_outdir, "YHD")
        drift1 = mk_drift_stage1(prev, "YHD")
        drift2 = mk_drift_stage2_template("YHD")
        all_results.extend(
            _run_dataset(
                ds=ds,
                drift_stage1=drift1,
                drift_stage2_template=drift2,
                seeds=seeds,
                stages=stages,
                split_spec=split_spec,
                llm_cfg=llm_cfg,
                output_root=output_root,
            )
        )
    if "MIMIC" in requested:
        ds = DatasetSpec("MIMIC", Path(args.mimic_csv), str(args.mimic_label_col))
        prev = mk_prev(args.mimic_prev_hl_outdir, "MIMIC")
        drift1 = mk_drift_stage1(prev, "MIMIC")
        drift2 = mk_drift_stage2_template("MIMIC")
        all_results.extend(
            _run_dataset(
                ds=ds,
                drift_stage1=drift1,
                drift_stage2_template=drift2,
                seeds=seeds,
                stages=stages,
                split_spec=split_spec,
                llm_cfg=llm_cfg,
                output_root=output_root,
            )
        )

    out_csv = Path("./continuous_learning/continuous_results.csv")
    _write_results_csv(out_csv, all_results)
    print(f"continuous_results_csv={out_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
