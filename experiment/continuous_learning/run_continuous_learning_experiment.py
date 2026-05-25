from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
script_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hl.config import LLMConfig
from hl.continuous_learning import ContinuousLearningConfig, DriftConfig, run_continuous_learning
from hl.metrics import compute_metrics
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
    return tuple(part for part in (piece.strip() for piece in s.split(",")) if part)


def _parse_renames(s: str) -> tuple[tuple[str, str], ...]:
    s = (s or "").strip()
    if not s:
        return ()

    items: list[tuple[str, str]] = []
    for part in [piece.strip() for piece in s.split(",") if piece.strip()]:
        if ":" not in part:
            raise ValueError(f"Invalid rename spec: {part}. Expected old:new")
        old_name, new_name = part.split(":", 1)
        old_name = old_name.strip()
        new_name = new_name.strip()
        if not old_name or not new_name:
            raise ValueError(f"Invalid rename spec: {part}. Expected old:new")
        items.append((old_name, new_name))
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
            f"Not enough samples for balanced splits. Need pos>={need_each}, neg>={need_each}, "
            f"got pos={len(pos_df)}, neg={len(neg_df)}"
        )

    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(pos_df.index.to_numpy(dtype=int))
    neg_idx = rng.permutation(neg_df.index.to_numpy(dtype=int))

    def take(indices: np.ndarray, start: int, n_items: int) -> np.ndarray:
        return indices[start : start + n_items]

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


def _apply_feature_drift(df: pd.DataFrame, *, label_col: str, drift: DriftConfig) -> tuple[pd.DataFrame, dict]:
    out_df = df.copy()
    rename_map = {old_name: new_name for old_name, new_name in drift.renamed_cols}
    if rename_map:
        out_df = out_df.rename(columns=rename_map)

    dropped_present = [col for col in drift.dropped_cols if col in out_df.columns and col != label_col]
    if dropped_present:
        out_df = out_df.drop(columns=dropped_present)

    added_missing = [col for col in drift.added_cols if col not in out_df.columns and col != label_col]
    for col in added_missing:
        out_df[col] = np.nan

    if label_col not in out_df.columns:
        raise ValueError(f"label_col={label_col} missing after feature drift application")

    meta = {
        "dropped_cols": list(drift.dropped_cols),
        "dropped_present": dropped_present,
        "added_cols": list(drift.added_cols),
        "added_missing_filled_nan": added_missing,
        "renamed_cols": [{"from": old_name, "to": new_name} for old_name, new_name in drift.renamed_cols],
        "change_note": drift.change_note,
        "prev_hl_out_dir": str(drift.prev_hl_out_dir) if drift.prev_hl_out_dir is not None else "",
    }
    return out_df, meta


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
    feature_cols = [col for col in df.columns if col != label_col]
    preds: list[int] = []
    for _, row in df.iterrows():
        features = {col: row[col] for col in feature_cols}
        preds.append(int(predict_fn(features)))
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

    def add_result(model: str, stage: str, metrics: dict | None, status: str, error: str, subdir: Path) -> None:
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
        for col in cols:
            series = xdf[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                arr = numeric.to_numpy(dtype=float)
                med = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
                numeric = numeric.fillna(med)
                medians[col] = med
                x_mat.append(numeric.to_numpy(dtype=float))
            else:
                codes, uniques = pd.factorize(series.astype(str), sort=True)
                maps[col] = {str(value): int(idx) for idx, value in enumerate(list(uniques))}
                x_mat.append(codes.astype(float))
        x = np.stack(x_mat, axis=1) if x_mat else np.zeros((len(df), 0), dtype=float)
        y = df[label_col].astype(int).to_numpy()
        return x, y, {"cols": cols, "maps": maps, "medians": medians}

    def transform_with_meta(df: pd.DataFrame, meta: dict) -> np.ndarray:
        cols = list(meta.get("cols") or [])
        maps = dict(meta.get("maps") or {})
        medians = dict(meta.get("medians") or {})
        x_mat = []
        for col in cols:
            series = df.get(col)
            if series is None:
                x_mat.append(np.zeros((len(df),), dtype=float))
                continue
            if col in medians:
                numeric = pd.to_numeric(series, errors="coerce")
                numeric = numeric.fillna(float(medians.get(col, 0.0)))
                x_mat.append(numeric.to_numpy(dtype=float))
                continue
            mapping = maps.get(col) or {}
            arr = [float(mapping.get(value, -1)) for value in series.astype(str).tolist()]
            x_mat.append(np.asarray(arr, dtype=float))
        return np.stack(x_mat, axis=1) if x_mat else np.zeros((len(df), 0), dtype=float)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:
        err = str(exc)
        for model_name in ["LogisticRegression", "MLP", "DecisionTree", "XGBoost", "LightGBM", "FT-Transformer"]:
            add_result(model_name, "stage1", None, "missing_dep", err, out_dir / model_name / "stage1")
            add_result(model_name, "stage2", None, "missing_dep", err, out_dir / model_name / "stage2")
        return results

    def align(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out_df = df.copy()
        for col in feature_cols:
            if col not in out_df.columns:
                out_df[col] = np.nan
        keep_cols = list(feature_cols) + [label_col]
        return out_df[[col for col in keep_cols if col in out_df.columns]].copy()

    all_feature_cols = set()
    for df in [train_stage1, val_stage1, test_stage1, train_stage2, val_stage2, test_stage2]:
        all_feature_cols |= {col for col in df.columns if col != label_col}
    feature_cols = sorted(all_feature_cols)

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
        add_result("LogisticRegression", "stage1", eval_pred(yt1, lr.predict(xt1)), "ok", "", lr_s1_dir)
        lr.fit(x2, y2)
        add_result("LogisticRegression", "stage2", eval_pred(yt2, lr.predict(xt2)), "ok", "", lr_s2_dir)
    except Exception as exc:
        add_result("LogisticRegression", "stage1", None, "error", str(exc), lr_s1_dir)
        add_result("LogisticRegression", "stage2", None, "error", str(exc), lr_s2_dir)

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
        add_result("MLP", "stage1", eval_pred(yt1, mlp.predict(xt1)), "ok", "", mlp_s1_dir)
        mlp.fit(x2, y2)
        add_result("MLP", "stage2", eval_pred(yt2, mlp.predict(xt2)), "ok", "", mlp_s2_dir)
    except Exception as exc:
        add_result("MLP", "stage1", None, "error", str(exc), mlp_s1_dir)
        add_result("MLP", "stage2", None, "error", str(exc), mlp_s2_dir)

    dt_s1_dir = baseline_dir / "DecisionTree" / "stage1"
    dt_s2_dir = baseline_dir / "DecisionTree" / "stage2"
    dt_s1_dir.mkdir(parents=True, exist_ok=True)
    dt_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        dt1 = DecisionTreeClassifier(random_state=seed, max_depth=None)
        dt1.fit(x1, y1)
        add_result("DecisionTree", "stage1", eval_pred(yt1, dt1.predict(xt1)), "ok", "", dt_s1_dir)
        dt2 = DecisionTreeClassifier(random_state=seed + 1, max_depth=None)
        dt2.fit(x2, y2)
        add_result("DecisionTree", "stage2", eval_pred(yt2, dt2.predict(xt2)), "retrain", "", dt_s2_dir)
    except Exception as exc:
        add_result("DecisionTree", "stage1", None, "error", str(exc), dt_s1_dir)
        add_result("DecisionTree", "stage2", None, "error", str(exc), dt_s2_dir)

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
        add_result("XGBoost", "stage1", eval_pred(yt1, xgb1.predict(xt1)), "ok", "", xgb_s1_dir)

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
        add_result("XGBoost", "stage2", eval_pred(yt2, xgb2.predict(xt2)), "ok", "", xgb_s2_dir)
    except Exception as exc:
        add_result("XGBoost", "stage1", None, "missing_dep", str(exc), xgb_s1_dir)
        add_result("XGBoost", "stage2", None, "missing_dep", str(exc), xgb_s2_dir)

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
        add_result("LightGBM", "stage1", eval_pred(yt1, lgb1.predict(xt1)), "ok", "", lgb_s1_dir)

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
        lgb2.fit(x2, y2, init_model=getattr(lgb1, "booster_", None))
        add_result("LightGBM", "stage2", eval_pred(yt2, lgb2.predict(xt2)), "ok", "", lgb_s2_dir)
    except Exception as exc:
        add_result("LightGBM", "stage1", None, "missing_dep", str(exc), lgb_s1_dir)
        add_result("LightGBM", "stage2", None, "missing_dep", str(exc), lgb_s2_dir)

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
                self.value_proj = nn.Linear(1, int(d_token))
                self.cls = nn.Parameter(torch.zeros(1, 1, int(d_token)))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=int(d_token),
                    nhead=max(1, min(int(n_heads), int(d_token))),
                    dim_feedforward=int(d_token) * 4,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
                self.head = nn.Sequential(nn.LayerNorm(int(d_token)), nn.Linear(int(d_token), 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                tokens = self.value_proj(x.float().unsqueeze(-1))
                cls = self.cls.expand(x.shape[0], -1, -1)
                encoded = self.encoder(torch.cat([cls, tokens], dim=1))
                return self.head(encoded[:, 0, :]).squeeze(-1)

        def _to_tensors(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
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
            optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss()

            best_state = None
            best_metrics = None
            best_epoch = 0
            bad_rounds = 0
            n_train = int(x_tr_t.shape[0])
            batch_size = max(2, min(int(batch_size), n_train))

            for epoch in range(1, int(max_epochs) + 1):
                model.train()
                indices = torch.randperm(n_train)
                for start in range(0, n_train, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    xb = x_tr_t[batch_idx].to(device)
                    yb = y_tr_t[batch_idx].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(x_val_t.to(device)).detach().cpu().numpy()
                metrics = _metrics_from_logits(y_val, val_logits)
                key = (float(metrics.get("F1", float("-inf"))), float(metrics.get("ACC", float("-inf"))))
                best_key = (
                    float(best_metrics.get("F1", float("-inf"))),
                    float(best_metrics.get("ACC", float("-inf"))),
                ) if best_metrics is not None else (float("-inf"), float("-inf"))

                if key > best_key:
                    best_metrics = metrics
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    best_epoch = epoch
                    bad_rounds = 0
                else:
                    bad_rounds += 1
                    if bad_rounds >= int(patience):
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
            logits_t1 = ft1.to(device)(torch.tensor(xt1, dtype=torch.float32).to(device)).detach().cpu().numpy()
        add_result("FT-Transformer", "stage1", _metrics_from_logits(yt1, logits_t1), "ok", "", ft_s1_dir)
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in ft1.state_dict().items()}, "best_epoch": best_epoch1, "best_val_metrics": best_val_m1},
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
            logits_t2 = ft2.to(device)(torch.tensor(xt2, dtype=torch.float32).to(device)).detach().cpu().numpy()
        add_result("FT-Transformer", "stage2", _metrics_from_logits(yt2, logits_t2), "ok", "", ft_s2_dir)
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in ft2.state_dict().items()}, "best_epoch": best_epoch2, "best_val_metrics": best_val_m2},
            ft_s2_dir / f"seed{seed}_best.pt",
        )
    except Exception as exc:
        add_result("FT-Transformer", "stage1", None, "missing_dep", str(exc), ft_s1_dir)
        add_result("FT-Transformer", "stage2", None, "missing_dep", str(exc), ft_s2_dir)

    return results


def _run_hl_stage(
    *,
    ds: DatasetSpec,
    drift: DriftConfig,
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
    train_df, train_meta = _sample_train_balanced(
        train_pool,
        label_col=ds.label_col,
        train_total=stage.train_total,
        seed=seed + 1000,
        spec=split_spec,
    )

    out_dir = output_root / ds.name / f"seed{seed}" / stage.stage_name / "HL" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    write_json(
        out_dir / "adaptation_spec.json",
        {
            "dataset": ds.name,
            "seed": int(seed),
            "stage": stage.stage_name,
            "train_total": int(stage.train_total),
            "split_spec": {"val_total": split_spec.val_total, "test_total": split_spec.test_total},
            "train_sampling": train_meta,
            "drift": drift_meta,
            "task_description": task_description,
        },
    )

    continuous_cfg = ContinuousLearningConfig(
        output_dir=out_dir,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=task_description,
        random_seed=int(seed),
        llm_enabled=True,
        drift=drift,
    )
    result = run_continuous_learning(
        train_df=train_df,
        test_df=val_df,
        label_col=ds.label_col,
        llm_cfg=llm_cfg,
        continuous_cfg=continuous_cfg,
    )

    predict_fn = _load_predict_fn(result.final_model_path)
    y_true = test_df[ds.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, test_df, label_col=ds.label_col)
    metrics = compute_metrics(y_true, y_pred)

    write_json(
        out_dir / "heldout_test_summary.json",
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
        out_dir / "heldout_test_summary.txt",
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

    stage_result = ModelStageResult(
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
        out_dir=str(out_dir),
    )
    return stage_result, out_dir, {"train_df": train_df, "val_df": val_df, "test_df": test_df}


def _run_dataset(
    *,
    ds: DatasetSpec,
    drift_stage1: DriftConfig,
    drift_stage2_template: DriftConfig,
    seeds: list[int],
    stages: list[StageSpec],
    split_spec: SplitSpec,
    llm_cfg: LLMConfig,
    output_root: Path,
) -> list[ModelStageResult]:
    all_results: list[ModelStageResult] = []
    stage1, stage2 = stages
    for seed in seeds:
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

        drift_stage2 = DriftConfig(
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
        baseline_results = _fit_baselines_two_stage(
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
        all_results.extend(baseline_results)
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
        for result in results:
            writer.writerow(
                {
                    "模型": result.model,
                    "数据集": result.dataset,
                    "seed": str(result.seed),
                    "阶段": result.stage,
                    "ACC": result.acc,
                    "F1": result.f1,
                    "Sensitivity": result.sensitivity,
                    "Specificity": result.specificity,
                    "status": result.status,
                    "error": result.error,
                    "out_dir": result.out_dir,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="MIMIC")
    parser.add_argument("--ukb-csv", type=str, default="./data/UKB.csv")
    parser.add_argument("--yhd-csv", type=str, default="./data/YHD_bicarbonate.csv")
    parser.add_argument("--mimic-csv", type=str, default="./data/MIMIC.csv")
    parser.add_argument("--ukb-prev-hl-outdir", type=str, default="")
    parser.add_argument("--yhd-prev-hl-outdir", type=str, default="")
    parser.add_argument("--mimic-prev-hl-outdir", type=str, default="")
    parser.add_argument("--ukb-label-col", type=str, default="label")
    parser.add_argument("--yhd-label-col", type=str, default="hospital_expire_flag")
    parser.add_argument("--mimic-label-col", type=str, default="death_within_hosp_28days")
    parser.add_argument("--seeds", type=str, default="36,40,42")
    parser.add_argument("--drop-cols", type=str, default="")
    parser.add_argument("--add-cols", type=str, default="")
    parser.add_argument("--rename-cols", type=str, default="")
    parser.add_argument("--change-note", type=str, default="")
    parser.add_argument("--stage1-drop-cols", type=str, default="")
    parser.add_argument("--stage2-drop-cols", type=str, default="")
    parser.add_argument("--stage1-add-cols", type=str, default="")
    parser.add_argument("--stage2-add-cols", type=str, default="")
    parser.add_argument("--stage1-rename-cols", type=str, default="")
    parser.add_argument("--stage2-rename-cols", type=str, default="")
    parser.add_argument("--stage1-change-note", type=str, default="")
    parser.add_argument("--stage2-change-note", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(script_dir / "outputs"))
    parser.add_argument("--llm-base-url", type=str, default=os.getenv("CONTINUOUS_LLM_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--llm-key-env", type=str, default=os.getenv("CONTINUOUS_LLM_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--llm-model", type=str, default=os.getenv("CONTINUOUS_LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--llm-temperature", type=float, default=float(os.getenv("CONTINUOUS_LLM_TEMPERATURE", "0.0")))
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in (args.seeds or "").split(",") if item.strip()]
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

    restored_cols = tuple(col for col in stage1_dropped_cols if col and col not in stage2_dropped_cols)
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
    requested = [name.strip() for name in str(args.datasets).split(",") if name.strip()]
    all_results: list[ModelStageResult] = []

    def mk_prev(prev_dir: str, ds_name: str) -> Path | None:
        prev_dir = str(prev_dir or "").strip()
        if not prev_dir:
            return None
        prev_out_dir = Path(prev_dir)
        if not prev_out_dir.exists():
            raise FileNotFoundError(f"{ds_name}: prev_hl_out_dir not found: {prev_out_dir}")
        return prev_out_dir

    def mk_drift_stage1(prev_out_dir: Path | None) -> DriftConfig:
        note = stage1_change_note
        if prev_out_dir is None and "start from scratch" not in note.lower():
            note = note + " (start from scratch: no previous HL output dir provided)"
        return DriftConfig(
            dropped_cols=stage1_dropped_cols,
            added_cols=stage1_added_cols,
            renamed_cols=stage1_renamed_cols,
            change_note=note,
            prev_hl_out_dir=prev_out_dir,
        )

    def mk_drift_stage2_template() -> DriftConfig:
        return DriftConfig(
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
        all_results.extend(
            _run_dataset(
                ds=ds,
                drift_stage1=mk_drift_stage1(prev),
                drift_stage2_template=mk_drift_stage2_template(),
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
        all_results.extend(
            _run_dataset(
                ds=ds,
                drift_stage1=mk_drift_stage1(prev),
                drift_stage2_template=mk_drift_stage2_template(),
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
        all_results.extend(
            _run_dataset(
                ds=ds,
                drift_stage1=mk_drift_stage1(prev),
                drift_stage2_template=mk_drift_stage2_template(),
                seeds=seeds,
                stages=stages,
                split_spec=split_spec,
                llm_cfg=llm_cfg,
                output_root=output_root,
            )
        )

    out_csv = script_dir / "continuous_results.csv"
    _write_results_csv(out_csv, all_results)
    print(f"continuous_results_csv={out_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
