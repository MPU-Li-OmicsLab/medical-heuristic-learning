from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from continuous_learning_experiment_common import (
    DEFAULT_SEEDS,
    DatasetSpec,
    MIMIC_CSV_PATH,
    MIMIC_LABEL_COL,
    ModelStageResult,
    StageDataBundle,
    build_stage1_drift,
    build_stage2_drift_template,
    get_default_experiment_settings,
    prepare_stage_data_bundle,
    stage_bundle_manifest,
    write_results_csv,
)
from hl.metrics import compute_metrics
from hl.utils.io import write_json


def run_baseline_experiments() -> list[ModelStageResult]:
    settings = get_default_experiment_settings()
    ds = settings.dataset

    results: list[ModelStageResult] = []
    for seed in settings.seeds:
        stage1_bundle = prepare_stage_data_bundle(
            ds=ds,
            drift=build_stage1_drift(settings, ds.prev_hl_out_dir),
            stage=settings.stages[0],
            seed=seed,
            split_spec=settings.split_spec,
        )
        stage2_bundle = prepare_stage_data_bundle(
            ds=ds,
            drift=build_stage2_drift_template(settings),
            stage=settings.stages[1],
            seed=seed,
            split_spec=settings.split_spec,
        )
        baseline_root = settings.output_root / ds.name / f"seed{seed}" / "baselines" / _timestamp()
        baseline_root.mkdir(parents=True, exist_ok=True)
        write_json(
            baseline_root / "stage_data_manifest.json",
            {
                "dataset": ds.name,
                "seed": int(seed),
                "data_flow": "shared_with_hl_via_continuous_learning_experiment_common",
                "stage1": stage_bundle_manifest(stage1_bundle),
                "stage2": stage_bundle_manifest(stage2_bundle),
            },
        )
        results.extend(
            _fit_baselines_two_stage(
                out_dir=baseline_root,
                dataset=ds.name,
                stage1_bundle=stage1_bundle,
                stage2_bundle=stage2_bundle,
            )
        )

    out_csv = SCRIPT_DIR / "continuous_baseline_results.csv"
    write_results_csv(out_csv, results)
    print(f"continuous_baseline_results_csv={out_csv}", flush=True)
    return results


def _fit_baselines_two_stage(
    *,
    out_dir: Path,
    dataset: str,
    stage1_bundle: StageDataBundle,
    stage2_bundle: StageDataBundle,
) -> list[ModelStageResult]:
    label_col = stage1_bundle.label_col
    seed = stage1_bundle.seed
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
                acc=_metric_text(metrics, "ACC"),
                f1=_metric_text(metrics, "F1"),
                sensitivity=_metric_text(metrics, "Sensitivity"),
                specificity=_metric_text(metrics, "Specificity"),
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
            add_result(model_name, stage1_bundle.stage, None, "missing_dep", err, out_dir / model_name / stage1_bundle.stage)
            add_result(model_name, stage2_bundle.stage, None, "missing_dep", err, out_dir / model_name / stage2_bundle.stage)
        return results

    def align(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out_df = df.copy()
        for col in feature_cols:
            if col not in out_df.columns:
                out_df[col] = np.nan
        keep_cols = list(feature_cols) + [label_col]
        return out_df[[col for col in keep_cols if col in out_df.columns]].copy()

    all_feature_cols = set()
    for df in [
        stage1_bundle.train_df,
        stage1_bundle.val_df,
        stage1_bundle.test_df,
        stage2_bundle.train_df,
        stage2_bundle.val_df,
        stage2_bundle.test_df,
    ]:
        all_feature_cols |= {col for col in df.columns if col != label_col}
    feature_cols = sorted(all_feature_cols)

    train1 = align(stage1_bundle.train_df, feature_cols)
    val1 = align(stage1_bundle.val_df, feature_cols)
    test1 = align(stage1_bundle.test_df, feature_cols)
    train2 = align(stage2_bundle.train_df, feature_cols)
    val2 = align(stage2_bundle.val_df, feature_cols)
    test2 = align(stage2_bundle.test_df, feature_cols)

    x1, y1, meta = as_xy(train1)
    xv1 = transform_with_meta(val1.drop(columns=[label_col]), meta)
    xt1 = transform_with_meta(test1.drop(columns=[label_col]), meta)
    yt1 = test1[label_col].astype(int).to_numpy()
    x2 = transform_with_meta(train2.drop(columns=[label_col]), meta)
    y2 = train2[label_col].astype(int).to_numpy()
    xv2 = transform_with_meta(val2.drop(columns=[label_col]), meta)
    xt2 = transform_with_meta(test2.drop(columns=[label_col]), meta)
    yt2 = test2[label_col].astype(int).to_numpy()

    baseline_dir = out_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    write_json(baseline_dir / "feature_meta.json", meta)

    def eval_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return compute_metrics(y_true, y_pred)

    lr_s1_dir = baseline_dir / "LogisticRegression" / stage1_bundle.stage
    lr_s2_dir = baseline_dir / "LogisticRegression" / stage2_bundle.stage
    lr_s1_dir.mkdir(parents=True, exist_ok=True)
    lr_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        lr = LogisticRegression(max_iter=2000, solver="lbfgs", warm_start=True, random_state=seed)
        lr.fit(x1, y1)
        add_result("LogisticRegression", stage1_bundle.stage, eval_pred(yt1, lr.predict(xt1)), "ok", "", lr_s1_dir)
        lr.fit(x2, y2)
        add_result("LogisticRegression", stage2_bundle.stage, eval_pred(yt2, lr.predict(xt2)), "ok", "", lr_s2_dir)
    except Exception as exc:
        add_result("LogisticRegression", stage1_bundle.stage, None, "error", str(exc), lr_s1_dir)
        add_result("LogisticRegression", stage2_bundle.stage, None, "error", str(exc), lr_s2_dir)

    mlp_s1_dir = baseline_dir / "MLP" / stage1_bundle.stage
    mlp_s2_dir = baseline_dir / "MLP" / stage2_bundle.stage
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
        add_result("MLP", stage1_bundle.stage, eval_pred(yt1, mlp.predict(xt1)), "ok", "", mlp_s1_dir)
        mlp.fit(x2, y2)
        add_result("MLP", stage2_bundle.stage, eval_pred(yt2, mlp.predict(xt2)), "ok", "", mlp_s2_dir)
    except Exception as exc:
        add_result("MLP", stage1_bundle.stage, None, "error", str(exc), mlp_s1_dir)
        add_result("MLP", stage2_bundle.stage, None, "error", str(exc), mlp_s2_dir)

    dt_s1_dir = baseline_dir / "DecisionTree" / stage1_bundle.stage
    dt_s2_dir = baseline_dir / "DecisionTree" / stage2_bundle.stage
    dt_s1_dir.mkdir(parents=True, exist_ok=True)
    dt_s2_dir.mkdir(parents=True, exist_ok=True)
    try:
        dt1 = DecisionTreeClassifier(random_state=seed, max_depth=None)
        dt1.fit(x1, y1)
        add_result("DecisionTree", stage1_bundle.stage, eval_pred(yt1, dt1.predict(xt1)), "ok", "", dt_s1_dir)
        dt2 = DecisionTreeClassifier(random_state=seed + 1, max_depth=None)
        dt2.fit(x2, y2)
        add_result("DecisionTree", stage2_bundle.stage, eval_pred(yt2, dt2.predict(xt2)), "retrain", "", dt_s2_dir)
    except Exception as exc:
        add_result("DecisionTree", stage1_bundle.stage, None, "error", str(exc), dt_s1_dir)
        add_result("DecisionTree", stage2_bundle.stage, None, "error", str(exc), dt_s2_dir)

    xgb_s1_dir = baseline_dir / "XGBoost" / stage1_bundle.stage
    xgb_s2_dir = baseline_dir / "XGBoost" / stage2_bundle.stage
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
        add_result("XGBoost", stage1_bundle.stage, eval_pred(yt1, xgb1.predict(xt1)), "ok", "", xgb_s1_dir)
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
        add_result("XGBoost", stage2_bundle.stage, eval_pred(yt2, xgb2.predict(xt2)), "ok", "", xgb_s2_dir)
    except Exception as exc:
        add_result("XGBoost", stage1_bundle.stage, None, "missing_dep", str(exc), xgb_s1_dir)
        add_result("XGBoost", stage2_bundle.stage, None, "missing_dep", str(exc), xgb_s2_dir)

    lgb_s1_dir = baseline_dir / "LightGBM" / stage1_bundle.stage
    lgb_s2_dir = baseline_dir / "LightGBM" / stage2_bundle.stage
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
        add_result("LightGBM", stage1_bundle.stage, eval_pred(yt1, lgb1.predict(xt1)), "ok", "", lgb_s1_dir)
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
        add_result("LightGBM", stage2_bundle.stage, eval_pred(yt2, lgb2.predict(xt2)), "ok", "", lgb_s2_dir)
    except Exception as exc:
        add_result("LightGBM", stage1_bundle.stage, None, "missing_dep", str(exc), lgb_s1_dir)
        add_result("LightGBM", stage2_bundle.stage, None, "missing_dep", str(exc), lgb_s2_dir)

    ft_s1_dir = baseline_dir / "FT-Transformer" / stage1_bundle.stage
    ft_s2_dir = baseline_dir / "FT-Transformer" / stage2_bundle.stage
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
            ft1, x1, y1, xv1, stage1_bundle.val_df[label_col].astype(int).to_numpy(), lr=1e-3, max_epochs=80, patience=10, batch_size=64
        )
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logits_t1 = ft1.to(device)(torch.tensor(xt1, dtype=torch.float32).to(device)).detach().cpu().numpy()
        add_result("FT-Transformer", stage1_bundle.stage, _metrics_from_logits(yt1, logits_t1), "ok", "", ft_s1_dir)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in ft1.state_dict().items()},
                "best_epoch": best_epoch1,
                "best_val_metrics": best_val_m1,
            },
            ft_s1_dir / f"seed{seed}_best.pt",
        )

        ft2 = _FT(n_features=n_features, d_token=64, n_layers=2, n_heads=8, dropout=0.1)
        ft2.load_state_dict({k: v.detach().cpu() for k, v in ft1.state_dict().items()}, strict=True)
        ft2, best_val_m2, best_epoch2 = _train_stage(
            ft2, x2, y2, xv2, stage2_bundle.val_df[label_col].astype(int).to_numpy(), lr=5e-4, max_epochs=80, patience=10, batch_size=16
        )
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logits_t2 = ft2.to(device)(torch.tensor(xt2, dtype=torch.float32).to(device)).detach().cpu().numpy()
        add_result("FT-Transformer", stage2_bundle.stage, _metrics_from_logits(yt2, logits_t2), "ok", "", ft_s2_dir)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in ft2.state_dict().items()},
                "best_epoch": best_epoch2,
                "best_val_metrics": best_val_m2,
            },
            ft_s2_dir / f"seed{seed}_best.pt",
        )
    except Exception as exc:
        add_result("FT-Transformer", stage1_bundle.stage, None, "missing_dep", str(exc), ft_s1_dir)
        add_result("FT-Transformer", stage2_bundle.stage, None, "missing_dep", str(exc), ft_s2_dir)

    return results


def _metric_text(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return f"{float(value):.3f}" if value is not None else ""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    print(
        f"Running baseline continuous learning on MIMIC from {MIMIC_CSV_PATH} "
        f"with label={MIMIC_LABEL_COL} and seeds={DEFAULT_SEEDS}.",
        flush=True,
    )
    run_baseline_experiments()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
