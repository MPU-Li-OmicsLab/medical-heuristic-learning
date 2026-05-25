from __future__ import annotations

import argparse
import csv
import importlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hl.metrics import compute_metrics


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    label_col: str


@dataclass(frozen=True)
class SplitSpec:
    val_total: int = 1000
    test_total: int = 1000
    pos_value: int = 1
    neg_value: int = 0


@dataclass(frozen=True)
class RatioSpec:
    pos: int
    neg: int

    @property
    def name(self) -> str:
        return f"{self.pos}:{self.neg}"


def _is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s):
        return True
    if pd.api.types.is_categorical_dtype(s):
        return True
    return False


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _split_val_test_balanced(df: pd.DataFrame, label_col: str, spec: SplitSpec, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _counts_from_ratio(total: int, ratio: RatioSpec) -> tuple[int, int]:
    if total <= 0:
        raise ValueError("train_total must be positive")
    denom = ratio.pos + ratio.neg
    if denom <= 0:
        raise ValueError("invalid ratio")
    pos = int(round(total * (ratio.pos / denom)))
    pos = max(1, min(pos, total - 1))
    neg = total - pos
    return pos, neg


def _sample_train_ratio(
    train_pool: pd.DataFrame,
    *,
    label_col: str,
    train_total: int,
    ratio: RatioSpec,
    seed: int,
    spec: SplitSpec,
) -> tuple[pd.DataFrame, dict]:
    pos_target, neg_target = _counts_from_ratio(train_total, ratio)
    y = train_pool[label_col].astype(int)
    pos_pool = train_pool.loc[y == spec.pos_value].copy()
    neg_pool = train_pool.loc[y == spec.neg_value].copy()
    if len(pos_pool) == 0 or len(neg_pool) == 0:
        raise ValueError(f"train_pool has no samples for one class. pos={len(pos_pool)}, neg={len(neg_pool)}")

    pos_replace = len(pos_pool) < pos_target
    neg_replace = len(neg_pool) < neg_target

    pos_df = pos_pool.sample(n=pos_target, replace=pos_replace, random_state=seed + 11)
    neg_df = neg_pool.sample(n=neg_target, replace=neg_replace, random_state=seed + 23)
    train_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=seed + 97).reset_index(drop=True)
    meta = {
        "train_total": int(train_total),
        "ratio": ratio.name,
        "pos_target": int(pos_target),
        "neg_target": int(neg_target),
        "pos_available": int(len(pos_pool)),
        "neg_available": int(len(neg_pool)),
        "pos_replace": bool(pos_replace),
        "neg_replace": bool(neg_replace),
    }
    return train_df, meta


def _build_preprocessor(df: pd.DataFrame, label_col: str) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c != label_col]
    cat_cols = [c for c in feature_cols if _is_categorical(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, num_cols, cat_cols


def _predict_labels(model, x_df: pd.DataFrame) -> np.ndarray:
    pred = model.predict(x_df)
    return np.asarray(pred).astype(int)


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _fit_eval_sklearn(
    *,
    model_name: str,
    estimator,
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> dict:
    x_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col].astype(int).to_numpy()

    x_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col].astype(int).to_numpy()

    pipe = Pipeline(steps=[("pre", preprocessor), ("model", estimator)])
    pipe.fit(x_train, y_train)
    y_pred = _predict_labels(pipe, x_test)
    metrics = compute_metrics(y_test, y_pred)
    cm = _confusion_counts(y_test, y_pred)
    return {
        "model": model_name,
        "ACC": metrics.get("ACC"),
        "F1": metrics.get("F1"),
        "Sensitivity": metrics.get("Sensitivity"),
        "Specificity": metrics.get("Specificity"),
        "TP": cm["TP"],
        "FP": cm["FP"],
        "FN": cm["FN"],
        "TN": cm["TN"],
        "status": "ok",
        "error": "",
    }


def _fit_eval_ft_transformer(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    seed: int,
    checkpoint_dir: Path,
) -> dict:
    torch = _try_import("torch")
    if torch is None:
        return {
            "model": "FT-Transformer",
            "ACC": None,
            "F1": None,
            "Sensitivity": None,
            "Specificity": None,
            "TP": None,
            "FP": None,
            "FN": None,
            "TN": None,
            "status": "missing_dependency",
            "error": "torch not installed",
        }

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    feature_cols = [c for c in train_df.columns if c != label_col]
    x_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].astype(int).to_numpy()
    x_val = val_df[feature_cols].copy()
    y_val = val_df[label_col].astype(int).to_numpy()
    x_test = test_df[feature_cols].copy()
    y_test = test_df[label_col].astype(int).to_numpy()

    cat_cols = [c for c in feature_cols if _is_categorical(train_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_medians: dict[str, float] = {}
    for c in num_cols:
        s = pd.to_numeric(x_train[c], errors="coerce")
        num_medians[c] = float(s.median()) if not s.dropna().empty else 0.0

    cat_maps: dict[str, dict[str, int]] = {}
    for c in cat_cols:
        vals = x_train[c].astype(str).fillna("__NA__")
        uniq = vals.unique().tolist()
        mapping = {"__UNK__": 0}
        for idx, v in enumerate(uniq, start=1):
            mapping[str(v)] = idx
        cat_maps[c] = mapping

    def featurize(df_x: pd.DataFrame):
        num = None
        cat = None
        if num_cols:
            arrs = []
            for c in num_cols:
                v = pd.to_numeric(df_x[c], errors="coerce").fillna(num_medians[c]).to_numpy(dtype=np.float32)
                arrs.append(v.reshape(-1, 1))
            num = np.concatenate(arrs, axis=1) if arrs else np.zeros((len(df_x), 0), dtype=np.float32)
        if cat_cols:
            arrs = []
            for c in cat_cols:
                mapping = cat_maps[c]
                v = df_x[c].astype(str).fillna("__NA__").map(lambda x: mapping.get(str(x), 0)).to_numpy(dtype=np.int64)
                arrs.append(v.reshape(-1, 1))
            cat = np.concatenate(arrs, axis=1) if arrs else np.zeros((len(df_x), 0), dtype=np.int64)
        return num, cat

    xtr_num, xtr_cat = featurize(x_train)
    xva_num, xva_cat = featurize(x_val)
    xte_num, xte_cat = featurize(x_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_token = 64
    n_layers = 3
    n_heads = 8
    dropout = 0.1

    class FTTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_num = xtr_num.shape[1] if xtr_num is not None else 0
            self.n_cat = xtr_cat.shape[1] if xtr_cat is not None else 0
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, d_token))
            if self.n_num > 0:
                self.num_proj = torch.nn.Linear(self.n_num, self.n_num * d_token)
            else:
                self.num_proj = None
            self.cat_embs = torch.nn.ModuleList()
            for c in cat_cols:
                sz = len(cat_maps[c])
                self.cat_embs.append(torch.nn.Embedding(sz, d_token))
            enc_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=4 * d_token,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(d_token),
                torch.nn.Linear(d_token, 1),
            )

        def forward(self, x_num, x_cat):
            tokens = []
            b = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            tokens.append(self.cls.expand(b, -1, -1))
            if self.num_proj is not None and x_num is not None:
                z = self.num_proj(x_num).view(b, self.n_num, d_token)
                tokens.append(z)
            if self.n_cat > 0 and x_cat is not None:
                for j, emb in enumerate(self.cat_embs):
                    tokens.append(emb(x_cat[:, j]).unsqueeze(1))
            x = torch.cat(tokens, dim=1)
            x = self.encoder(x)
            return self.head(x[:, 0, :]).squeeze(1)

    model = FTTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    batch_size = 256
    max_epochs = 200
    patience = 10

    def to_tensor_num(a):
        if a is None:
            return None
        return torch.from_numpy(a).to(device)

    def to_tensor_cat(a):
        if a is None:
            return None
        return torch.from_numpy(a).to(device)

    xtr_num_t = to_tensor_num(xtr_num)
    xtr_cat_t = to_tensor_cat(xtr_cat)
    ytr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    xva_num_t = to_tensor_num(xva_num)
    xva_cat_t = to_tensor_cat(xva_cat)

    n = y_train.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)

    def eval_split(xn_t, xc_t, y_np: np.ndarray) -> dict:
        model.eval()
        with torch.no_grad():
            logits = model(xn_t, xc_t)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_pred = (probs >= 0.5).astype(int)
        return compute_metrics(y_np, y_pred)

    best_epoch = -1
    best_key: tuple[float, float] | None = None
    bad_epochs = 0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"seed{int(seed)}_best.pth"

    model.train()
    for epoch in range(max_epochs):
        rng.shuffle(idx)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            bn = xtr_num_t[batch] if xtr_num_t is not None else None
            bc = xtr_cat_t[batch] if xtr_cat_t is not None else None
            by = ytr_t[batch]
            opt.zero_grad(set_to_none=True)
            logits = model(bn, bc)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()

        val_metrics = eval_split(xva_num_t, xva_cat_t, y_val)
        f1 = float(val_metrics.get("F1") or 0.0)
        acc = float(val_metrics.get("ACC") or 0.0)
        key = (f1, acc)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": {"d_token": d_token, "n_layers": n_layers, "n_heads": n_heads, "dropout": dropout},
                    "feature_cols": feature_cols,
                    "cat_cols": cat_cols,
                    "num_cols": num_cols,
                    "num_medians": num_medians,
                    "cat_maps": cat_maps,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_val_metrics": val_metrics,
                    "threshold": 0.5,
                },
                checkpoint_path,
            )
            bad_epochs = 0
            model.train()
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

    xte_num_t = to_tensor_num(xte_num)
    xte_cat_t = to_tensor_cat(xte_cat)
    model.eval()
    with torch.no_grad():
        logits = model(xte_num_t, xte_cat_t)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_pred_test = (probs >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred_test)
    cm = _confusion_counts(y_test, y_pred_test)

    return {
        "model": "FT-Transformer",
        "ACC": metrics.get("ACC"),
        "F1": metrics.get("F1"),
        "Sensitivity": metrics.get("Sensitivity"),
        "Specificity": metrics.get("Specificity"),
        "TP": cm["TP"],
        "FP": cm["FP"],
        "FN": cm["FN"],
        "TN": cm["TN"],
        "status": "ok",
        "error": "",
    }


def _run_cpu_models_block(
    *,
    dataset_name: str,
    csv_path: str,
    label_col: str,
    seed: int,
    train_total: int,
    ratio: RatioSpec,
    val_total: int,
    test_total: int,
) -> list[dict]:
    df = _load_csv(Path(csv_path))
    if label_col not in df.columns:
        raise ValueError(f"{dataset_name}: label_col={label_col} not found")
    df = df.copy()
    df[label_col] = df[label_col].astype(int)

    split_spec = SplitSpec(val_total=val_total, test_total=test_total)
    train_pool, _val_df, test_df = _split_val_test_balanced(df, label_col, split_spec, seed=seed)
    train_df, _meta = _sample_train_ratio(
        train_pool,
        label_col=label_col,
        train_total=train_total,
        ratio=ratio,
        seed=seed + train_total + ratio.pos * 1000 + ratio.neg,
        spec=split_spec,
    )
    preprocessor, _num_cols, _cat_cols = _build_preprocessor(df, label_col)

    xgb = _try_import("xgboost")
    lgb = _try_import("lightgbm")

    rows: list[dict] = []

    mlp_batch_size = min(256, int(train_total))
    use_mlp_early_stopping = int(train_total) >= 100

    models: list[tuple[str, object]] = []
    models.append(
        (
            "LogisticRegression",
            LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None, random_state=seed),
        )
    )
    models.append(("DecisionTree", DecisionTreeClassifier(random_state=seed, max_depth=None)))
    models.append(
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=mlp_batch_size,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=use_mlp_early_stopping,
                random_state=seed,
            ),
        )
    )

    if xgb is not None:
        models.append(
            (
                "XGBoost",
                xgb.XGBClassifier(
                    n_estimators=600,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=1,
                ),
            )
        )
    else:
        rows.append(
            {
                "模型": "XGBoost",
                "数据集": dataset_name,
                "训练集数据量": str(train_total),
                "训练集正负比": ratio.name,
                "ACC": "",
                "F1": "",
                "Sensitivity": "",
                "Specificity": "",
                "TP": "",
                "FP": "",
                "FN": "",
                "TN": "",
                "status": "missing_dependency",
                "error": "xgboost not installed",
            }
        )

    if lgb is not None:
        models.append(
            (
                "LightGBM",
                lgb.LGBMClassifier(
                    n_estimators=1200,
                    learning_rate=0.03,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=1,
                ),
            )
        )
    else:
        rows.append(
            {
                "模型": "LightGBM",
                "数据集": dataset_name,
                "训练集数据量": str(train_total),
                "训练集正负比": ratio.name,
                "ACC": "",
                "F1": "",
                "Sensitivity": "",
                "Specificity": "",
                "TP": "",
                "FP": "",
                "FN": "",
                "TN": "",
                "status": "missing_dependency",
                "error": "lightgbm not installed",
            }
        )

    for model_name, estimator in models:
        try:
            r = _fit_eval_sklearn(
                model_name=model_name,
                estimator=estimator,
                preprocessor=preprocessor,
                train_df=train_df,
                test_df=test_df,
                label_col=label_col,
            )
            rows.append(
                {
                    "模型": model_name,
                    "数据集": dataset_name,
                    "训练集数据量": str(train_total),
                    "训练集正负比": ratio.name,
                    "ACC": f"{float(r['ACC']):.3f}" if r["ACC"] is not None else "",
                    "F1": f"{float(r['F1']):.3f}" if r["F1"] is not None else "",
                    "Sensitivity": f"{float(r['Sensitivity']):.3f}" if r["Sensitivity"] is not None else "",
                    "Specificity": f"{float(r['Specificity']):.3f}" if r["Specificity"] is not None else "",
                    "TP": str(r["TP"]) if r.get("TP") is not None else "",
                    "FP": str(r["FP"]) if r.get("FP") is not None else "",
                    "FN": str(r["FN"]) if r.get("FN") is not None else "",
                    "TN": str(r["TN"]) if r.get("TN") is not None else "",
                    "status": r["status"],
                    "error": r["error"],
                }
            )
        except Exception as e:
            rows.append(
                {
                    "模型": model_name,
                    "数据集": dataset_name,
                    "训练集数据量": str(train_total),
                    "训练集正负比": ratio.name,
                    "ACC": "",
                    "F1": "",
                    "Sensitivity": "",
                    "Specificity": "",
                    "TP": "",
                    "FP": "",
                    "FN": "",
                    "TN": "",
                    "status": "error",
                    "error": str(e),
                }
            )
    return rows


def run_contrast2(seed: int = 42, workers: int = 1) -> Path:
    out_dir = Path("./contrast2")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        DatasetSpec("UKB", Path("./data/UKB.csv"), "label"),
        DatasetSpec("YHD", Path("./data/YHD_bicarbonate.csv"), "hospital_expire_flag"),
    ]
    split_spec = SplitSpec(val_total=1000, test_total=1000)

    train_totals = [1000, 3000]
    ratios = [
        RatioSpec(1, 1),
        RatioSpec(1, 2),
        RatioSpec(2, 1),
        RatioSpec(1, 5),
        RatioSpec(5, 1),
        RatioSpec(1, 10),
        RatioSpec(10, 1),
        RatioSpec(1, 50),
        RatioSpec(50, 1),
    ]

    rows: list[dict] = []

    cpu_jobs: list[tuple[DatasetSpec, int, RatioSpec]] = [(ds, t, r) for ds in datasets for t in train_totals for r in ratios]
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = [
                ex.submit(
                    _run_cpu_models_block,
                    dataset_name=ds.name,
                    csv_path=str(ds.csv_path),
                    label_col=ds.label_col,
                    seed=int(seed),
                    train_total=int(t),
                    ratio=r,
                    val_total=int(split_spec.val_total),
                    test_total=int(split_spec.test_total),
                )
                for ds, t, r in cpu_jobs
            ]
            for fut in as_completed(futs):
                rows.extend(fut.result())
    else:
        for ds, t, r in cpu_jobs:
            rows.extend(
                _run_cpu_models_block(
                    dataset_name=ds.name,
                    csv_path=str(ds.csv_path),
                    label_col=ds.label_col,
                    seed=int(seed),
                    train_total=int(t),
                    ratio=r,
                    val_total=int(split_spec.val_total),
                    test_total=int(split_spec.test_total),
                )
            )

    for ds in datasets:
        df = _load_csv(ds.csv_path)
        if ds.label_col not in df.columns:
            raise ValueError(f"{ds.name}: label_col={ds.label_col} not found")
        df = df.copy()
        df[ds.label_col] = df[ds.label_col].astype(int)

        train_pool, val_df, test_df = _split_val_test_balanced(df, ds.label_col, split_spec, seed=seed)
        for t in train_totals:
            for r in ratios:
                train_df, _meta = _sample_train_ratio(
                    train_pool,
                    label_col=ds.label_col,
                    train_total=t,
                    ratio=r,
                    seed=seed + t + r.pos * 1000 + r.neg,
                    spec=split_spec,
                )
                try:
                    rft = _fit_eval_ft_transformer(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        label_col=ds.label_col,
                        seed=seed,
                        checkpoint_dir=out_dir / "checkpoints" / ds.name / f"train{t}" / f"ratio{r.name.replace(':','_')}",
                    )
                    rows.append(
                        {
                            "模型": "FT-Transformer",
                            "数据集": ds.name,
                            "训练集数据量": str(t),
                            "训练集正负比": r.name,
                            "ACC": f"{float(rft['ACC']):.3f}" if rft["ACC"] is not None else "",
                            "F1": f"{float(rft['F1']):.3f}" if rft["F1"] is not None else "",
                            "Sensitivity": f"{float(rft['Sensitivity']):.3f}" if rft["Sensitivity"] is not None else "",
                            "Specificity": f"{float(rft['Specificity']):.3f}" if rft["Specificity"] is not None else "",
                            "TP": str(rft["TP"]) if rft.get("TP") is not None else "",
                            "FP": str(rft["FP"]) if rft.get("FP") is not None else "",
                            "FN": str(rft["FN"]) if rft.get("FN") is not None else "",
                            "TN": str(rft["TN"]) if rft.get("TN") is not None else "",
                            "status": rft["status"],
                            "error": rft["error"],
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "模型": "FT-Transformer",
                            "数据集": ds.name,
                            "训练集数据量": str(t),
                            "训练集正负比": r.name,
                            "ACC": "",
                            "F1": "",
                            "Sensitivity": "",
                            "Specificity": "",
                            "TP": "",
                            "FP": "",
                            "FN": "",
                            "TN": "",
                            "status": "error",
                            "error": str(e),
                        }
                    )

    model_order = {"XGBoost": 0, "LightGBM": 1, "DecisionTree": 2, "MLP": 3, "FT-Transformer": 4, "LogisticRegression": 5}
    dataset_order = {"UKB": 0, "YHD": 1}
    train_order = {1000: 0, 3000: 1}
    ratio_order = {r.name: i for i, r in enumerate(ratios)}

    def key_fn(row: dict) -> tuple:
        m = str(row.get("模型", ""))
        ds = str(row.get("数据集", ""))
        try:
            t = int(str(row.get("训练集数据量", "")))
        except Exception:
            t = 1_000_000_000
        rr = str(row.get("训练集正负比", ""))
        return (model_order.get(m, 99), dataset_order.get(ds, 99), train_order.get(t, 99), ratio_order.get(rr, 99))

    rows.sort(key=key_fn)

    out_path = out_dir / "contrast2.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "模型",
                "数据集",
                "训练集数据量",
                "训练集正负比",
                "ACC",
                "F1",
                "Sensitivity",
                "Specificity",
                "TP",
                "FP",
                "FN",
                "TN",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = run_contrast2(seed=int(args.seed), workers=int(args.workers))
    print(f"contrast2_csv={out}", flush=True)


if __name__ == "__main__":
    main()
