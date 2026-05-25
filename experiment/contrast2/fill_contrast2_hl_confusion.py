from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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

    @property
    def slug(self) -> str:
        return f"{self.pos}_{self.neg}"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _split_val_test_balanced(
    df: pd.DataFrame, label_col: str, spec: SplitSpec, seed: int
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


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


_MODEL_RE = re.compile(r"^HL\((\d+):(\d+)\)$")


def _parse_ratio(model_cell: str) -> RatioSpec:
    m = _MODEL_RE.match(model_cell.strip())
    if not m:
        raise ValueError(f"Unexpected 模型 value: {model_cell}")
    return RatioSpec(pos=int(m.group(1)), neg=int(m.group(2)))


def _find_latest_run_dir(output_root: Path, ds_name: str, train_total: int, ratio: RatioSpec) -> Path | None:
    base = output_root / ds_name / f"train{int(train_total)}" / f"ratio{ratio.slug}"
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.name)


def _infer_seed_from_filename(csv_path: Path) -> int:
    m = re.search(r"contrast2_hl_(\d+)\.csv$", csv_path.name)
    if not m:
        raise ValueError(f"Cannot infer seed from filename: {csv_path.name}")
    return int(m.group(1))


def _fill_one_csv(*, output_root: Path, csv_path: Path, repo_root: Path) -> None:
    seed = _infer_seed_from_filename(csv_path)

    datasets = {
        "UKB": DatasetSpec("UKB", repo_root / "data" / "UKB.csv", "label"),
        "YHD": DatasetSpec("YHD", repo_root / "data" / "YHD_bicarbonate.csv", "hospital_expire_flag"),
    }
    split_spec = SplitSpec(val_total=1000, test_total=1000)

    raw_rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            raw_rows.append({k: (row.get(k, "") or "") for k in fieldnames})

    if "TP" not in fieldnames:
        insert_at = fieldnames.index("Specificity") + 1 if "Specificity" in fieldnames else len(fieldnames)
        fieldnames = fieldnames[:insert_at] + ["TP", "FP", "FN", "TN"] + fieldnames[insert_at:]

    cache_test: dict[str, pd.DataFrame] = {}

    def get_test_df(ds: DatasetSpec) -> pd.DataFrame:
        if ds.name in cache_test:
            return cache_test[ds.name]
        df = _load_csv(ds.csv_path)
        if ds.label_col not in df.columns:
            raise ValueError(f"{ds.name}: label_col={ds.label_col} not found")
        df = df.copy()
        df[ds.label_col] = df[ds.label_col].astype(int)
        _train_pool, _val_df, test_df = _split_val_test_balanced(df, ds.label_col, split_spec, seed=seed)
        cache_test[ds.name] = test_df
        return test_df

    for row in raw_rows:
        if row.get("TP") and row.get("FP") and row.get("FN") and row.get("TN"):
            continue
        ds_name = row.get("数据集", "").strip()
        if ds_name not in datasets:
            continue
        ds = datasets[ds_name]
        try:
            train_total = int(str(row.get("训练集数据量", "")).strip())
        except Exception:
            continue
        try:
            ratio = _parse_ratio(row.get("模型", ""))
        except Exception:
            continue

        run_dir = _find_latest_run_dir(output_root, ds.name, train_total, ratio)
        if run_dir is None:
            row["TP"] = ""
            row["FP"] = ""
            row["FN"] = ""
            row["TN"] = ""
            continue
        model_path = run_dir / "final_heuristic_model.py"
        if not model_path.exists():
            row["TP"] = ""
            row["FP"] = ""
            row["FN"] = ""
            row["TN"] = ""
            continue

        test_df = get_test_df(ds)
        y_true = test_df[ds.label_col].astype(int).to_numpy()
        predict_fn = _load_predict_fn(model_path)
        y_pred = _predict_labels(predict_fn, test_df, label_col=ds.label_col)
        cm = _confusion_counts(y_true, y_pred)
        row["TP"] = str(cm["TP"])
        row["FP"] = str(cm["FP"])
        row["FN"] = str(cm["FN"])
        row["TN"] = str(cm["TN"])

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fieldnames} for r in raw_rows])


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=str, default=str(repo_root))
    p.add_argument(
        "--output-roots",
        type=str,
        default="",
        help="Comma-separated output roots, e.g. ./experiment/contrast2/outputs_hl or ./experiment/contrast2/outputs_hl_42 . If empty, auto-detect ./experiment/contrast2/outputs_hl and ./experiment/contrast2/outputs_hl_*.",
    )
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()

    if args.output_roots.strip():
        roots = [Path(s.strip()) for s in args.output_roots.split(",") if s.strip()]
    else:
        roots = []
        default_root = script_dir / "outputs_hl"
        if default_root.exists():
            roots.append(default_root)
        roots.extend(sorted(script_dir.glob("outputs_hl_*")))

    for root in roots:
        root = root.resolve()
        csv_files = list(root.glob("contrast2_hl_*.csv"))
        if not csv_files:
            continue
        for csv_path in csv_files:
            _fill_one_csv(output_root=root, csv_path=csv_path, repo_root=repo_root)
            print(f"updated_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
