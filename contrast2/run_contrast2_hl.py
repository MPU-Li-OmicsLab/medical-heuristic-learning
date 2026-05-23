from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hl.config import LLMConfig, RunConfig
from hl.metrics import compute_metrics
from hl.orchestrator import run_heuristic_learning
from hl.utils.io import write_text


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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    denom = ratio.pos + ratio.neg
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


def _read_final_version(model_path: Path) -> str:
    try:
        code_all = model_path.read_text(encoding="utf-8")
        for line in code_all.splitlines():
            if line.startswith("FINAL_VERSION"):
                return str(json.loads(line.split("=", 1)[1].strip()))
    except Exception:
        return ""
    return ""


def _run_one(
    *,
    ds: DatasetSpec,
    split_spec: SplitSpec,
    seed: int,
    train_total: int,
    ratio: RatioSpec,
    output_root: Path,
) -> dict:
    df = _load_csv(ds.csv_path)
    if ds.label_col not in df.columns:
        raise ValueError(f"{ds.name}: label_col={ds.label_col} not found")
    df = df.copy()
    df[ds.label_col] = df[ds.label_col].astype(int)

    train_pool, val_df, test_df = _split_val_test_balanced(df, ds.label_col, split_spec, seed=seed)
    train_df, train_meta = _sample_train_ratio(
        train_pool=train_pool,
        label_col=ds.label_col,
        train_total=train_total,
        ratio=ratio,
        seed=seed + train_total + ratio.pos * 1000 + ratio.neg,
        spec=split_spec,
    )

    out_dir = output_root / ds.name / f"train{train_total}" / f"ratio{ratio.slug}" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = RunConfig(
        output_dir=out_dir,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=(
            f"Dataset={ds.name}. Probes=U1_K1. "
            f"TrainTotal={train_total}. TrainRatio={ratio.name}. "
            "Binary classification on a clinical/tabular dataset with balanced val/test splits."
        ),
    )

    llm_cfg = LLMConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model_name="deepseek-v4-pro",
        temperature=0.0,
    )

    run_heuristic_learning(train_df=train_df, test_df=val_df, label_col=ds.label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)

    model_path = out_dir / "final_heuristic_model.py"
    predict_fn = _load_predict_fn(model_path)

    y_true = test_df[ds.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, test_df, label_col=ds.label_col)
    metrics = compute_metrics(y_true, y_pred)
    final_version = _read_final_version(model_path)

    write_text(
        out_dir / "heldout_test_summary.txt",
        "\n".join(
            [
                f"dataset={ds.name}",
                f"train_total={train_total}",
                f"train_ratio={ratio.name}",
                f"final_version={final_version}",
                f"train_sampling={train_meta}",
                f"heldout_test_metrics={metrics}",
            ]
        )
        + "\n",
    )

    return {
        "模型": f"HL({ratio.name})",
        "数据集": ds.name,
        "训练集数据量": str(train_total),
        "ACC": f"{float(metrics.get('ACC')):.3f}" if metrics.get("ACC") is not None else "",
        "F1": f"{float(metrics.get('F1')):.3f}" if metrics.get("F1") is not None else "",
        "Sensitivity": f"{float(metrics.get('Sensitivity')):.3f}" if metrics.get("Sensitivity") is not None else "",
        "Specificity": f"{float(metrics.get('Specificity')):.3f}" if metrics.get("Specificity") is not None else "",
        "status": "ok",
        "out_dir": str(out_dir),
        "error": "",
    }


def _run_one_safe(*, ds: DatasetSpec, split_spec: SplitSpec, seed: int, train_total: int, ratio: RatioSpec, output_root: Path) -> dict:
    try:
        return _run_one(ds=ds, split_spec=split_spec, seed=seed, train_total=train_total, ratio=ratio, output_root=output_root)
    except Exception as e:
        return {
            "模型": f"HL({ratio.name})",
            "数据集": ds.name,
            "训练集数据量": str(train_total),
            "ACC": "",
            "F1": "",
            "Sensitivity": "",
            "Specificity": "",
            "status": "error",
            "out_dir": "",
            "error": str(e) + "\n" + traceback.format_exc(),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-root", type=str, default="./contrast2/outputs_hl")
    args = p.parse_args()

    seed = int(args.seed)
    workers = int(args.workers)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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

    tasks: list[tuple[DatasetSpec, int, RatioSpec]] = [(ds, t, r) for ds in datasets for t in train_totals for r in ratios]

    results: list[dict] = []
    if workers <= 1:
        for ds, t, r in tasks:
            results.append(_run_one_safe(ds=ds, split_spec=split_spec, seed=seed, train_total=t, ratio=r, output_root=output_root))
    else:
        if workers > (os.cpu_count() or 1) * 4:
            print("[contrast2_hl] warning: workers is very high; you may hit API rate limits.", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_run_one_safe, ds=ds, split_spec=split_spec, seed=seed, train_total=t, ratio=r, output_root=output_root)
                for ds, t, r in tasks
            ]
            for fut in as_completed(futs):
                results.append(fut.result())

    dataset_order = {"UKB": 0, "YHD": 1}
    train_order = {1000: 0, 3000: 1}
    ratio_order = {r.name: i for i, r in enumerate(ratios)}

    def sort_key(row: dict) -> tuple:
        ds = str(row.get("数据集", ""))
        try:
            t = int(str(row.get("训练集数据量", "")))
        except Exception:
            t = 1_000_000_000
        model = str(row.get("模型", ""))
        ratio_name = model.removeprefix("HL(").removesuffix(")")
        return (dataset_order.get(ds, 99), train_order.get(t, 99), ratio_order.get(ratio_name, 99))

    results.sort(key=sort_key)

    out_csv = Path("./contrast2/contrast2_hl.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["模型", "数据集", "训练集数据量", "ACC", "F1", "Sensitivity", "Specificity"],
        )
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in writer.fieldnames} for r in results])

    print(f"contrast2_hl_csv={out_csv}", flush=True)


if __name__ == "__main__":
    main()

