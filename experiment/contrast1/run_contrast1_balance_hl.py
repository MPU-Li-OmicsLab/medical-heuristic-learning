from __future__ import annotations

import argparse
import csv
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

repo_root = Path(__file__).resolve().parents[2]
script_dir = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hl.config import LLMConfig, RunConfig
from hl.metrics import compute_metrics
from hl.orchestrator import run_heuristic_learning
from hl.utils.io import write_json, write_text


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


def _sample_train_balanced(
    train_pool: pd.DataFrame,
    *,
    label_col: str,
    train_size: int,
    seed: int,
    spec: SplitSpec,
) -> tuple[pd.DataFrame, dict]:
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if train_size % 2 != 0:
        raise ValueError("train_size must be even for 1:1 balanced sampling")

    pos_each = train_size // 2
    neg_each = train_size // 2

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
        "train_size": int(train_size),
        "pos_target": int(pos_each),
        "neg_target": int(neg_each),
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
    train_size: int,
    output_root: Path,
    llm_cfg: LLMConfig,
) -> dict:
    df = _load_csv(ds.csv_path)
    if ds.label_col not in df.columns:
        raise ValueError(f"{ds.name}: label_col={ds.label_col} not found")
    df = df.copy()
    df[ds.label_col] = df[ds.label_col].astype(int)

    train_pool, val_df, test_df = _split_val_test_balanced(df, ds.label_col, split_spec, seed=seed)
    train_df, train_meta = _sample_train_balanced(
        train_pool,
        label_col=ds.label_col,
        train_size=int(train_size),
        seed=int(seed) + int(train_size),
        spec=split_spec,
    )

    out_dir = output_root / ds.name / f"train{train_size}" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = RunConfig(
        output_dir=out_dir,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=f"contrast1_balance_hl. Dataset={ds.name}. Probes=U1_K1. TrainSize={train_size} balanced 1:1. Val/Test balanced.",
        random_seed=int(seed),
        llm_enabled=True,
    )

    run_heuristic_learning(train_df=train_df, test_df=val_df, label_col=ds.label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)

    model_path = out_dir / "final_heuristic_model.py"
    predict_fn = _load_predict_fn(model_path)
    y_true = test_df[ds.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, test_df, label_col=ds.label_col)
    metrics = compute_metrics(y_true, y_pred)
    final_version = _read_final_version(model_path)

    write_json(
        out_dir / "heldout_test_summary.json",
        {
            "dataset": ds.name,
            "label_col": ds.label_col,
            "seed": int(seed),
            "train_size": int(train_size),
            "train_sampling": train_meta,
            "final_version": final_version,
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
                f"train_size={train_size}",
                f"final_version={final_version}",
                f"train_sampling={train_meta}",
                f"heldout_test_metrics={metrics}",
            ]
        )
        + "\n",
    )

    return {
        "模型": "HL",
        "数据集": ds.name,
        "训练集数据量": str(train_size),
        "ACC": f"{float(metrics.get('ACC')):.3f}" if metrics.get("ACC") is not None else "",
        "F1": f"{float(metrics.get('F1')):.3f}" if metrics.get("F1") is not None else "",
        "Sensitivity": f"{float(metrics.get('Sensitivity')):.3f}" if metrics.get("Sensitivity") is not None else "",
        "Specificity": f"{float(metrics.get('Specificity')):.3f}" if metrics.get("Specificity") is not None else "",
        "best_epoch": "",
        "checkpoint": "",
        "status": "ok",
        "error": "",
    }


def _run_one_safe(
    *,
    ds: DatasetSpec,
    split_spec: SplitSpec,
    seed: int,
    train_size: int,
    output_root: Path,
    llm_cfg: LLMConfig,
) -> dict:
    try:
        return _run_one(
            ds=ds,
            split_spec=split_spec,
            seed=seed,
            train_size=train_size,
            output_root=output_root,
            llm_cfg=llm_cfg,
        )
    except Exception as e:
        return {
            "模型": "HL",
            "数据集": ds.name,
            "训练集数据量": str(train_size),
            "ACC": "",
            "F1": "",
            "Sensitivity": "",
            "Specificity": "",
            "best_epoch": "",
            "checkpoint": "",
            "status": "error",
            "error": str(e) + "\n" + traceback.format_exc(),
        }


def run_contrast1_balance_hl(seed: int = 42, workers: int = 1, output_root: str = "") -> Path:
    out_dir = script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_root_path = Path(output_root) if str(output_root).strip() else (out_dir / f"outputs_balance_hl_seed{int(seed)}")
    output_root_path.mkdir(parents=True, exist_ok=True)

    datasets = [
        DatasetSpec("UKB", repo_root / "data" / "UKB.csv", "label"),
        DatasetSpec("YHD", repo_root / "data" / "YHD_bicarbonate.csv", "hospital_expire_flag"),
    ]
    split_spec = SplitSpec(val_total=1000, test_total=1000)
    train_sizes = [3000, 1000, 500, 100, 50, 10]

    llm_cfg = LLMConfig(
        base_url=os.getenv("CONTRAST1_HL_BASE_URL", "https://api.deepseek.com/v1"),
        api_key_env=os.getenv("CONTRAST1_HL_KEY_ENV", "DEEPSEEK_API_KEY"),
        model_name=os.getenv("CONTRAST1_HL_MODEL", "deepseek-v4-pro"),
        temperature=float(os.getenv("CONTRAST1_HL_TEMPERATURE", "0.0")),
    )

    tasks: list[tuple[DatasetSpec, int]] = [(ds, ts) for ds in datasets for ts in train_sizes]
    rows: list[dict] = []
    if int(workers) > 1:
        if int(workers) > (os.cpu_count() or 1) * 4:
            print("[contrast1_balance_hl] warning: workers is very high; you may hit API rate limits.", flush=True)
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = [
                ex.submit(
                    _run_one_safe,
                    ds=ds,
                    split_spec=split_spec,
                    seed=int(seed),
                    train_size=int(ts),
                    output_root=output_root_path,
                    llm_cfg=llm_cfg,
                )
                for ds, ts in tasks
            ]
            for fut in as_completed(futs):
                rows.append(fut.result())
    else:
        for ds, ts in tasks:
            rows.append(
                _run_one_safe(
                    ds=ds,
                    split_spec=split_spec,
                    seed=int(seed),
                    train_size=int(ts),
                    output_root=output_root_path,
                    llm_cfg=llm_cfg,
                )
            )

    dataset_order = {"UKB": 0, "YHD": 1}

    def key_fn(r: dict) -> tuple:
        ds = str(r.get("数据集", ""))
        try:
            ts = int(str(r.get("训练集数据量", "")))
        except Exception:
            ts = 1_000_000_000
        return (dataset_order.get(ds, 99), ts)

    rows.sort(key=key_fn)

    out_path = out_dir / "contrast1_balance_hl.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "模型",
                "数据集",
                "训练集数据量",
                "ACC",
                "F1",
                "Sensitivity",
                "Specificity",
                "best_epoch",
                "checkpoint",
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
    p.add_argument("--output-root", type=str, default="")
    args = p.parse_args()
    out = run_contrast1_balance_hl(seed=int(args.seed), workers=int(args.workers), output_root=str(args.output_root))
    print(f"contrast1_balance_hl_csv={out}", flush=True)


if __name__ == "__main__":
    main()
