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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hl.config import LLMConfig, RunConfig
from hl.metrics import compute_metrics
from hl.orchestrator import run_heuristic_learning
from hl.utils.io import write_json, write_text


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
            f"Not enough samples for balanced splits. Need pos>= {need_each}, neg>= {need_each}, "
            f"got pos={len(pos_df)}, neg={len(neg_df)}"
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

    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    used_idx = set(pos_test.tolist()) | set(neg_test.tolist()) | set(pos_val.tolist()) | set(neg_val.tolist())
    train_pool = df.loc[~df.index.isin(list(used_idx))].copy().reset_index(drop=True)

    return train_pool, val_df, test_df


def _sample_train_random(train_pool: pd.DataFrame, train_size: int, seed: int) -> pd.DataFrame:
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if len(train_pool) < train_size:
        raise ValueError(f"Not enough remaining samples for train_size={train_size}. pool_size={len(train_pool)}")
    return train_pool.sample(n=train_size, replace=False, random_state=seed).reset_index(drop=True)


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


def _summarize_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _run_one(
    *,
    dataset_name: str,
    csv_path: Path,
    label_col: str,
    seed: int,
    run_univariate_probe: bool,
    run_knowledge_probe: bool,
    train_size: int,
    split_spec: SplitSpec,
    base_output_dir: Path,
) -> Path:
    df = _load_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label_col={label_col} not found in dataset columns")
    df = df.copy()
    df[label_col] = df[label_col].astype(int)

    train_pool, val_df, test_df = _split_val_test_balanced(df=df, label_col=label_col, spec=split_spec, seed=seed)
    train_df = _sample_train_random(train_pool=train_pool, train_size=train_size, seed=seed + train_size)

    cfg_name = f"U{int(run_univariate_probe)}_K{int(run_knowledge_probe)}"
    out_dir = base_output_dir / dataset_name / cfg_name / f"train{train_size}" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ablation] dataset={dataset_name} ablation={cfg_name} train={train_size} out_dir={out_dir}", flush=True)

    run_cfg = RunConfig(
        output_dir=out_dir,
        run_univariate_probe=run_univariate_probe,
        run_knowledge_probe=run_knowledge_probe,
        run_v0_generation=True,
        run_iterations=True,
        task_description=(
            f"Dataset={dataset_name}. Ablation={cfg_name}. "
            "Binary classification on a clinical/tabular dataset with balanced splits."
        ),
    )
    llm_cfg = LLMConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model_name="deepseek-v4-pro",
        temperature=0.0,
    )

    run_heuristic_learning(train_df=train_df, test_df=val_df, label_col=label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)

    model_path = out_dir / "final_heuristic_model.py"
    predict_fn = _load_predict_fn(model_path)
    y_true = test_df[label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, test_df, label_col=label_col)

    code_all = model_path.read_text(encoding="utf-8")
    final_version = ""
    for line in code_all.splitlines():
        if line.startswith("FINAL_VERSION"):
            try:
                final_version = json.loads(line.split("=", 1)[1].strip())
            except Exception:
                final_version = ""
            break

    metrics = compute_metrics(y_true, y_pred)
    summary = {
        "dataset": dataset_name,
        "csv_path": str(csv_path),
        "label_col": label_col,
        "seed": seed,
        "split_spec": {
            "val_total": split_spec.val_total,
            "test_total": split_spec.test_total,
            "train_total": train_size,
        },
        "ablation": cfg_name,
        "final_version": final_version,
        "heldout_test_size": int(len(test_df)),
        "heldout_test_counts": _summarize_counts(y_true, y_pred),
        "heldout_test_metrics": metrics,
    }
    write_json(out_dir / "heldout_test_summary.json", summary)

    write_text(
        out_dir / "heldout_test_summary.txt",
        "\n".join(
            [
                f"dataset={dataset_name}",
                f"ablation={cfg_name}",
                f"train_size={train_size}",
                f"final_version={final_version}",
                f"heldout_test_counts={summary['heldout_test_counts']}",
                f"heldout_test_metrics={metrics}",
            ]
        )
        + "\n",
    )
    print(
        f"[ablation] done dataset={dataset_name} ablation={cfg_name} train={train_size} final_version={final_version} out_dir={out_dir}",
        flush=True,
    )
    return out_dir


def _run_one_safe(
    dataset_name: str,
    csv_path: Path,
    label_col: str,
    seed: int,
    run_univariate_probe: bool,
    run_knowledge_probe: bool,
    train_size: int,
    split_spec: SplitSpec,
    base_output_dir: Path,
) -> dict:
    ablation = f"U{int(run_univariate_probe)}_K{int(run_knowledge_probe)}"
    try:
        out_dir = _run_one(
            dataset_name=dataset_name,
            csv_path=csv_path,
            label_col=label_col,
            seed=seed,
            run_univariate_probe=run_univariate_probe,
            run_knowledge_probe=run_knowledge_probe,
            train_size=train_size,
            split_spec=split_spec,
            base_output_dir=base_output_dir,
        )
        return {"dataset": dataset_name, "ablation": ablation, "train_size": train_size, "out_dir": str(out_dir), "status": "ok"}
    except Exception as e:
        return {
            "dataset": dataset_name,
            "ablation": ablation,
            "train_size": train_size,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    base_output_dir = Path("./ablation/outputs")
    seed = 42
    split_spec = SplitSpec(val_total=1000, test_total=1000)

    datasets = [
        ("YHD", Path("./data/YHD_bicarbonate.csv"), "hospital_expire_flag"),
        ("UKB", Path("./data/UKB.csv"), "label"),
    ]
    ablations = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    train_sizes = [3000, 1000, 100, 10]

    tasks: list[tuple] = []
    for dataset_name, csv_path, label_col in datasets:
        for run_univariate_probe, run_knowledge_probe in ablations:
            for train_size in train_sizes:
                tasks.append(
                    (
                        dataset_name,
                        csv_path,
                        label_col,
                        seed,
                        run_univariate_probe,
                        run_knowledge_probe,
                        train_size,
                        split_spec,
                        base_output_dir,
                    )
                )

    workers = int(args.workers)
    if workers <= 1:
        results = [_run_one_safe(*t) for t in tasks]
    else:
        if workers > (os.cpu_count() or 1) * 4:
            print(f"[ablation] warning: workers={workers} is very high; you may hit API rate limits.", flush=True)
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_run_one_safe, *t): t for t in tasks}
            for fut in as_completed(fut_map):
                r = fut.result()
                results.append(r)
                if r.get("status") == "ok":
                    print(
                        f"[ablation] finished dataset={r['dataset']} ablation={r['ablation']} train={r['train_size']} out_dir={r['out_dir']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[ablation] failed dataset={r['dataset']} ablation={r['ablation']} train={r['train_size']} error={r.get('error','')}",
                        flush=True,
                    )

    index_path = base_output_dir / f"index_{_timestamp()}.json"
    write_json(index_path, results)
    print(f"[ablation] index_written={index_path}", flush=True)


if __name__ == "__main__":
    main()
