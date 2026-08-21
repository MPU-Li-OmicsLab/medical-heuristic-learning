from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs_rerun" / "contrast1"
ROW_ID_COL = "__source_row_id__"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.modeling import (
    ALL_MODEL_NAMES,
    evaluate_model,
    fit_model,
    predict_model,
    predict_positive_probability,
    save_fitted_model,
)
from experiment.modeling.config import EXPERIMENT_SEEDS, parse_model_names


FIELDNAMES = [
    "模型", "数据集", "训练集数据量", "ACC", "F1", "Sensitivity", "Specificity",
    "best_epoch", "checkpoint", "status", "error",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    label_col: str
    holdout_total: int


DATASETS = (
    DatasetSpec("UKB", REPO_ROOT / "data" / "UKB.csv", "label", 1000),
    DatasetSpec("YHD", REPO_ROOT / "data" / "YHD_bicarbonate.csv", "hospital_expire_flag", 1000),
)
TRAIN_SIZES = (3000, 1000, 500, 100, 50, 10)


def _load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    if not spec.csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {spec.csv_path}")
    frame = pd.read_csv(spec.csv_path)
    if spec.label_col not in frame.columns:
        raise ValueError(f"{spec.name}: label column not found: {spec.label_col}")
    if ROW_ID_COL in frame.columns:
        raise ValueError(f"Reserved column already exists: {ROW_ID_COL}")
    frame = frame.copy()
    frame[spec.label_col] = frame[spec.label_col].astype(int)
    frame[ROW_ID_COL] = np.arange(len(frame), dtype=int)
    return frame


def _split_balanced(frame: pd.DataFrame, spec: DatasetSpec, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    each = spec.holdout_total // 2
    y = frame[spec.label_col]
    positive = frame.loc[y == 1]
    negative = frame.loc[y == 0]
    if len(positive) < 2 * each or len(negative) < 2 * each:
        raise ValueError(
            f"{spec.name}: balanced val/test need {2 * each} rows per class, "
            f"got positive={len(positive)}, negative={len(negative)}"
        )
    rng = np.random.default_rng(seed)
    pos_ids = rng.permutation(positive.index.to_numpy())
    neg_ids = rng.permutation(negative.index.to_numpy())
    test_ids = np.concatenate([pos_ids[:each], neg_ids[:each]])
    val_ids = np.concatenate([pos_ids[each : 2 * each], neg_ids[each : 2 * each]])
    used = np.concatenate([test_ids, val_ids])
    test_df = frame.loc[test_ids].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    val_df = frame.loc[val_ids].sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    train_pool = frame.loc[~frame.index.isin(used)].reset_index(drop=True)
    return train_pool, val_df, test_df


def _sample_train(train_pool: pd.DataFrame, spec: DatasetSpec, train_size: int, seed: int) -> tuple[pd.DataFrame, dict]:
    if train_size <= 0 or train_size % 2:
        raise ValueError("contrast1 train_size must be a positive even integer")
    each = train_size // 2
    positive = train_pool.loc[train_pool[spec.label_col] == 1]
    negative = train_pool.loc[train_pool[spec.label_col] == 0]
    if positive.empty or negative.empty:
        raise ValueError(f"{spec.name}: training pool is missing one class")
    pos_replace = len(positive) < each
    neg_replace = len(negative) < each
    sampled = pd.concat(
        [
            positive.sample(each, replace=pos_replace, random_state=seed + 11),
            negative.sample(each, replace=neg_replace, random_state=seed + 23),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed + 97).reset_index(drop=True)
    meta = {
        "requested_rows": int(train_size),
        "positive": int((sampled[spec.label_col] == 1).sum()),
        "negative": int((sampled[spec.label_col] == 0).sum()),
        "positive_replace": bool(pos_replace),
        "negative_replace": bool(neg_replace),
        "unique_source_rows": int(sampled[ROW_ID_COL].nunique()),
        "source_row_hash": _row_hash(sampled),
    }
    return sampled, meta


def _row_hash(frame: pd.DataFrame) -> str:
    values = ",".join(str(int(x)) for x in frame[ROW_ID_COL].tolist())
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[ROW_ID_COL]).copy()


def _metric_text(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else f"{float(value):.3f}"


def _task_key(row: dict) -> tuple[str, str, int]:
    return str(row["模型"]), str(row["数据集"]), int(row["训练集数据量"])


def _read_existing(path: Path) -> dict[tuple[str, str, int], dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {_task_key(row): row for row in csv.DictReader(handle)}


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _ordered_rows(rows_by_key: dict[tuple[str, str, int], dict]) -> list[dict]:
    model_order = {name: idx for idx, name in enumerate(ALL_MODEL_NAMES)}
    dataset_order = {spec.name: idx for idx, spec in enumerate(DATASETS)}
    return sorted(
        rows_by_key.values(),
        key=lambda row: (
            model_order.get(str(row["模型"]), 999),
            dataset_order.get(str(row["数据集"]), 999),
            TRAIN_SIZES.index(int(row["训练集数据量"])) if int(row["训练集数据量"]) in TRAIN_SIZES else 999,
        ),
    )


def run_contrast_balance(
    seed: int,
    *,
    models: tuple[str, ...] = ALL_MODEL_NAMES,
    train_sizes: tuple[int, ...] = TRAIN_SIZES,
    datasets: tuple[str, ...] = ("UKB", "YHD"),
    resume: bool = False,
    retry_errors: bool = False,
    rerun_existing: bool = False,
) -> Path:
    """Run one seed of contrast1 and atomically update its result CSV."""

    out_path = SCRIPT_DIR / f"contrast1_balance_rerun_seed{seed}.csv"
    rows_by_key = _read_existing(out_path) if resume else {}
    selected_specs = tuple(spec for spec in DATASETS if spec.name in datasets)
    total = len(selected_specs) * len(train_sizes) * len(models)
    completed = 0

    for spec in selected_specs:
        raw = _load_dataset(spec)
        train_pool, val_raw, test_raw = _split_balanced(raw, spec, seed)
        split_manifest = {
            "dataset": spec.name,
            "seed": int(seed),
            "label_col": spec.label_col,
            "validation_rows": len(val_raw),
            "test_rows": len(test_raw),
            "validation_source_row_hash": _row_hash(val_raw),
            "test_source_row_hash": _row_hash(test_raw),
            "validation_class_counts": val_raw[spec.label_col].value_counts().sort_index().to_dict(),
            "test_class_counts": test_raw[spec.label_col].value_counts().sort_index().to_dict(),
        }
        val_df = _model_frame(val_raw)
        test_df = _model_frame(test_raw)
        for train_size in train_sizes:
            train_raw, sample_meta = _sample_train(train_pool, spec, int(train_size), seed + int(train_size))
            train_df = _model_frame(train_raw)
            for model_name in models:
                completed += 1
                key = (model_name, spec.name, int(train_size))
                previous = rows_by_key.get(key)
                if (
                    previous is not None
                    and not rerun_existing
                    and (not retry_errors or previous.get("status") in {"ok", "continued"})
                ):
                    print(f"[{completed}/{total}] skip {key} status={previous.get('status')}", flush=True)
                    continue

                artifact_dir = OUTPUT_ROOT / f"seed{seed}" / spec.name / f"train{train_size}" / model_name
                artifact_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{completed}/{total}] fit seed={seed} dataset={spec.name} train={train_size} model={model_name}", flush=True)
                started = time.perf_counter()
                try:
                    fitted = fit_model(
                        model_name, train_df, val_df, spec.label_col, seed,
                        checkpoint_dir=artifact_dir / "checkpoints",
                    )
                    metrics = evaluate_model(fitted, test_df, spec.label_col)
                    predictions = predict_model(fitted, test_df.drop(columns=[spec.label_col]))
                    probabilities = predict_positive_probability(fitted, test_df.drop(columns=[spec.label_col]))
                    model_path = save_fitted_model(fitted, artifact_dir)
                    pd.DataFrame(
                        {
                            ROW_ID_COL: test_raw[ROW_ID_COL].astype(int),
                            "y_true": test_df[spec.label_col].astype(int),
                            "y_pred": predictions.astype(int),
                            "positive_probability": probabilities.astype(float),
                        }
                    ).to_csv(artifact_dir / "predictions.csv", index=False)
                    _write_json(artifact_dir / "metrics.json", metrics)
                    _write_json(
                        artifact_dir / "split_manifest.json",
                        {**split_manifest, "train_sampling": sample_meta},
                    )
                    _write_json(
                        artifact_dir / "run_manifest.json",
                        {
                            **split_manifest,
                            "model": model_name,
                            "train_size": int(train_size),
                            "train_sampling": sample_meta,
                            "feature_columns": [col for col in train_df.columns if col != spec.label_col],
                            "training_summary": fitted.training_summary,
                            "model_path": str(model_path),
                            "elapsed_seconds_total": time.perf_counter() - started,
                        },
                    )
                    row = {
                        "模型": model_name, "数据集": spec.name, "训练集数据量": str(train_size),
                        "ACC": _metric_text(metrics, "ACC"), "F1": _metric_text(metrics, "F1"),
                        "Sensitivity": _metric_text(metrics, "Sensitivity"),
                        "Specificity": _metric_text(metrics, "Specificity"),
                        "best_epoch": str(fitted.training_summary.get("best_epoch", "")),
                        "checkpoint": str(model_path) if fitted.family == "deeptab" else "",
                        "status": "ok", "error": "",
                    }
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    (artifact_dir / "error.txt").write_text(error, encoding="utf-8")
                    row = {
                        "模型": model_name, "数据集": spec.name, "训练集数据量": str(train_size),
                        "ACC": "", "F1": "", "Sensitivity": "", "Specificity": "",
                        "best_epoch": "", "checkpoint": "", "status": "error", "error": error,
                    }
                    print(f"[{completed}/{total}] error {key}: {error}", flush=True)
                rows_by_key[key] = row
                _write_csv_atomic(out_path, _ordered_rows(rows_by_key))

    _write_csv_atomic(out_path, _ordered_rows(rows_by_key))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run balanced training-size contrast experiment.")
    parser.add_argument("--models", default="all", help="all or comma-separated canonical model names")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(EXPERIMENT_SEEDS))
    parser.add_argument("--seed", type=int, help="Backward-compatible single-seed override")
    parser.add_argument("--train-sizes", nargs="+", type=int, default=list(TRAIN_SIZES))
    parser.add_argument("--datasets", nargs="+", choices=[spec.name for spec in DATASETS], default=[spec.name for spec in DATASETS])
    parser.add_argument("--workers", type=int, default=1, help="Accepted for compatibility; model runs remain sequential")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun selected tasks even when resume loaded an existing successful row.",
    )
    args = parser.parse_args()
    seeds = (args.seed,) if args.seed is not None else tuple(args.seeds)
    models = parse_model_names(args.models)
    for seed in seeds:
        path = run_contrast_balance(
            int(seed), models=models, train_sizes=tuple(args.train_sizes), datasets=tuple(args.datasets),
            resume=bool(args.resume), retry_errors=bool(args.retry_errors),
            rerun_existing=bool(args.rerun_existing),
        )
        print(f"contrast1_balance_rerun_csv={path}", flush=True)


if __name__ == "__main__":
    main()
