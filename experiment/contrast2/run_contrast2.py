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
OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs_rerun" / "contrast2"
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
    "模型", "数据集", "训练集数据量", "训练集正负比", "ACC", "F1", "Sensitivity", "Specificity",
    "TP", "FP", "FN", "TN", "status", "error",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    label_col: str
    holdout_total: int


@dataclass(frozen=True)
class RatioSpec:
    pos: int
    neg: int

    @property
    def name(self) -> str:
        return f"{self.pos}:{self.neg}"


DATASETS = (
    DatasetSpec("UKB", REPO_ROOT / "data" / "UKB.csv", "label", 1000),
    DatasetSpec("YHD", REPO_ROOT / "data" / "YHD_bicarbonate.csv", "hospital_expire_flag", 1000),
)
TRAIN_TOTALS = (1000, 3000)
RATIOS = (
    RatioSpec(1, 1), RatioSpec(1, 2), RatioSpec(2, 1), RatioSpec(1, 5), RatioSpec(5, 1),
    RatioSpec(1, 10), RatioSpec(10, 1), RatioSpec(1, 50), RatioSpec(50, 1),
)


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
    positive = frame.loc[frame[spec.label_col] == 1]
    negative = frame.loc[frame[spec.label_col] == 0]
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


def _counts_from_ratio(total: int, ratio: RatioSpec) -> tuple[int, int]:
    positive = round(total * ratio.pos / (ratio.pos + ratio.neg))
    positive = max(1, min(int(positive), int(total) - 1))
    return positive, int(total) - positive


def _sample_train(
    train_pool: pd.DataFrame,
    spec: DatasetSpec,
    total: int,
    ratio: RatioSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    positive_target, negative_target = _counts_from_ratio(total, ratio)
    positive = train_pool.loc[train_pool[spec.label_col] == 1]
    negative = train_pool.loc[train_pool[spec.label_col] == 0]
    if positive.empty or negative.empty:
        raise ValueError(f"{spec.name}: training pool is missing one class")
    pos_replace = len(positive) < positive_target
    neg_replace = len(negative) < negative_target
    sampled = pd.concat(
        [
            positive.sample(positive_target, replace=pos_replace, random_state=seed + 11),
            negative.sample(negative_target, replace=neg_replace, random_state=seed + 23),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed + 97).reset_index(drop=True)
    actual_pos = int((sampled[spec.label_col] == 1).sum())
    actual_neg = int((sampled[spec.label_col] == 0).sum())
    if (actual_pos, actual_neg) != (positive_target, negative_target):
        raise AssertionError(f"Class count mismatch: target={(positive_target, negative_target)}, actual={(actual_pos, actual_neg)}")
    return sampled, {
        "requested_rows": int(total),
        "ratio": ratio.name,
        "positive_target": positive_target,
        "negative_target": negative_target,
        "positive_replace": bool(pos_replace),
        "negative_replace": bool(neg_replace),
        "unique_source_rows": int(sampled[ROW_ID_COL].nunique()),
        "source_row_hash": _row_hash(sampled),
    }


def _row_hash(frame: pd.DataFrame) -> str:
    values = ",".join(str(int(x)) for x in frame[ROW_ID_COL].tolist())
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[ROW_ID_COL]).copy()


def _metric_text(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else f"{float(value):.3f}"


def _task_key(row: dict) -> tuple[str, str, int, str]:
    return str(row["模型"]), str(row["数据集"]), int(row["训练集数据量"]), str(row["训练集正负比"])


def _read_existing(path: Path) -> dict[tuple[str, str, int, str], dict]:
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


def _ordered_rows(rows_by_key: dict[tuple[str, str, int, str], dict]) -> list[dict]:
    model_order = {name: idx for idx, name in enumerate(ALL_MODEL_NAMES)}
    dataset_order = {spec.name: idx for idx, spec in enumerate(DATASETS)}
    ratio_order = {ratio.name: idx for idx, ratio in enumerate(RATIOS)}
    return sorted(
        rows_by_key.values(),
        key=lambda row: (
            model_order.get(str(row["模型"]), 999),
            dataset_order.get(str(row["数据集"]), 999),
            TRAIN_TOTALS.index(int(row["训练集数据量"])) if int(row["训练集数据量"]) in TRAIN_TOTALS else 999,
            ratio_order.get(str(row["训练集正负比"]), 999),
        ),
    )


def run_contrast2(
    seed: int,
    *,
    models: tuple[str, ...] = ALL_MODEL_NAMES,
    train_totals: tuple[int, ...] = TRAIN_TOTALS,
    ratios: tuple[RatioSpec, ...] = RATIOS,
    datasets: tuple[str, ...] = ("UKB", "YHD"),
    resume: bool = False,
    retry_errors: bool = False,
    rerun_existing: bool = False,
) -> Path:
    """Run one seed of contrast2 and atomically update its result CSV."""

    out_path = SCRIPT_DIR / f"contrast2_rerun_seed{seed}.csv"
    rows_by_key = _read_existing(out_path) if resume else {}
    selected_specs = tuple(spec for spec in DATASETS if spec.name in datasets)
    total_tasks = len(selected_specs) * len(train_totals) * len(ratios) * len(models)
    completed = 0

    for spec in selected_specs:
        raw = _load_dataset(spec)
        train_pool, val_raw, test_raw = _split_balanced(raw, spec, seed)
        val_df = _model_frame(val_raw)
        test_df = _model_frame(test_raw)
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
        for total in train_totals:
            for ratio in ratios:
                sample_seed = seed + int(total) + ratio.pos * 1000 + ratio.neg
                train_raw, sample_meta = _sample_train(train_pool, spec, int(total), ratio, sample_seed)
                train_df = _model_frame(train_raw)
                for model_name in models:
                    completed += 1
                    key = (model_name, spec.name, int(total), ratio.name)
                    previous = rows_by_key.get(key)
                    if (
                        previous is not None
                        and not rerun_existing
                        and (not retry_errors or previous.get("status") in {"ok", "continued"})
                    ):
                        print(f"[{completed}/{total_tasks}] skip {key} status={previous.get('status')}", flush=True)
                        continue

                    artifact_dir = OUTPUT_ROOT / f"seed{seed}" / spec.name / f"train{total}" / f"ratio{ratio.pos}_{ratio.neg}" / model_name
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"[{completed}/{total_tasks}] fit seed={seed} dataset={spec.name} "
                        f"train={total} ratio={ratio.name} model={model_name}",
                        flush=True,
                    )
                    started = time.perf_counter()
                    try:
                        fitted = fit_model(
                            model_name, train_df, val_df, spec.label_col, seed,
                            checkpoint_dir=artifact_dir / "checkpoints",
                        )
                        metrics = evaluate_model(fitted, test_df, spec.label_col)
                        predictions = predict_model(fitted, test_df.drop(columns=[spec.label_col]))
                        probabilities = predict_positive_probability(fitted, test_df.drop(columns=[spec.label_col]))
                        if sum(int(metrics[key]) for key in ("TP", "FP", "FN", "TN")) != len(test_df):
                            raise AssertionError("Confusion counts do not sum to test size")
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
                                "train_total": int(total),
                                "ratio": ratio.name,
                                "train_sampling": sample_meta,
                                "feature_columns": [col for col in train_df.columns if col != spec.label_col],
                                "training_summary": fitted.training_summary,
                                "model_path": str(model_path),
                                "elapsed_seconds_total": time.perf_counter() - started,
                            },
                        )
                        row = {
                            "模型": model_name, "数据集": spec.name, "训练集数据量": str(total), "训练集正负比": ratio.name,
                            "ACC": _metric_text(metrics, "ACC"), "F1": _metric_text(metrics, "F1"),
                            "Sensitivity": _metric_text(metrics, "Sensitivity"), "Specificity": _metric_text(metrics, "Specificity"),
                            "TP": str(metrics["TP"]), "FP": str(metrics["FP"]), "FN": str(metrics["FN"]), "TN": str(metrics["TN"]),
                            "status": "ok", "error": "",
                        }
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        (artifact_dir / "error.txt").write_text(error, encoding="utf-8")
                        row = {
                            "模型": model_name, "数据集": spec.name, "训练集数据量": str(total), "训练集正负比": ratio.name,
                            "ACC": "", "F1": "", "Sensitivity": "", "Specificity": "",
                            "TP": "", "FP": "", "FN": "", "TN": "", "status": "error", "error": error,
                        }
                        print(f"[{completed}/{total_tasks}] error {key}: {error}", flush=True)
                    rows_by_key[key] = row
                    _write_csv_atomic(out_path, _ordered_rows(rows_by_key))

    _write_csv_atomic(out_path, _ordered_rows(rows_by_key))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run class-ratio contrast experiment.")
    parser.add_argument("--models", default="all", help="all or comma-separated canonical model names")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(EXPERIMENT_SEEDS))
    parser.add_argument("--seed", type=int, help="Backward-compatible single-seed override")
    parser.add_argument("--train-totals", nargs="+", type=int, default=list(TRAIN_TOTALS))
    parser.add_argument("--ratios", nargs="+", default=[ratio.name for ratio in RATIOS])
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
    ratio_map = {ratio.name: ratio for ratio in RATIOS}
    unknown_ratios = sorted(set(args.ratios) - set(ratio_map))
    if unknown_ratios:
        raise ValueError(f"Unknown ratios: {unknown_ratios}")
    selected_ratios = tuple(ratio_map[name] for name in args.ratios)
    for seed in seeds:
        path = run_contrast2(
            int(seed), models=models, train_totals=tuple(args.train_totals), ratios=selected_ratios,
            datasets=tuple(args.datasets), resume=bool(args.resume), retry_errors=bool(args.retry_errors),
            rerun_existing=bool(args.rerun_existing),
        )
        print(f"contrast2_rerun_csv={path}", flush=True)


if __name__ == "__main__":
    main()
