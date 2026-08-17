from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs_rerun" / "continuous_learning"
RESULTS_PATH = SCRIPT_DIR / "continuous_baselines_rerun_results.csv"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from continuous_learning_experiment_common import (
    DEFAULT_SEEDS,
    MIMIC_CSV_PATH,
    MIMIC_LABEL_COL,
    ModelStageResult,
    StageDataBundle,
    build_stage1_drift,
    build_stage2_drift_template,
    get_default_experiment_settings,
    prepare_two_stage_data_bundles,
    stage_bundle_manifest,
    write_results_csv,
)
from experiment.modeling import (
    ALL_MODEL_NAMES,
    evaluate_model,
    fit_model,
    predict_model,
    predict_positive_probability,
    save_fitted_model,
)
from experiment.modeling.config import NEW_MODEL_NAMES, parse_model_names


def _metric_text(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else f"{float(value):.3f}"


def _result(
    *,
    model: str,
    bundle: StageDataBundle,
    metrics: dict | None,
    status: str,
    error: str,
    out_dir: Path,
) -> ModelStageResult:
    return ModelStageResult(
        model=model,
        dataset=bundle.dataset,
        seed=bundle.seed,
        stage=bundle.stage,
        acc=_metric_text(metrics or {}, "ACC"),
        f1=_metric_text(metrics or {}, "F1"),
        sensitivity=_metric_text(metrics or {}, "Sensitivity"),
        specificity=_metric_text(metrics or {}, "Specificity"),
        status=status,
        error=error,
        out_dir=str(out_dir),
    )


def _result_key(result: ModelStageResult) -> tuple[int, str, str]:
    return int(result.seed), result.model, result.stage


def _read_existing(path: Path) -> dict[tuple[int, str, str], ModelStageResult]:
    if not path.exists():
        return {}
    rows: dict[tuple[int, str, str], ModelStageResult] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result = ModelStageResult(
                model=row["模型"],
                dataset=row["数据集"],
                seed=int(row["seed"]),
                stage=row["阶段"],
                acc=row["ACC"],
                f1=row["F1"],
                sensitivity=row["Sensitivity"],
                specificity=row["Specificity"],
                status=row["status"],
                error=row["error"],
                out_dir=row["out_dir"],
            )
            rows[_result_key(result)] = result
    return rows


def _ordered_results(rows: dict[tuple[int, str, str], ModelStageResult]) -> list[ModelStageResult]:
    model_order = {name: idx for idx, name in enumerate(ALL_MODEL_NAMES)}
    return sorted(
        rows.values(),
        key=lambda result: (
            int(result.seed),
            model_order.get(result.model, 999),
            0 if result.stage.startswith("stage1") else 1,
        ),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _validate_stage_data(stage1: StageDataBundle, stage2: StageDataBundle) -> dict:
    if stage1.dataset != stage2.dataset or stage1.seed != stage2.seed or stage1.label_col != stage2.label_col:
        raise ValueError("Stage1 and Stage2 identity fields do not match")
    feature1 = [col for col in stage1.train_df.columns if col != stage1.label_col]
    feature2 = [col for col in stage2.train_df.columns if col != stage2.label_col]
    if "SIRS" not in feature1 or "SOFA" in feature1:
        raise ValueError("Stage1 must contain SIRS and exclude SOFA")
    if "SOFA" not in feature2 or "SIRS" in feature2:
        raise ValueError("Stage2 must contain SOFA and exclude SIRS")
    expected = (1000, 500, 800, 40, 500, 800)
    actual = (
        len(stage1.train_df), len(stage1.val_df), len(stage1.test_df),
        len(stage2.train_df), len(stage2.val_df), len(stage2.test_df),
    )
    if actual != expected:
        raise ValueError(f"Unexpected continuous stage sizes: expected={expected}, actual={actual}")
    row_sets = {
        "stage1_train": set(stage1.train_sampling_meta["train_source_row_ids"]),
        "stage1_val": set(stage1.split_meta["val_source_row_ids"]),
        "stage1_test": set(stage1.split_meta["test_source_row_ids"]),
        "stage2_train": set(stage2.train_sampling_meta["train_source_row_ids"]),
        "stage2_val": set(stage2.split_meta["val_source_row_ids"]),
        "stage2_test": set(stage2.split_meta["test_source_row_ids"]),
    }
    keys = list(row_sets)
    for idx, left in enumerate(keys):
        for right in keys[idx + 1 :]:
            overlap = row_sets[left] & row_sets[right]
            if overlap:
                raise ValueError(f"Continuous splits overlap: {left}/{right}: {sorted(overlap)[:10]}")
    return {
        "sizes": actual,
        "all_pairwise_source_row_intersections_empty": True,
        "stage1_feature_cols": feature1,
        "stage2_feature_cols": feature2,
    }


def _fill_value(series: pd.Series) -> float | str:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        value = numeric.median()
        return 0.0 if pd.isna(value) else float(value)
    mode = series.dropna().mode()
    return "" if mode.empty else str(mode.iloc[0])


def _with_union_schema(
    stage1: StageDataBundle,
    stage2: StageDataBundle,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict]:
    """Align existing baselines by feature name; never treat SOFA as the old SIRS column."""

    label = stage1.label_col
    features1 = [col for col in stage1.train_df.columns if col != label]
    features2 = [col for col in stage2.train_df.columns if col != label]
    union = features1 + [col for col in features2 if col not in features1]
    sirs_fill = _fill_value(stage1.train_df["SIRS"])

    def align(frame: pd.DataFrame, stage: int) -> pd.DataFrame:
        out = frame.copy()
        if stage == 1 and "SOFA" not in out:
            out["SOFA"] = 0.0
        if stage == 2 and "SIRS" not in out:
            out["SIRS"] = sirs_fill
        return out[union + [label]].copy()

    aligned1 = tuple(align(frame, 1) for frame in (stage1.train_df, stage1.val_df, stage1.test_df))
    aligned2 = tuple(align(frame, 2) for frame in (stage2.train_df, stage2.val_df, stage2.test_df))
    return aligned1, aligned2, {
        "strategy": "feature_name_union",
        "feature_columns": union,
        "stage1_added_SOFA_constant": 0.0,
        "stage2_added_SIRS_train_median": sirs_fill,
    }


def _stage1_compatible_views(
    stage1: StageDataBundle,
    stage2: StageDataBundle,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict]:
    """Build Stage1 views of Stage2 without reading unavailable true SIRS values."""

    label = stage1.label_col
    features = [col for col in stage1.train_df.columns if col != label]
    fill_values = {col: _fill_value(stage1.train_df[col]) for col in features}

    def compatible(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for col in features:
            out[col] = frame[col] if col in frame.columns else fill_values[col]
        out[label] = frame[label].astype(int)
        return out[features + [label]]

    return tuple(compatible(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)), {
        "strategy": "stage1_shared_features_with_train_fill",
        "stage1_feature_columns": features,
        "fill_values": fill_values,
        "ignored_stage2_feature": "SOFA",
        "true_stage2_SIRS_accessed": False,
    }


def _save_predictions(
    path: Path,
    bundle: StageDataBundle,
    fitted,
    frame: pd.DataFrame,
) -> None:
    label = bundle.label_col
    ids = bundle.split_meta.get("test_source_row_ids_ordered") or list(range(len(frame)))
    predictions = predict_model(fitted, frame.drop(columns=[label]))
    probabilities = predict_positive_probability(fitted, frame.drop(columns=[label]))
    pd.DataFrame(
        {
            "__continuous_row_id__": ids,
            "y_true": frame[label].astype(int).to_numpy(),
            "y_pred": predictions.astype(int),
            "positive_probability": probabilities.astype(float),
        }
    ).to_csv(path, index=False)


def _fit_existing_model(
    model_name: str,
    stage1: StageDataBundle,
    stage2: StageDataBundle,
    model_root: Path,
) -> tuple[ModelStageResult, ModelStageResult]:
    label = stage1.label_col
    (train1, val1, test1), (train2, val2, test2), alignment = _with_union_schema(stage1, stage2)
    dir1 = model_root / stage1.stage
    dir2 = model_root / stage2.stage
    fitted1 = fit_model(
        model_name, train1, val1, label, stage1.seed,
        checkpoint_dir=dir1 / "checkpoints", variant="continuous", stage=1,
    )
    metrics1 = evaluate_model(fitted1, test1, label)
    path1 = save_fitted_model(fitted1, dir1)
    _save_predictions(dir1 / "predictions.csv", stage1, fitted1, test1)
    _write_json(dir1 / "metrics.json", metrics1)
    _write_json(dir1 / "run_manifest.json", {"alignment": alignment, "training_summary": fitted1.training_summary, "model_path": str(path1)})
    result1 = _result(model=model_name, bundle=stage1, metrics=metrics1, status="ok", error="", out_dir=dir1)

    continuation: dict = {"alignment": alignment}
    if model_name == "DecisionTree":
        prior_name = "stage1_tree_prediction"
        for frame in (train2, val2, test2):
            frame[prior_name] = predict_model(fitted1, frame.drop(columns=[label]))
        fitted2 = fit_model(
            model_name, train2, val2, label, stage1.seed + 1,
            checkpoint_dir=dir2 / "checkpoints", variant="continuous", stage=2,
        )
        continuation.update({"strategy": "stage1_tree_prediction_feature", "prior_feature": prior_name})
    else:
        fitted2 = fit_model(
            model_name, train2, val2, label, stage1.seed + 1,
            checkpoint_dir=dir2 / "checkpoints", variant="continuous", stage=2,
            continue_from=fitted1,
        )
        continuation.update({"strategy": "native_warm_start_or_booster_continuation"})
    metrics2 = evaluate_model(fitted2, test2, label)
    path2 = save_fitted_model(fitted2, dir2)
    _save_predictions(dir2 / "predictions.csv", stage2, fitted2, test2)
    _write_json(dir2 / "metrics.json", metrics2)
    _write_json(
        dir2 / "continuation_manifest.json",
        {**continuation, "stage1_model_path": str(path1), "stage2_model_path": str(path2), "training_summary": fitted2.training_summary},
    )
    result2 = _result(model=model_name, bundle=stage2, metrics=metrics2, status="continued", error="", out_dir=dir2)
    return result1, result2


def _fit_new_model(
    model_name: str,
    stage1: StageDataBundle,
    stage2: StageDataBundle,
    model_root: Path,
) -> tuple[ModelStageResult, ModelStageResult]:
    label = stage1.label_col
    dir1 = model_root / stage1.stage
    dir2 = model_root / stage2.stage
    fitted1 = fit_model(
        model_name, stage1.train_df, stage1.val_df, label, stage1.seed,
        checkpoint_dir=dir1 / "checkpoints",
    )
    metrics1 = evaluate_model(fitted1, stage1.test_df, label)
    path1 = save_fitted_model(fitted1, dir1)
    _save_predictions(dir1 / "predictions.csv", stage1, fitted1, stage1.test_df)
    _write_json(dir1 / "metrics.json", metrics1)
    _write_json(dir1 / "run_manifest.json", {"training_summary": fitted1.training_summary, "model_path": str(path1)})
    result1 = _result(model=model_name, bundle=stage1, metrics=metrics1, status="ok", error="", out_dir=dir1)

    (compatible_train, compatible_val, compatible_test), compatible_meta = _stage1_compatible_views(stage1, stage2)
    prior_name = "stage1_prior_prediction" if model_name == "CORELS" else "stage1_prior_probability"
    stage2_frames = [stage2.train_df.copy(), stage2.val_df.copy(), stage2.test_df.copy()]
    compatible_frames = [compatible_train, compatible_val, compatible_test]
    for target, compatible in zip(stage2_frames, compatible_frames, strict=True):
        if model_name == "CORELS":
            target[prior_name] = predict_model(fitted1, compatible.drop(columns=[label]))
        else:
            target[prior_name] = predict_positive_probability(fitted1, compatible.drop(columns=[label]))
    train2, val2, test2 = stage2_frames
    fitted2 = fit_model(
        model_name, train2, val2, label, stage1.seed + 1,
        checkpoint_dir=dir2 / "checkpoints",
    )
    metrics2 = evaluate_model(fitted2, test2, label)
    path2 = save_fitted_model(fitted2, dir2)
    _save_predictions(dir2 / "predictions.csv", stage2, fitted2, test2)
    _write_json(dir2 / "metrics.json", metrics2)
    _write_json(
        dir2 / "continuation_manifest.json",
        {
            "continuation_strategy": "prior_feature_cascade",
            "stage1_model_path": str(path1),
            "stage2_model_path": str(path2),
            "prior_feature_name": prior_name,
            "compatible_view": compatible_meta,
            "training_summary": fitted2.training_summary,
        },
    )
    result2 = _result(model=model_name, bundle=stage2, metrics=metrics2, status="continued", error="", out_dir=dir2)
    return result1, result2


def run_baseline_experiments(
    *,
    models: tuple[str, ...] = ALL_MODEL_NAMES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    resume: bool = False,
    retry_errors: bool = False,
) -> list[ModelStageResult]:
    settings = get_default_experiment_settings()
    dataset = settings.dataset
    rows = _read_existing(RESULTS_PATH) if resume else {}
    total = len(seeds) * len(models)
    completed = 0

    for seed in seeds:
        stage1, stage2 = prepare_two_stage_data_bundles(
            ds=dataset,
            stage1_drift=build_stage1_drift(settings, dataset.prev_hl_out_dir),
            stage2_drift=build_stage2_drift_template(settings),
            stage1=settings.stages[0],
            stage2=settings.stages[1],
            seed=int(seed),
            split_spec=settings.split_spec,
        )
        validation = _validate_stage_data(stage1, stage2)
        seed_root = OUTPUT_ROOT / f"seed{seed}" / dataset.name
        _write_json(
            seed_root / "stage_data_manifest.json",
            {
                "dataset": dataset.name,
                "seed": int(seed),
                "stage1": stage_bundle_manifest(stage1),
                "stage2": stage_bundle_manifest(stage2),
                "validation": validation,
            },
        )

        for model_name in models:
            completed += 1
            key1 = (int(seed), model_name, stage1.stage)
            key2 = (int(seed), model_name, stage2.stage)
            previous1 = rows.get(key1)
            previous2 = rows.get(key2)
            if previous1 is not None and previous2 is not None:
                if not retry_errors or (
                    previous1.status in {"ok", "continued"} and previous2.status in {"ok", "continued"}
                ):
                    print(f"[{completed}/{total}] skip seed={seed} model={model_name}", flush=True)
                    continue

            model_root = seed_root / model_name
            print(f"[{completed}/{total}] fit continuous seed={seed} model={model_name}", flush=True)
            started = time.perf_counter()
            try:
                if model_name in NEW_MODEL_NAMES:
                    result1, result2 = _fit_new_model(model_name, stage1, stage2, model_root)
                else:
                    result1, result2 = _fit_existing_model(model_name, stage1, stage2, model_root)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                model_root.mkdir(parents=True, exist_ok=True)
                (model_root / "error.txt").write_text(error, encoding="utf-8")
                result1 = _result(model=model_name, bundle=stage1, metrics=None, status="error", error=error, out_dir=model_root / stage1.stage)
                result2 = _result(model=model_name, bundle=stage2, metrics=None, status="error", error=f"Stage1/continuation failed: {error}", out_dir=model_root / stage2.stage)
                print(f"[{completed}/{total}] error seed={seed} model={model_name}: {error}", flush=True)
            rows[key1] = result1
            rows[key2] = result2
            _write_json(model_root / "elapsed.json", {"elapsed_seconds": time.perf_counter() - started})
            write_results_csv(RESULTS_PATH, _ordered_results(rows))

    ordered = _ordered_results(rows)
    write_results_csv(RESULTS_PATH, ordered)
    print(f"continuous_baselines_rerun_results_csv={RESULTS_PATH}", flush=True)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage continuous-learning baselines.")
    parser.add_argument("--models", default="all", help="all or comma-separated canonical model names")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    print(
        f"Running baseline continuous learning on {MIMIC_CSV_PATH} with label={MIMIC_LABEL_COL}, "
        f"seeds={tuple(args.seeds)}, models={args.models}.",
        flush=True,
    )
    run_baseline_experiments(
        models=parse_model_names(args.models),
        seeds=tuple(int(seed) for seed in args.seeds),
        resume=bool(args.resume),
        retry_errors=bool(args.retry_errors),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
