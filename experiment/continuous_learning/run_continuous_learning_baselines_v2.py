from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "experiment" / "outputs_rerun" / "continuous_learning_v2"
RESULTS_PATH = SCRIPT_DIR / "continuous_baselines_v2_results.csv"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from continuous_baseline_v2 import (
    BASELINE_MODEL_NAMES,
    REGIME_ORDER,
    STAGE1_DIRECT,
    STAGE2_CONTINUAL,
    STAGE2_DIRECT,
    frames_for_model,
    train_model_trio,
)
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
    FittedModel,
    evaluate_model,
    predict_positive_probability,
    save_fitted_model,
)


def _metric_text(metrics: dict[str, float], key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else f"{float(value):.3f}"


def _result(
    *,
    model: str,
    bundle: StageDataBundle,
    stage: str,
    metrics: dict[str, float] | None,
    status: str,
    error: str,
    out_dir: Path,
) -> ModelStageResult:
    return ModelStageResult(
        model=model,
        dataset=bundle.dataset,
        seed=bundle.seed,
        stage=stage,
        acc=_metric_text(metrics or {}, "ACC"),
        f1=_metric_text(metrics or {}, "F1"),
        sensitivity=_metric_text(metrics or {}, "Sensitivity"),
        specificity=_metric_text(metrics or {}, "Specificity"),
        status=status,
        error=error,
        out_dir=str(out_dir),
    )


def _key(result: ModelStageResult) -> tuple[int, str, str]:
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
            rows[_key(result)] = result
    return rows


def _ordered(rows: dict[tuple[int, str, str], ModelStageResult]) -> list[ModelStageResult]:
    model_order = {name: idx for idx, name in enumerate((*BASELINE_MODEL_NAMES, "HL"))}
    stage_order = {name: idx for idx, name in enumerate(REGIME_ORDER)}
    return sorted(
        rows.values(),
        key=lambda result: (
            int(result.seed),
            model_order.get(result.model, 999),
            stage_order.get(result.stage, 999),
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _validate_stage_data(stage1: StageDataBundle, stage2: StageDataBundle) -> dict[str, Any]:
    if stage1.dataset != stage2.dataset or stage1.seed != stage2.seed or stage1.label_col != stage2.label_col:
        raise ValueError("Stage identity mismatch")
    actual = (
        len(stage1.train_df), len(stage1.val_df), len(stage1.test_df),
        len(stage2.train_df), len(stage2.val_df), len(stage2.test_df),
    )
    expected = (1000, 500, 800, 40, 500, 800)
    if actual != expected:
        raise ValueError(f"Unexpected stage sizes: expected={expected}, actual={actual}")
    row_sets = {
        "stage1_train": set(stage1.train_sampling_meta["train_source_row_ids"]),
        "stage1_val": set(stage1.split_meta["val_source_row_ids"]),
        "stage1_test": set(stage1.split_meta["test_source_row_ids"]),
        "stage2_train": set(stage2.train_sampling_meta["train_source_row_ids"]),
        "stage2_val": set(stage2.split_meta["val_source_row_ids"]),
        "stage2_test": set(stage2.split_meta["test_source_row_ids"]),
    }
    keys = list(row_sets)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            overlap = row_sets[left] & row_sets[right]
            if overlap:
                raise ValueError(f"Split overlap between {left} and {right}: {sorted(overlap)[:10]}")
    return {
        "sizes": actual,
        "all_pairwise_source_row_intersections_empty": True,
        "stage1_features": [col for col in stage1.train_df.columns if col != stage1.label_col],
        "stage2_features": [col for col in stage2.train_df.columns if col != stage2.label_col],
    }


def _persist_endpoint(
    *,
    model_name: str,
    fitted: FittedModel,
    bundle: StageDataBundle,
    frame: pd.DataFrame,
    stage: str,
    out_dir: Path,
    manifest: dict[str, Any],
) -> ModelStageResult:
    label = bundle.label_col
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(fitted, frame, label)
    probabilities = predict_positive_probability(fitted, frame.drop(columns=[label]))
    predictions = (probabilities >= 0.5).astype(int)
    ids = bundle.split_meta.get("test_source_row_ids_ordered") or list(range(len(frame)))
    if len(probabilities) != len(frame) or len(ids) != len(frame):
        raise ValueError("Prediction/test row count mismatch")
    pd.DataFrame(
        {
            "__continuous_row_id__": ids,
            "y_true": frame[label].astype(int).to_numpy(),
            "y_pred": predictions,
            "positive_probability": probabilities,
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    _write_json(out_dir / "metrics.json", metrics)
    model_path = save_fitted_model(fitted, out_dir)
    reload_diff = _verify_reload(fitted, model_path, frame.drop(columns=[label]), probabilities)
    full_manifest = {
        **manifest,
        "model_path": str(model_path),
        "training_summary": fitted.training_summary,
        "prediction_threshold": 0.5,
        "test_rows": int(len(frame)),
        "reload_max_abs_probability_diff": reload_diff,
    }
    name = "continuation_manifest.json" if stage == STAGE2_CONTINUAL else "run_manifest.json"
    _write_json(out_dir / name, full_manifest)
    status = "continued" if stage == STAGE2_CONTINUAL else "ok"
    return _result(
        model=model_name,
        bundle=bundle,
        stage=stage,
        metrics=metrics,
        status=status,
        error="",
        out_dir=out_dir,
    )


def _verify_reload(
    fitted: FittedModel,
    model_path: Path,
    features: pd.DataFrame,
    expected: np.ndarray,
) -> float:
    if fitted.family == "deeptab":
        loaded = type(fitted.estimator).load(str(model_path))
        actual_matrix = np.asarray(loaded.predict_proba(features), dtype=float)
        actual = actual_matrix[:, 1]
    else:
        loaded = joblib.load(model_path)
        actual = predict_positive_probability(loaded, features)
    diff = float(np.max(np.abs(np.asarray(expected, dtype=float) - np.asarray(actual, dtype=float))))
    tolerance = 1e-6 if fitted.family == "deeptab" else 1e-12
    if diff > tolerance:
        raise AssertionError(f"Reloaded {fitted.model_name} prediction mismatch: {diff} > {tolerance}")
    return diff


def _model_complete(
    rows: dict[tuple[int, str, str], ModelStageResult],
    seed: int,
    model_name: str,
) -> bool:
    for stage in REGIME_ORDER:
        result = rows.get((seed, model_name, stage))
        if result is None or result.status not in {"ok", "continued"}:
            return False
        out_dir = Path(result.out_dir)
        if not (out_dir / "predictions.csv").exists() or not (out_dir / "metrics.json").exists():
            return False
    return True


def _parse_models(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return BASELINE_MODEL_NAMES
    requested = {item.strip() for item in value.split(",") if item.strip()}
    unknown = sorted(requested - set(BASELINE_MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown baseline models: {unknown}; expected {BASELINE_MODEL_NAMES}")
    return tuple(name for name in BASELINE_MODEL_NAMES if name in requested)


def run_baseline_experiments_v2(
    *,
    models: tuple[str, ...] = BASELINE_MODEL_NAMES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    resume: bool = False,
    retry_errors: bool = False,
) -> list[ModelStageResult]:
    settings = get_default_experiment_settings()
    dataset = settings.dataset
    existing_rows = _read_existing(RESULTS_PATH)
    # This CSV becomes the combined V2 table after HL is run. A deliberate
    # baseline rerun must never erase completed HL rows.
    rows = existing_rows if resume else {
        key: result for key, result in existing_rows.items() if result.model == "HL"
    }
    total = len(models) * len(seeds)
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
            if resume and _model_complete(rows, int(seed), model_name):
                print(f"[{completed}/{total}] skip complete seed={seed} model={model_name}", flush=True)
                continue
            if resume and not retry_errors:
                existing = [rows.get((int(seed), model_name, stage)) for stage in REGIME_ORDER]
                if all(row is not None for row in existing) and any(row.status == "error" for row in existing if row):
                    print(f"[{completed}/{total}] keep error seed={seed} model={model_name}", flush=True)
                    continue

            print(f"[{completed}/{total}] fit redesigned baselines seed={seed} model={model_name}", flush=True)
            model_root = seed_root / model_name
            started = time.perf_counter()
            try:
                trio = train_model_trio(model_name, stage1, stage2, work_dir=model_root / "lightning_runtime")
                frames = frames_for_model(model_name, stage1, stage2)
                endpoints = {
                    STAGE1_DIRECT: (trio.stage1, stage1),
                    STAGE2_CONTINUAL: (trio.continual, stage2),
                    STAGE2_DIRECT: (trio.stage2_direct, stage2),
                }
                for stage, (fitted, bundle) in endpoints.items():
                    out_dir = model_root / stage
                    result = _persist_endpoint(
                        model_name=model_name,
                        fitted=fitted,
                        bundle=bundle,
                        frame=frames[stage],
                        stage=stage,
                        out_dir=out_dir,
                        manifest=trio.manifests[stage],
                    )
                    rows[_key(result)] = result
                print(
                    f"[{completed}/{total}] done seed={seed} model={model_name} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                model_root.mkdir(parents=True, exist_ok=True)
                (model_root / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                for stage in REGIME_ORDER:
                    bundle = stage1 if stage == STAGE1_DIRECT else stage2
                    result = _result(
                        model=model_name,
                        bundle=bundle,
                        stage=stage,
                        metrics=None,
                        status="error",
                        error=error,
                        out_dir=model_root / stage,
                    )
                    rows[_key(result)] = result
                print(f"[{completed}/{total}] error seed={seed} model={model_name}: {error}", flush=True)
            _write_json(model_root / "elapsed.json", {"elapsed_seconds": time.perf_counter() - started})
            write_results_csv(RESULTS_PATH, _ordered(rows))

    ordered = _ordered(rows)
    write_results_csv(RESULTS_PATH, ordered)
    print(f"continuous_baselines_v2_results_csv={RESULTS_PATH}", flush=True)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run redesigned three-branch continual-learning baselines.")
    parser.add_argument("--models", default="all", help="all or comma-separated redesigned baseline names")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    models = _parse_models(args.models)
    print(
        f"Running redesigned baseline experiment on {MIMIC_CSV_PATH} with label={MIMIC_LABEL_COL}, "
        f"seeds={tuple(args.seeds)}, models={models}.",
        flush=True,
    )
    run_baseline_experiments_v2(
        models=models,
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
