from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

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
from hl.config import LLMConfig, RunConfig
from hl.metrics import compute_metrics
from hl.orchestrator import run_heuristic_learning


LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_MODEL = "deepseek-v4-pro"
LLM_TEMPERATURE = 0.0

def _read_results(path: Path) -> dict[tuple[int, str, str], ModelStageResult]:
    if not path.exists():
        return {}
    rows: dict[tuple[int, str, str], ModelStageResult] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stage = row["阶段"]
            result = ModelStageResult(
                model=row["模型"],
                dataset=row["数据集"],
                seed=int(row["seed"]),
                stage=stage,
                acc=row["ACC"],
                f1=row["F1"],
                sensitivity=row["Sensitivity"],
                specificity=row["Specificity"],
                status=row["status"],
                error=row["error"],
                out_dir=row["out_dir"],
            )
            rows[(result.seed, result.model, result.stage)] = result
    return rows


def _ordered(rows: dict[tuple[int, str, str], ModelStageResult]) -> list[ModelStageResult]:
    model_order = {name: index for index, name in enumerate((*BASELINE_MODEL_NAMES, "HL"))}
    stage_order = {name: index for index, name in enumerate(REGIME_ORDER)}
    return sorted(
        rows.values(),
        key=lambda row: (
            int(row.seed),
            model_order.get(row.model, 999),
            stage_order.get(row.stage, 999),
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _metric_text(metrics: dict[str, float], key: str) -> str:
    return f"{float(metrics[key]):.3f}"


def _load_predict_fn(model_path: Path) -> Callable[[dict[str, Any]], int]:
    module_name = f"hl_stage2_direct_{model_path.parent.parent.parent.name}_{model_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict_fn = getattr(module, "predict", None)
    if not callable(predict_fn):
        raise RuntimeError(f"`predict(features)` not found in {model_path}")
    return predict_fn


def _predict_labels(
    predict_fn: Callable[[dict[str, Any]], int],
    frame: pd.DataFrame,
    *,
    label_col: str,
) -> np.ndarray:
    feature_cols = [column for column in frame.columns if column != label_col]
    predictions = [
        int(predict_fn({column: row[column] for column in feature_cols}))
        for _, row in frame.iterrows()
    ]
    output = np.asarray(predictions, dtype=int)
    if not np.isin(output, [0, 1]).all():
        raise ValueError(f"HL predictions must be binary, got {sorted(np.unique(output).tolist())}")
    return output


def _validate_stage_data(stage1: StageDataBundle, stage2: StageDataBundle) -> dict[str, Any]:
    sizes = (
        len(stage1.train_df),
        len(stage1.val_df),
        len(stage1.test_df),
        len(stage2.train_df),
        len(stage2.val_df),
        len(stage2.test_df),
    )
    if sizes != (1000, 500, 800, 40, 500, 800):
        raise ValueError(f"Unexpected stage sizes: {sizes}")
    row_sets = {
        "stage1_train": set(stage1.train_sampling_meta["train_source_row_ids"]),
        "stage1_val": set(stage1.split_meta["val_source_row_ids"]),
        "stage1_test": set(stage1.split_meta["test_source_row_ids"]),
        "stage2_train": set(stage2.train_sampling_meta["train_source_row_ids"]),
        "stage2_val": set(stage2.split_meta["val_source_row_ids"]),
        "stage2_test": set(stage2.split_meta["test_source_row_ids"]),
    }
    names = list(row_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if row_sets[left] & row_sets[right]:
                raise ValueError(f"Split overlap between {left} and {right}")
    return {
        "sizes": sizes,
        "all_pairwise_source_row_intersections_empty": True,
        "stage1_features": [column for column in stage1.train_df.columns if column != stage1.label_col],
        "stage2_features": [column for column in stage2.train_df.columns if column != stage2.label_col],
    }


def _direct_task_description() -> str:
    return (
        "You are building a prediction model for 28-day mortality from scratch. The data are derived from "
        "baseline information collected when patients are admitted to the ICU in the MIMIC database. The "
        "available admission features include the SOFA score; SIRS is not available. Design a clinically "
        "meaningful rule using only the supplied Stage 2 training and validation data."
    )


def _is_complete(result: ModelStageResult | None) -> bool:
    if result is None or result.status != "ok":
        return False
    out_dir = Path(result.out_dir)
    required = (
        out_dir / "final_heuristic_model.py",
        out_dir / "predictions.csv",
        out_dir / "metrics.json",
        out_dir / "run_manifest.json",
    )
    return all(path.exists() for path in required)


def _run_stage2_direct(
    *,
    stage2: StageDataBundle,
    llm_cfg: LLMConfig,
    out_dir: Path,
) -> ModelStageResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    task_description = _direct_task_description()
    run_cfg = RunConfig(
        output_dir=out_dir,
        iterations=10,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=task_description,
        random_seed=int(stage2.seed),
        llm_enabled=True,
    )
    run_heuristic_learning(
        train_df=stage2.train_df,
        val_df=stage2.val_df,
        label_col=stage2.label_col,
        run_cfg=run_cfg,
        llm_cfg=llm_cfg,
    )

    final_model_path = out_dir / "final_heuristic_model.py"
    predict_fn = _load_predict_fn(final_model_path)
    y_true = stage2.test_df[stage2.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, stage2.test_df, label_col=stage2.label_col)
    metrics = compute_metrics(y_true, y_pred)
    row_ids = stage2.split_meta["test_source_row_ids_ordered"]
    if len(row_ids) != len(y_pred):
        raise ValueError("HL prediction/test row ID count mismatch")
    pd.DataFrame(
        {
            "__continuous_row_id__": row_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            # HL exposes hard labels only. Keeping this column makes the artifact
            # schema identical to the baseline endpoints without inventing scores.
            "positive_probability": y_pred.astype(float),
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(
        out_dir / "run_manifest.json",
        {
            **stage_bundle_manifest(stage2),
            "reported_stage": STAGE2_DIRECT,
            "task_description": task_description,
            "executor": "hl.orchestrator.run_heuristic_learning",
            "model_reloaded_for_test": True,
            "continuation_strategy": "none_random_initialization",
            "stage1_state_accessed": False,
            "stage1_model_accessed": False,
            "stage1_training_rows_consumed": 0,
            "prediction_output": "hard_binary_label",
            "positive_probability_column": "hard_label_cast_to_float_not_calibrated_probability",
            "prediction_threshold": 0.5,
            "test_rows": int(len(y_pred)),
            "final_model_path": str(final_model_path),
            "llm": {
                "base_url": llm_cfg.base_url,
                "model_name": llm_cfg.model_name,
                "api_key_env": llm_cfg.api_key_env,
                "temperature": llm_cfg.temperature,
            },
        },
    )
    (out_dir / "error.txt").unlink(missing_ok=True)
    return ModelStageResult(
        model="HL",
        dataset=stage2.dataset,
        seed=int(stage2.seed),
        stage=STAGE2_DIRECT,
        acc=_metric_text(metrics, "ACC"),
        f1=_metric_text(metrics, "F1"),
        sensitivity=_metric_text(metrics, "Sensitivity"),
        specificity=_metric_text(metrics, "Specificity"),
        status="ok",
        error="",
        out_dir=str(out_dir),
    )


def _require_legacy_hl_rows(
    rows: dict[tuple[int, str, str], ModelStageResult],
    seeds: tuple[int, ...],
) -> None:
    """Provenance guard: the previously completed Stage 1/continual HL rows
    must already be present in the combined results CSV before this runner
    appends the direct Stage 2 rows."""

    for seed in seeds:
        for stage in (STAGE1_DIRECT, STAGE2_CONTINUAL):
            key = (int(seed), "HL", stage)
            if key not in rows:
                raise FileNotFoundError(f"Missing previously completed HL row {key} in {RESULTS_PATH}")


def run_hl_stage2_direct(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    resume: bool = False,
) -> list[ModelStageResult]:
    if not os.getenv(LLM_KEY_ENV):
        raise RuntimeError(f"Environment variable {LLM_KEY_ENV} is not set")
    settings = get_default_experiment_settings()
    rows = _read_results(RESULTS_PATH)
    _require_legacy_hl_rows(rows, seeds)
    write_results_csv(RESULTS_PATH, _ordered(rows))
    llm_cfg = LLMConfig(
        base_url=LLM_BASE_URL,
        api_key_env=LLM_KEY_ENV,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    for index, seed in enumerate(seeds, start=1):
        key = (int(seed), "HL", STAGE2_DIRECT)
        if resume and _is_complete(rows.get(key)):
            print(f"[{index}/{len(seeds)}] skip complete HL Stage 2 direct seed={seed}", flush=True)
            continue
        stage1, stage2 = prepare_two_stage_data_bundles(
            ds=settings.dataset,
            stage1_drift=build_stage1_drift(settings, settings.dataset.prev_hl_out_dir),
            stage2_drift=build_stage2_drift_template(settings),
            stage1=settings.stages[0],
            stage2=settings.stages[1],
            seed=int(seed),
            split_spec=settings.split_spec,
        )
        validation = _validate_stage_data(stage1, stage2)
        seed_root = OUTPUT_ROOT / f"seed{seed}" / settings.dataset.name
        _write_json(
            seed_root / "stage_data_manifest.json",
            {
                "dataset": settings.dataset.name,
                "seed": int(seed),
                "stage1": stage_bundle_manifest(stage1),
                "stage2": stage_bundle_manifest(stage2),
                "validation": validation,
            },
        )
        out_dir = seed_root / "HL" / STAGE2_DIRECT
        started = time.perf_counter()
        print(f"[{index}/{len(seeds)}] fit HL Stage 2 direct seed={seed}", flush=True)
        try:
            rows[key] = _run_stage2_direct(stage2=stage2, llm_cfg=llm_cfg, out_dir=out_dir)
            print(
                f"[{index}/{len(seeds)}] done HL Stage 2 direct seed={seed} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        except Exception as exc:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            rows[key] = ModelStageResult(
                model="HL",
                dataset=stage2.dataset,
                seed=int(seed),
                stage=STAGE2_DIRECT,
                acc="",
                f1="",
                sensitivity="",
                specificity="",
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                out_dir=str(out_dir),
            )
            print(f"[{index}/{len(seeds)}] error HL Stage 2 direct seed={seed}: {exc}", flush=True)
        _write_json(out_dir / "elapsed.json", {"elapsed_seconds": time.perf_counter() - started})
        write_results_csv(RESULTS_PATH, _ordered(rows))

    ordered = _ordered(rows)
    write_results_csv(RESULTS_PATH, ordered)
    print(f"continuous_learning_v2_results_csv={RESULTS_PATH}", flush=True)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run from-scratch HL on the frozen Stage 2 split.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    print(
        f"Running direct Stage 2 HL on {MIMIC_CSV_PATH} with label={MIMIC_LABEL_COL}, "
        f"seeds={seeds}, model={LLM_MODEL}.",
        flush=True,
    )
    run_hl_stage2_direct(seeds=seeds, resume=bool(args.resume))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
