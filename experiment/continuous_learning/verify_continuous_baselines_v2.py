from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
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
from continuous_learning_experiment_common import DEFAULT_SEEDS
from hl.metrics import compute_metrics
from run_continuous_learning_baselines_v2 import RESULTS_PATH


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_stage_data_manifest(out_dir: Path) -> Path:
    for directory in (out_dir, *out_dir.parents):
        candidate = directory / "stage_data_manifest.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No stage_data_manifest.json found above {out_dir}")


def verify(*, allow_partial: bool = False) -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results CSV does not exist: {RESULTS_PATH}")
    results = pd.read_csv(RESULTS_PATH, keep_default_na=False)
    required_columns = {
        "模型", "数据集", "seed", "阶段", "ACC", "F1",
        "Sensitivity", "Specificity", "status", "error", "out_dir",
    }
    if set(results.columns) != required_columns:
        raise AssertionError(f"Unexpected result columns: {list(results.columns)}")
    keys = list(zip(results["模型"], results["seed"].astype(int), results["阶段"], strict=True))
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate (model, seed, stage) result rows")
    expected_baselines = {
        (model, int(seed), stage)
        for model in BASELINE_MODEL_NAMES
        for seed in DEFAULT_SEEDS
        for stage in REGIME_ORDER
    }
    expected_hl = {
        ("HL", int(seed), stage)
        for seed in DEFAULT_SEEDS
        for stage in REGIME_ORDER
    }
    expected = expected_baselines | expected_hl
    actual = set(keys)
    if allow_partial:
        if not actual or not actual.issubset(expected):
            raise AssertionError(f"Partial result keys are invalid: {sorted(actual - expected)}")
    elif actual != expected:
        raise AssertionError(
            f"Expected {len(expected)} result rows, got {len(actual)}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if not set(results["status"]).issubset({"ok", "continued"}):
        failures = results.loc[~results["status"].isin(["ok", "continued"]), ["模型", "seed", "阶段", "error"]]
        raise AssertionError(f"Failed result rows:\n{failures.to_string(index=False)}")

    metric_columns = ("ACC", "F1", "Sensitivity", "Specificity")
    for metric_name in metric_columns:
        values = pd.to_numeric(results[metric_name], errors="raise")
        if not values.between(0.0, 1.0).all():
            raise AssertionError(f"Metric outside [0,1] in result column {metric_name}")

    checked_predictions = 0
    legacy_hl_rows = 0
    for row in results.to_dict(orient="records"):
        model = row["模型"]
        stage = row["阶段"]
        out_dir = Path(row["out_dir"])
        # Preserve compatibility until the old Stage 1/continual HL rows are
        # replaced. Their artifact paths belong to a former workspace.
        if model == "HL" and stage != STAGE2_DIRECT and not (out_dir / "predictions.csv").exists():
            legacy_hl_rows += 1
            continue
        predictions_path = out_dir / "predictions.csv"
        metrics_path = out_dir / "metrics.json"
        if not predictions_path.exists() or not metrics_path.exists():
            raise AssertionError(f"Missing artifacts in {out_dir}")
        predictions = pd.read_csv(predictions_path)
        if list(predictions.columns) != [
            "__continuous_row_id__", "y_true", "y_pred", "positive_probability"
        ]:
            raise AssertionError(f"Unexpected prediction columns in {predictions_path}")
        if len(predictions) != 800:
            raise AssertionError(f"Expected 800 predictions in {predictions_path}, got {len(predictions)}")
        data_manifest = _read_json(_find_stage_data_manifest(out_dir))
        data_stage = "stage1" if stage == STAGE1_DIRECT else "stage2"
        expected_row_ids = data_manifest[data_stage]["split"]["test_source_row_ids_ordered"]
        actual_row_ids = predictions["__continuous_row_id__"].astype(int).tolist()
        if actual_row_ids != expected_row_ids:
            raise AssertionError(f"Test row IDs do not match the frozen split in {predictions_path}")
        if data_manifest["validation"]["all_pairwise_source_row_intersections_empty"] is not True:
            raise AssertionError(f"Data split overlap validation failed for {out_dir}")
        if not predictions["positive_probability"].between(0.0, 1.0).all():
            raise AssertionError(f"Probability outside [0,1] in {predictions_path}")
        if not predictions["y_pred"].equals((predictions["positive_probability"] >= 0.5).astype(int)):
            raise AssertionError(f"Prediction threshold mismatch in {predictions_path}")
        metrics = _read_json(metrics_path)
        if not all(0.0 <= float(metrics[name]) <= 1.0 for name in ("ACC", "F1", "Sensitivity", "Specificity")):
            raise AssertionError(f"Metric outside [0,1] in {metrics_path}")
        recomputed = compute_metrics(
            predictions["y_true"].astype(int).to_numpy(),
            predictions["y_pred"].astype(int).to_numpy(),
        )
        for metric_name in ("ACC", "F1", "Sensitivity", "Specificity"):
            if abs(float(metrics[metric_name]) - float(recomputed[metric_name])) > 1e-12:
                raise AssertionError(f"Metric recomputation mismatch for {metric_name} in {out_dir}")
            if f"{float(row[metric_name]):.3f}" != f"{float(metrics[metric_name]):.3f}":
                raise AssertionError(f"Result CSV mismatch for {metric_name} in {out_dir}")
        manifest_name = "continuation_manifest.json" if stage == STAGE2_CONTINUAL else "run_manifest.json"
        manifest = _read_json(out_dir / manifest_name)
        if int(manifest["test_rows"]) != 800:
            raise AssertionError(f"Manifest test_rows mismatch in {out_dir}")
        if model != "HL" and float(manifest["reload_max_abs_probability_diff"]) > 1e-6:
            raise AssertionError(f"Reload prediction mismatch in {out_dir}")
        if stage == STAGE2_DIRECT:
            if manifest.get("continuation_strategy") != "none_random_initialization":
                raise AssertionError(f"Direct Stage 2 is not marked as random initialization: {out_dir}")
            if manifest.get("stage1_state_accessed") is not False:
                raise AssertionError(f"Direct Stage 2 accessed Stage 1 state: {out_dir}")
            if model == "HL":
                if manifest.get("stage1_model_accessed") is not False:
                    raise AssertionError(f"Direct HL accessed a Stage 1 model: {out_dir}")
                if int(manifest.get("stage1_training_rows_consumed", -1)) != 0:
                    raise AssertionError(f"Direct HL consumed Stage 1 training rows: {out_dir}")
                if manifest.get("prediction_output") != "hard_binary_label":
                    raise AssertionError(f"Unexpected direct HL prediction type: {out_dir}")
                if manifest.get("model_reloaded_for_test") is not True:
                    raise AssertionError(f"Direct HL was not reloaded for held-out testing: {out_dir}")
        checked_predictions += len(predictions)

    for model in BASELINE_MODEL_NAMES:
        for seed in DEFAULT_SEEDS:
            subset = results.loc[(results["模型"] == model) & (results["seed"].astype(int) == int(seed))]
            if subset.empty:
                if allow_partial:
                    continue
                raise AssertionError(f"Missing model/seed results: {model}/{seed}")
            if set(subset["阶段"]) != set(REGIME_ORDER):
                raise AssertionError(f"Incomplete regimes for {model}/{seed}")
            continual_dir = Path(subset.loc[subset["阶段"] == STAGE2_CONTINUAL, "out_dir"].iloc[0])
            manifest = _read_json(continual_dir / "continuation_manifest.json")
            if model == "MLP":
                if manifest["source_model_hash"] != manifest["initial_target_model_hash"]:
                    raise AssertionError(f"MLP transfer hash mismatch for seed {seed}")
            elif model in {"FT-Transformer", "ResNet"}:
                if manifest["source_model_state_hash"] != manifest["initial_target_model_state_hash"]:
                    raise AssertionError(f"DeepTab transfer hash mismatch for {model}/seed {seed}")
                model_root = continual_dir.parent
                checkpoints = list(model_root.rglob("*.ckpt"))
                if checkpoints:
                    raise AssertionError(f"Unexpected DeepTab checkpoints: {checkpoints}")
            elif model in {"XGBoost", "LightGBM"}:
                if manifest.get("feature_names_equal") is not True:
                    raise AssertionError(f"Booster feature mismatch for {model}/seed {seed}")
            elif model == "EBM":
                if not manifest.get("fit_uses_source_raw_init_score"):
                    raise AssertionError(f"EBM fit omitted init_score for seed {seed}")
                if not manifest.get("prediction_uses_source_raw_init_score"):
                    raise AssertionError(f"EBM prediction omitted init_score for seed {seed}")

    return {
        "result_rows": int(len(results)),
        "models": sorted(results["模型"].unique().tolist()),
        "seeds": sorted(int(seed) for seed in results["seed"].unique()),
        "prediction_rows_checked": int(checked_predictions),
        "legacy_hl_rows_without_v2_artifact_check": int(legacy_hl_rows),
        "allow_partial": bool(allow_partial),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify continual-learning baseline and HL artifacts.")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(allow_partial=bool(args.allow_partial)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
