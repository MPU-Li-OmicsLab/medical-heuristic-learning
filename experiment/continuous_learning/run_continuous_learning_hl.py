from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from continuous_learning_experiment_common import (
    DatasetSpec,
    DEFAULT_SEEDS,
    MIMIC_CSV_PATH,
    MIMIC_LABEL_COL,
    ModelStageResult,
    StageDataBundle,
    build_stage1_drift,
    build_stage2_drift_template,
    get_default_experiment_settings,
    make_stage2_drift,
    prepare_two_stage_data_bundles,
    stage_bundle_manifest,
    write_results_csv,
)
from hl.config import LLMConfig, RunConfig
from hl.continuous_learning import ContinuousLearningConfig, run_continuous_learning
from hl.orchestrator import run_heuristic_learning
from hl.metrics import compute_metrics
from hl.utils.io import write_json, write_text

LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_MODEL = "deepseek-v4-pro"
LLM_TEMPERATURE = 0.0


def run_hl_experiments() -> list[ModelStageResult]:
    settings = get_default_experiment_settings()
    llm_cfg = LLMConfig(
        base_url=LLM_BASE_URL,
        api_key_env=LLM_KEY_ENV,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    results: list[ModelStageResult] = []
    ds = settings.dataset
    for seed in settings.seeds:
        stage1_bundle, stage2_bundle_template = prepare_two_stage_data_bundles(
            ds=ds,
            stage1_drift=build_stage1_drift(settings, ds.prev_hl_out_dir),
            stage2_drift=build_stage2_drift_template(settings),
            stage1=settings.stages[0],
            stage2=settings.stages[1],
            seed=seed,
            split_spec=settings.split_spec,
        )
        stage1_result, stage1_out_dir = _run_hl_stage(
            ds=ds,
            bundle=stage1_bundle,
            llm_cfg=llm_cfg,
            output_root=settings.output_root,
        )
        results.append(stage1_result)

        stage2_bundle = replace(
            stage2_bundle_template,
            drift=make_stage2_drift(settings, stage1_out_dir),
        )
        stage2_result, _stage2_out_dir = _run_hl_stage(
            ds=ds,
            bundle=stage2_bundle,
            llm_cfg=llm_cfg,
            output_root=settings.output_root,
        )
        results.append(stage2_result)

    out_csv = SCRIPT_DIR / "continuous_hl_results.csv"
    write_results_csv(out_csv, results)
    print(f"continuous_hl_results_csv={out_csv}", flush=True)
    return results


def _run_hl_stage(
    *,
    ds: DatasetSpec,
    bundle: StageDataBundle,
    llm_cfg: LLMConfig,
    output_root: Path,
) -> tuple[ModelStageResult, Path]:
    out_dir = output_root / ds.name / f"seed{bundle.seed}" / bundle.stage / "HL" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_dir_text = str(bundle.drift.prev_hl_out_dir) if bundle.drift.prev_hl_out_dir is not None else "(none)"
    task_description = _build_task_description(ds=ds, bundle=bundle)
    manifest = stage_bundle_manifest(bundle)
    manifest["task_description"] = task_description
    write_json(out_dir / "adaptation_spec.json", manifest)

    if bundle.stage == "stage1_train1000":
        run_cfg = RunConfig(
            output_dir=out_dir,
            run_univariate_probe=True,
            run_knowledge_probe=True,
            run_v0_generation=True,
            run_iterations=True,
            task_description=task_description,
            random_seed=int(bundle.seed),
            llm_enabled=True,
        )
        run_heuristic_learning(
            train_df=bundle.train_df,
            test_df=bundle.val_df,
            label_col=ds.label_col,
            run_cfg=run_cfg,
            llm_cfg=llm_cfg,
        )
        heuristic_path = out_dir / "heuristic_system.py"
        final_model_path = out_dir / "final_heuristic_model.py"
    else:
        continuous_cfg = ContinuousLearningConfig(
            output_dir=out_dir,
            run_univariate_probe=True,
            run_knowledge_probe=True,
            run_v0_generation=True,
            run_iterations=True,
            task_description=task_description,
            random_seed=int(bundle.seed),
            llm_enabled=True,
            drift=bundle.drift,
        )
        result = run_continuous_learning(
            train_df=bundle.train_df,
            test_df=bundle.val_df,
            label_col=ds.label_col,
            llm_cfg=llm_cfg,
            continuous_cfg=continuous_cfg,
        )
        heuristic_path = result.heuristic_path
        final_model_path = result.final_model_path

    predict_fn = _load_predict_fn(final_model_path)
    y_true = bundle.test_df[ds.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, bundle.test_df, label_col=ds.label_col)
    metrics = compute_metrics(y_true, y_pred)
    write_json(
        out_dir / "heldout_test_summary.json",
        {
            **manifest,
            "heldout_test_metrics": metrics,
            "llm": {
                "base_url": llm_cfg.base_url,
                "model_name": llm_cfg.model_name,
                "api_key_env": llm_cfg.api_key_env,
            },
            "executor": "hl.orchestrator" if bundle.stage == "stage1_train1000" else "hl.continuous_learning",
            "heuristic_path": str(heuristic_path),
            "final_model_path": str(final_model_path),
        },
    )
    write_text(
        out_dir / "heldout_test_summary.txt",
        "\n".join(
            [
                f"dataset={ds.name}",
                f"seed={bundle.seed}",
                f"stage={bundle.stage}",
                f"executor={'hl.orchestrator' if bundle.stage == 'stage1_train1000' else 'hl.continuous_learning'}",
                f"prev_hl_out_dir={prev_dir_text}",
                f"heldout_test_metrics={metrics}",
                f"final_model_path={final_model_path}",
            ]
        )
        + "\n",
    )
    stage_result = ModelStageResult(
        model="HL",
        dataset=ds.name,
        seed=bundle.seed,
        stage=bundle.stage,
        acc=_metric_text(metrics, "ACC"),
        f1=_metric_text(metrics, "F1"),
        sensitivity=_metric_text(metrics, "Sensitivity"),
        specificity=_metric_text(metrics, "Specificity"),
        status="ok",
        error="",
        out_dir=str(out_dir),
    )
    return stage_result, out_dir


def _build_task_description(*, ds: DatasetSpec, bundle: StageDataBundle) -> str:
    if ds.name != "MIMIC":
        raise ValueError(f"Only MIMIC task descriptions are supported, got dataset={ds.name}")
    if bundle.stage == "stage1_train1000":
        return (
            "You are building a prediction model for 28-day mortality. The data are derived from baseline "
            "information collected when patients are admitted to the ICU in the MIMIC database. The prediction "
            "target is 28-day death, and the rule should be designed to capture clinically meaningful risk "
            "patterns present at ICU admission."
        )
    if bundle.stage == "stage2_train40":
        return (
            "Due to changes in sepsis assessment guidelines, the SIRS index has been replaced by the SOFA "
            "index. The prediction model for 28-day mortality therefore needs to be reconstructed under this "
            "feature shift. The data still describe baseline information collected at ICU admission in the "
            "MIMIC database, and the updated rule should adapt to this change while continuing to predict "
            "28-day death in a clinically meaningful way."
        )
    raise ValueError(f"Unsupported stage for MIMIC task description: {bundle.stage}")


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


def _predict_labels(predict_fn, df, *, label_col: str) -> np.ndarray:
    feature_cols = [col for col in df.columns if col != label_col]
    preds: list[int] = []
    for _, row in df.iterrows():
        features = {col: row[col] for col in feature_cols}
        preds.append(int(predict_fn(features)))
    return np.asarray(preds, dtype=int)


def _metric_text(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return f"{float(value):.3f}" if value is not None else ""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    print(
        f"Running HL continuous learning on MIMIC from {MIMIC_CSV_PATH} "
        f"with label={MIMIC_LABEL_COL} and seeds={DEFAULT_SEEDS}.",
        flush=True,
    )
    run_hl_experiments()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
