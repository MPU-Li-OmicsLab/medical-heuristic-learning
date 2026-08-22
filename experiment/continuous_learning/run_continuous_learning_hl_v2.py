from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
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
    ExperimentSettings,
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
from hl.metrics import compute_metrics
from hl.orchestrator import run_heuristic_learning


LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_MODEL = "deepseek-v4-pro"
LLM_TEMPERATURE = 0.0

STAGE_ALIASES = {
    "stage1": STAGE1_DIRECT,
    "stage1-direct": STAGE1_DIRECT,
    STAGE1_DIRECT: STAGE1_DIRECT,
    "stage1-to-stage2": STAGE2_CONTINUAL,
    "stage1->2": STAGE2_CONTINUAL,
    "continual": STAGE2_CONTINUAL,
    STAGE2_CONTINUAL: STAGE2_CONTINUAL,
    "stage2": STAGE2_DIRECT,
    "stage2-direct": STAGE2_DIRECT,
    STAGE2_DIRECT: STAGE2_DIRECT,
}


def _key(result: ModelStageResult) -> tuple[int, str, str]:
    return int(result.seed), result.model, result.stage


def _read_results(path: Path) -> dict[tuple[int, str, str], ModelStageResult]:
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
    module_name = f"hl_continuous_v2_{abs(hash(model_path.resolve()))}"
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
    if stage1.dataset != stage2.dataset or stage1.seed != stage2.seed or stage1.label_col != stage2.label_col:
        raise ValueError("Stage identity mismatch")
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


def _stage1_task_description() -> str:
    return (
        "You are building a prediction model for 28-day mortality. The data are derived from baseline "
        "information collected when patients are admitted to the ICU in the MIMIC database. The prediction "
        "target is 28-day death, and the rule should be designed to capture clinically meaningful risk "
        "patterns present at ICU admission."
    )


def _continual_task_description() -> str:
    return (
        "Due to changes in sepsis assessment guidelines, the SIRS index has been replaced by the SOFA "
        "index. Continue the existing prediction model for 28-day mortality under this feature shift. "
        "The data still describe baseline information collected at ICU admission in the MIMIC database, "
        "and the updated rule should adapt the Stage 1 rule rather than train an unrelated rule from scratch."
    )


def _direct_task_description() -> str:
    return (
        "You are building a prediction model for 28-day mortality from scratch. The data are derived from "
        "baseline information collected when patients are admitted to the ICU in the MIMIC database. The "
        "available admission features include the SOFA score; SIRS is not available. Design a clinically "
        "meaningful rule using only the supplied Stage 2 training and validation data."
    )


def _manifest_name(stage: str) -> str:
    return "continuation_manifest.json" if stage == STAGE2_CONTINUAL else "run_manifest.json"


def _required_artifacts(stage: str, *, as_stage1_dependency: bool = False) -> tuple[str, ...]:
    required = (
        "final_heuristic_model.py",
        "predictions.csv",
        "metrics.json",
        _manifest_name(stage),
    )
    if as_stage1_dependency:
        required += ("probe_univariate_results.csv", "probe_knowledge.md")
    return required


def _is_complete(result: ModelStageResult | None, stage: str | None = None) -> bool:
    if result is None:
        return False
    expected_status = "continued" if result.stage == STAGE2_CONTINUAL else "ok"
    if result.status != expected_status or (stage is not None and result.stage != stage):
        return False
    out_dir = Path(result.out_dir)
    required = _required_artifacts(
        result.stage,
        as_stage1_dependency=result.stage == STAGE1_DIRECT,
    )
    return all((out_dir / name).exists() for name in required)


def _require_stage1_result(
    rows: dict[tuple[int, str, str], ModelStageResult],
    seed: int,
) -> tuple[ModelStageResult, Path]:
    key = (int(seed), "HL", STAGE1_DIRECT)
    result = rows.get(key)
    if result is None:
        raise FileNotFoundError(
            f"Cannot run {STAGE2_CONTINUAL} for seed={seed}: missing Stage 1 result row {key} in {RESULTS_PATH}. "
            "Run --stages stage1 first."
        )
    if result.status != "ok":
        raise RuntimeError(
            f"Cannot run {STAGE2_CONTINUAL} for seed={seed}: Stage 1 status is {result.status!r}, not 'ok'."
        )
    out_dir = Path(result.out_dir)
    missing = [
        name
        for name in _required_artifacts(STAGE1_DIRECT, as_stage1_dependency=True)
        if not (out_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot run {STAGE2_CONTINUAL} for seed={seed}: Stage 1 artifacts are incomplete in {out_dir}; "
            f"missing={missing}. Run --stages stage1 first."
        )
    return result, out_dir


def _parse_stages(values: list[str]) -> tuple[str, ...]:
    tokens = [token.strip() for value in values for token in value.split(",") if token.strip()]
    if not tokens or tokens == ["all"]:
        return REGIME_ORDER
    if "all" in tokens:
        raise ValueError("--stages all cannot be combined with individual stages")
    unknown = sorted(set(tokens) - set(STAGE_ALIASES))
    if unknown:
        valid = "stage1, stage1-to-stage2, stage2, all"
        raise ValueError(f"Unknown HL stages: {unknown}; expected one or more of: {valid}")
    requested = {STAGE_ALIASES[token] for token in tokens}
    return tuple(stage for stage in REGIME_ORDER if stage in requested)


def _validate_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("--run-id must be a non-empty single path component")
    return run_id


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _fresh_out_dir(seed_root: Path, stage: str, run_id: str) -> Path:
    out_dir = seed_root / "HL" / stage / run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing HL artifacts in {out_dir}. Use a new --run-id or --resume."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ensure_stage_data_manifest(
    *,
    seed_root: Path,
    dataset: str,
    seed: int,
    stage1: StageDataBundle,
    stage2: StageDataBundle,
    validation: dict[str, Any],
) -> None:
    path = seed_root / "stage_data_manifest.json"
    payload = {
        "dataset": dataset,
        "seed": int(seed),
        "stage1": stage_bundle_manifest(stage1),
        "stage2": stage_bundle_manifest(stage2),
        "validation": validation,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(payload, default=_json_default)):
            raise ValueError(f"Existing frozen stage-data manifest does not match regenerated data: {path}")
        return
    _write_json(path, payload)


def _persist_hl_endpoint(
    *,
    bundle: StageDataBundle,
    stage: str,
    llm_cfg: LLMConfig,
    out_dir: Path,
    final_model_path: Path,
    task_description: str,
    executor: str,
    extra_manifest: dict[str, Any],
) -> ModelStageResult:
    predict_fn = _load_predict_fn(final_model_path)
    y_true = bundle.test_df[bundle.label_col].astype(int).to_numpy()
    y_pred = _predict_labels(predict_fn, bundle.test_df, label_col=bundle.label_col)
    metrics = compute_metrics(y_true, y_pred)
    row_ids = bundle.split_meta["test_source_row_ids_ordered"]
    if len(row_ids) != len(y_pred):
        raise ValueError("HL prediction/test row ID count mismatch")
    pd.DataFrame(
        {
            "__continuous_row_id__": row_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            # HL exposes hard labels only. This keeps the shared artifact schema
            # without pretending that hard labels are calibrated probabilities.
            "positive_probability": y_pred.astype(float),
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(
        out_dir / _manifest_name(stage),
        {
            **stage_bundle_manifest(bundle),
            "reported_stage": stage,
            "task_description": task_description,
            "executor": executor,
            "model_reloaded_for_test": True,
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
            **extra_manifest,
        },
    )
    (out_dir / "error.txt").unlink(missing_ok=True)
    return ModelStageResult(
        model="HL",
        dataset=bundle.dataset,
        seed=int(bundle.seed),
        stage=stage,
        acc=_metric_text(metrics, "ACC"),
        f1=_metric_text(metrics, "F1"),
        sensitivity=_metric_text(metrics, "Sensitivity"),
        specificity=_metric_text(metrics, "Specificity"),
        status="continued" if stage == STAGE2_CONTINUAL else "ok",
        error="",
        out_dir=str(out_dir),
    )


def _run_stage1(
    *,
    stage1: StageDataBundle,
    llm_cfg: LLMConfig,
    out_dir: Path,
) -> ModelStageResult:
    task_description = _stage1_task_description()
    run_cfg = RunConfig(
        output_dir=out_dir,
        iterations=10,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=task_description,
        random_seed=int(stage1.seed),
        llm_enabled=True,
    )
    run_heuristic_learning(
        train_df=stage1.train_df,
        val_df=stage1.val_df,
        label_col=stage1.label_col,
        run_cfg=run_cfg,
        llm_cfg=llm_cfg,
    )
    return _persist_hl_endpoint(
        bundle=stage1,
        stage=STAGE1_DIRECT,
        llm_cfg=llm_cfg,
        out_dir=out_dir,
        final_model_path=out_dir / "final_heuristic_model.py",
        task_description=task_description,
        executor="hl.orchestrator.run_heuristic_learning",
        extra_manifest={
            "continuation_strategy": "none_random_initialization",
            "stage1_state_accessed": False,
        },
    )


def _run_stage2_continual(
    *,
    settings: ExperimentSettings,
    stage2_template: StageDataBundle,
    stage1_out_dir: Path,
    llm_cfg: LLMConfig,
    out_dir: Path,
) -> ModelStageResult:
    stage2 = replace(
        stage2_template,
        drift=make_stage2_drift(settings, stage1_out_dir),
        drift_meta={
            **stage2_template.drift_meta,
            "prev_hl_out_dir": str(stage1_out_dir),
        },
    )
    task_description = _continual_task_description()
    continuous_cfg = ContinuousLearningConfig(
        output_dir=out_dir,
        iterations=10,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=task_description,
        random_seed=int(stage2.seed),
        llm_enabled=True,
        drift=stage2.drift,
    )
    result = run_continuous_learning(
        train_df=stage2.train_df,
        val_df=stage2.val_df,
        label_col=stage2.label_col,
        llm_cfg=llm_cfg,
        continuous_cfg=continuous_cfg,
    )
    return _persist_hl_endpoint(
        bundle=stage2,
        stage=STAGE2_CONTINUAL,
        llm_cfg=llm_cfg,
        out_dir=out_dir,
        final_model_path=result.final_model_path,
        task_description=task_description,
        executor="hl.continuous_learning.run_continuous_learning",
        extra_manifest={
            "continuation_strategy": "hl_drift_aware_rule_adaptation",
            "source_stage": STAGE1_DIRECT,
            "source_stage1_out_dir": str(stage1_out_dir),
            "stage1_model_accessed": True,
            "stage1_training_rows_consumed": 0,
        },
    )


def _run_stage2_direct(
    *,
    stage2: StageDataBundle,
    llm_cfg: LLMConfig,
    out_dir: Path,
) -> ModelStageResult:
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
    return _persist_hl_endpoint(
        bundle=stage2,
        stage=STAGE2_DIRECT,
        llm_cfg=llm_cfg,
        out_dir=out_dir,
        final_model_path=out_dir / "final_heuristic_model.py",
        task_description=task_description,
        executor="hl.orchestrator.run_heuristic_learning",
        extra_manifest={
            "continuation_strategy": "none_random_initialization",
            "stage1_state_accessed": False,
            "stage1_model_accessed": False,
            "stage1_training_rows_consumed": 0,
        },
    )


def _error_result(*, bundle: StageDataBundle, stage: str, error: str, out_dir: Path) -> ModelStageResult:
    return ModelStageResult(
        model="HL",
        dataset=bundle.dataset,
        seed=int(bundle.seed),
        stage=stage,
        acc="",
        f1="",
        sensitivity="",
        specificity="",
        status="error",
        error=error,
        out_dir=str(out_dir),
    )


def _should_skip(
    *,
    rows: dict[tuple[int, str, str], ModelStageResult],
    seed: int,
    stage: str,
    resume: bool,
    retry_errors: bool,
) -> tuple[bool, str]:
    if not resume:
        return False, ""
    existing = rows.get((int(seed), "HL", stage))
    if _is_complete(existing, stage):
        return True, "complete"
    if existing is not None and existing.status == "error" and not retry_errors:
        return True, "error"
    return False, ""


def _needs_llm(
    *,
    rows: dict[tuple[int, str, str], ModelStageResult],
    seeds: tuple[int, ...],
    stages: tuple[str, ...],
    resume: bool,
    retry_errors: bool,
) -> bool:
    return any(
        not _should_skip(
            rows=rows,
            seed=int(seed),
            stage=stage,
            resume=resume,
            retry_errors=retry_errors,
        )[0]
        for seed in seeds
        for stage in stages
    )


def run_hl_experiments_v2(
    *,
    stages: tuple[str, ...] = REGIME_ORDER,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    resume: bool = False,
    retry_errors: bool = False,
    run_id: str | None = None,
) -> list[ModelStageResult]:
    settings = get_default_experiment_settings()
    rows = _read_results(RESULTS_PATH)

    # A standalone continual run must prove its Stage 1 dependency before any
    # LLM request is made. When Stage 1 is selected too, the dependency is
    # checked immediately after that stage finishes or is resumed.
    if STAGE2_CONTINUAL in stages and STAGE1_DIRECT not in stages:
        for seed in seeds:
            _require_stage1_result(rows, int(seed))

    if _needs_llm(
        rows=rows,
        seeds=seeds,
        stages=stages,
        resume=resume,
        retry_errors=retry_errors,
    ) and not os.getenv(LLM_KEY_ENV):
        raise RuntimeError(f"Environment variable {LLM_KEY_ENV} is not set")

    resolved_run_id = _validate_run_id(run_id or _new_run_id())
    llm_cfg = LLMConfig(
        base_url=LLM_BASE_URL,
        api_key_env=LLM_KEY_ENV,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    total = len(seeds) * len(stages)
    completed = 0

    for seed in seeds:
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
        _ensure_stage_data_manifest(
            seed_root=seed_root,
            dataset=settings.dataset.name,
            seed=int(seed),
            stage1=stage1,
            stage2=stage2,
            validation=validation,
        )

        for stage in stages:
            completed += 1
            skip, reason = _should_skip(
                rows=rows,
                seed=int(seed),
                stage=stage,
                resume=resume,
                retry_errors=retry_errors,
            )
            if skip:
                print(f"[{completed}/{total}] skip {reason} HL seed={seed} stage={stage}", flush=True)
                continue

            bundle = stage1 if stage == STAGE1_DIRECT else stage2
            try:
                stage1_out_dir: Path | None = None
                if stage == STAGE2_CONTINUAL:
                    _stage1_result, stage1_out_dir = _require_stage1_result(rows, int(seed))
                out_dir = _fresh_out_dir(seed_root, stage, resolved_run_id)
            except FileExistsError:
                # This is a run-level safety violation, not an endpoint failure:
                # do not write even an error marker into the occupied directory.
                raise
            except Exception as exc:
                # Dependency failures during an all-stage run are recorded for
                # this endpoint while allowing independent Stage 2 to proceed.
                fallback_out_dir = seed_root / "HL" / stage / resolved_run_id
                error = f"{type(exc).__name__}: {exc}"
                fallback_out_dir.mkdir(parents=True, exist_ok=True)
                (fallback_out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                rows[(int(seed), "HL", stage)] = _error_result(
                    bundle=bundle,
                    stage=stage,
                    error=error,
                    out_dir=fallback_out_dir,
                )
                write_results_csv(RESULTS_PATH, _ordered(rows))
                print(f"[{completed}/{total}] error HL seed={seed} stage={stage}: {error}", flush=True)
                continue

            started = time.perf_counter()
            print(f"[{completed}/{total}] fit HL seed={seed} stage={stage}", flush=True)
            try:
                if stage == STAGE1_DIRECT:
                    result = _run_stage1(stage1=stage1, llm_cfg=llm_cfg, out_dir=out_dir)
                elif stage == STAGE2_CONTINUAL:
                    assert stage1_out_dir is not None
                    result = _run_stage2_continual(
                        settings=settings,
                        stage2_template=stage2,
                        stage1_out_dir=stage1_out_dir,
                        llm_cfg=llm_cfg,
                        out_dir=out_dir,
                    )
                elif stage == STAGE2_DIRECT:
                    result = _run_stage2_direct(stage2=stage2, llm_cfg=llm_cfg, out_dir=out_dir)
                else:  # pragma: no cover - guarded by _parse_stages/public defaults
                    raise ValueError(f"Unsupported HL stage: {stage}")
                rows[_key(result)] = result
                print(
                    f"[{completed}/{total}] done HL seed={seed} stage={stage} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                rows[(int(seed), "HL", stage)] = _error_result(
                    bundle=bundle,
                    stage=stage,
                    error=error,
                    out_dir=out_dir,
                )
                print(f"[{completed}/{total}] error HL seed={seed} stage={stage}: {error}", flush=True)
            _write_json(out_dir / "elapsed.json", {"elapsed_seconds": time.perf_counter() - started})
            write_results_csv(RESULTS_PATH, _ordered(rows))

    ordered = _ordered(rows)
    write_results_csv(RESULTS_PATH, ordered)
    print(f"continuous_learning_v2_results_csv={RESULTS_PATH}", flush=True)
    return ordered


def run_hl_stage2_direct(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    resume: bool = False,
) -> list[ModelStageResult]:
    """Backward-compatible Python entry point for the former direct-only runner."""

    return run_hl_experiments_v2(stages=(STAGE2_DIRECT,), seeds=seeds, resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or more HL endpoints on the frozen V2 data splits.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=(
            "Stages to run: all, stage1, stage1-to-stage2, stage2; aliases and comma-separated values "
            "are accepted. Default: all."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action="store_true", help="Skip selected endpoints with complete artifacts.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, rerun selected endpoints whose existing result status is error.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional output leaf name. Defaults to a unique timestamp; an existing non-empty leaf is never overwritten.",
    )
    args = parser.parse_args()
    stages = _parse_stages(args.stages)
    seeds = tuple(int(seed) for seed in args.seeds)
    print(
        f"Running HL experiment on {MIMIC_CSV_PATH} with label={MIMIC_LABEL_COL}, "
        f"seeds={seeds}, stages={stages}, model={LLM_MODEL}.",
        flush=True,
    )
    run_hl_experiments_v2(
        stages=stages,
        seeds=seeds,
        resume=bool(args.resume),
        retry_errors=bool(args.retry_errors),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
