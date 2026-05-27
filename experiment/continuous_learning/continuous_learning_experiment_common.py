from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hl.continuous_learning import DriftConfig

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
ROW_ID_COL = "__continuous_row_id__"
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
MIMIC_CSV_PATH = REPO_ROOT / "data" / "merged_by_subject_id_complete_rows_without_unit_cols_renamed.csv"
MIMIC_LABEL_COL = "death_within_hosp_28days"
DEFAULT_SEEDS = (36, 40, 42)

STAGE1_FEATURE_COLS = (
    "Age",
    "Sex (M-0, F-1)",
    "White Blood Cell Count",
    "Red Blood Cell Count",
    "Platelet Count",
    "Hemoglobin",
    "Red Cell Distribution Width",
    "Hematocrit",
    "Albumin",
    "Sodium",
    "Potassium",
    "Total Calcium",
    "Chloride",
    "Blood Glucose",
    "Anion Gap",
    "pH",
    "Partial Pressure of Carbon Dioxide",
    "Partial Pressure of Oxygen",
    "Blood Lactate",
    "Total Carbon Dioxide",
    "Ionized Calcium",
    "Prothrombin Time",
    "Activated Partial Thromboplastin Time",
    "International Normalized Ratio",
    "Bilirubin",
    "Alanine Aminotransferase",
    "Aspartate Aminotransferase",
    "Blood Urea Nitrogen",
    "Creatinine",
    "Lactate Dehydrogenase",
    "SIRS",
)

STAGE2_FEATURE_COLS = (
    "Age",
    "Sex (M-0, F-1)",
    "White Blood Cell Count",
    "Red Blood Cell Count",
    "Platelet Count",
    "Hemoglobin",
    "Red Cell Distribution Width",
    "Hematocrit",
    "Albumin",
    "Sodium",
    "Potassium",
    "Total Calcium",
    "Chloride",
    "Blood Glucose",
    "Anion Gap",
    "pH",
    "Partial Pressure of Carbon Dioxide",
    "Partial Pressure of Oxygen",
    "Blood Lactate",
    "Total Carbon Dioxide",
    "Ionized Calcium",
    "Prothrombin Time",
    "Activated Partial Thromboplastin Time",
    "International Normalized Ratio",
    "Bilirubin",
    "Alanine Aminotransferase",
    "Aspartate Aminotransferase",
    "Blood Urea Nitrogen",
    "Creatinine",
    "Lactate Dehydrogenase",
    "SOFA",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    label_col: str
    prev_hl_out_dir: Path | None


@dataclass(frozen=True)
class SplitSpec:
    val_total: int
    test_total: int
    pos_value: int = 1
    neg_value: int = 0


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    train_total: int
    feature_cols: tuple[str, ...]


@dataclass(frozen=True)
class ModelStageResult:
    model: str
    dataset: str
    seed: int
    stage: str
    acc: str
    f1: str
    sensitivity: str
    specificity: str
    status: str
    error: str
    out_dir: str


@dataclass(frozen=True)
class ExperimentSettings:
    dataset: DatasetSpec
    seeds: tuple[int, ...]
    split_spec: SplitSpec
    stages: tuple[StageSpec, StageSpec]
    output_root: Path
    stage1_change_note: str
    stage2_change_note: str


@dataclass(frozen=True)
class StageDataBundle:
    dataset: str
    seed: int
    stage: str
    label_col: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    drift: DriftConfig
    drift_meta: dict
    split_meta: dict
    train_sampling_meta: dict


def get_default_experiment_settings() -> ExperimentSettings:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return ExperimentSettings(
        dataset=DatasetSpec("MIMIC", MIMIC_CSV_PATH, MIMIC_LABEL_COL, None),
        seeds=DEFAULT_SEEDS,
        split_spec=SplitSpec(val_total=500, test_total=800),
        stages=(
            StageSpec("stage1_train1000", 1000, STAGE1_FEATURE_COLS),
            StageSpec("stage2_train40", 40, STAGE2_FEATURE_COLS),
        ),
        output_root=OUTPUT_ROOT,
        stage1_change_note="Stage 1 uses the original baseline ICU feature set with SIRS available.",
        stage2_change_note=(
            "Due to changes in sepsis assessment guidelines, the SIRS index is removed and replaced by the SOFA index."
        ),
    )


def build_stage1_drift(settings: ExperimentSettings, prev_hl_out_dir: Path | None) -> DriftConfig:
    return DriftConfig(
        dropped_cols=(),
        added_cols=(),
        renamed_cols=(),
        change_note=settings.stage1_change_note,
        prev_hl_out_dir=prev_hl_out_dir,
    )


def build_stage2_drift_template(settings: ExperimentSettings) -> DriftConfig:
    return DriftConfig(
        dropped_cols=("SIRS",),
        added_cols=("SOFA",),
        renamed_cols=(),
        change_note=settings.stage2_change_note,
        prev_hl_out_dir=None,
    )


def make_stage2_drift(settings: ExperimentSettings, prev_hl_out_dir: Path) -> DriftConfig:
    template = build_stage2_drift_template(settings)
    return DriftConfig(
        dropped_cols=template.dropped_cols,
        added_cols=template.added_cols,
        renamed_cols=template.renamed_cols,
        change_note=template.change_note,
        prev_hl_out_dir=prev_hl_out_dir,
    )


def prepare_two_stage_data_bundles(
    *,
    ds: DatasetSpec,
    stage1_drift: DriftConfig,
    stage2_drift: DriftConfig,
    stage1: StageSpec,
    stage2: StageSpec,
    seed: int,
    split_spec: SplitSpec,
) -> tuple[StageDataBundle, StageDataBundle]:
    df = _load_csv(ds.csv_path)
    if ds.label_col not in df.columns:
        raise ValueError(f"{ds.name}: label_col={ds.label_col} not found")

    df = df.copy()
    df[ds.label_col] = df[ds.label_col].astype(int)
    if ROW_ID_COL in df.columns:
        raise ValueError(f"Dataset contains reserved column name: {ROW_ID_COL}")
    df[ROW_ID_COL] = np.arange(len(df), dtype=int)

    partition = _partition_two_stage_rows_balanced(
        df=df,
        label_col=ds.label_col,
        stage1=stage1,
        stage2=stage2,
        spec=split_spec,
        seed=seed,
    )
    stage1_bundle = _build_stage_bundle_from_partition(
        ds=ds,
        stage=stage1,
        drift=stage1_drift,
        seed=seed,
        split_spec=split_spec,
        raw_train_df=partition["stage1"]["train"],
        raw_val_df=partition["stage1"]["val"],
        raw_test_df=partition["stage1"]["test"],
        partition_seed=int(partition["partition_seed"]),
    )
    stage2_bundle = _build_stage_bundle_from_partition(
        ds=ds,
        stage=stage2,
        drift=stage2_drift,
        seed=seed,
        split_spec=split_spec,
        raw_train_df=partition["stage2"]["train"],
        raw_val_df=partition["stage2"]["val"],
        raw_test_df=partition["stage2"]["test"],
        partition_seed=int(partition["partition_seed"]),
    )
    return stage1_bundle, stage2_bundle


def _build_stage_bundle_from_partition(
    *,
    ds: DatasetSpec,
    stage: StageSpec,
    drift: DriftConfig,
    seed: int,
    split_spec: SplitSpec,
    raw_train_df: pd.DataFrame,
    raw_val_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    partition_seed: int,
) -> StageDataBundle:
    train_df, drift_meta = _apply_stage_view(raw_train_df, label_col=ds.label_col, drift=drift, feature_cols=stage.feature_cols)
    val_df, _ = _apply_stage_view(raw_val_df, label_col=ds.label_col, drift=drift, feature_cols=stage.feature_cols)
    test_df, _ = _apply_stage_view(raw_test_df, label_col=ds.label_col, drift=drift, feature_cols=stage.feature_cols)
    split_meta = {
        "split_seed": int(partition_seed),
        "partition_strategy": "two_stage_disjoint_balanced",
        "val_source_row_ids": _sorted_row_ids(raw_val_df),
        "test_source_row_ids": _sorted_row_ids(raw_test_df),
        "split_spec_obj": split_spec,
    }
    train_meta = {
        "sampling_seed": int(partition_seed),
        "sampling_strategy": "prepartitioned_two_stage_disjoint_balanced",
        "train_total": int(stage.train_total),
        "pos_target": int(stage.train_total // 2),
        "neg_target": int(stage.train_total // 2),
        "pos_available": int((raw_train_df[ds.label_col].astype(int) == split_spec.pos_value).sum()),
        "neg_available": int((raw_train_df[ds.label_col].astype(int) == split_spec.neg_value).sum()),
        "pos_replace": False,
        "neg_replace": False,
        "train_source_row_ids": _sorted_row_ids(raw_train_df),
    }
    return StageDataBundle(
        dataset=ds.name,
        seed=seed,
        stage=stage.stage_name,
        label_col=ds.label_col,
        train_df=_strip_row_id(train_df),
        val_df=_strip_row_id(val_df),
        test_df=_strip_row_id(test_df),
        drift=drift,
        drift_meta=drift_meta,
        split_meta=split_meta,
        train_sampling_meta=train_meta,
    )


def stage_bundle_manifest(bundle: StageDataBundle) -> dict:
    return {
        "dataset": bundle.dataset,
        "seed": int(bundle.seed),
        "stage": bundle.stage,
        "label_col": bundle.label_col,
        "feature_cols": list(bundle.train_df.columns.drop(bundle.label_col)),
        "train_shape": list(bundle.train_df.shape),
        "val_shape": list(bundle.val_df.shape),
        "test_shape": list(bundle.test_df.shape),
        "split_spec": asdict(bundle.split_meta.get("split_spec_obj")),
        "split": {k: v for k, v in bundle.split_meta.items() if k != "split_spec_obj"},
        "train_sampling": bundle.train_sampling_meta,
        "drift": bundle.drift_meta,
    }


def write_results_csv(path: Path, results: list[ModelStageResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "模型",
                "数据集",
                "seed",
                "阶段",
                "ACC",
                "F1",
                "Sensitivity",
                "Specificity",
                "status",
                "error",
                "out_dir",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "模型": result.model,
                    "数据集": result.dataset,
                    "seed": str(result.seed),
                    "阶段": result.stage,
                    "ACC": result.acc,
                    "F1": result.f1,
                    "Sensitivity": result.sensitivity,
                    "Specificity": result.specificity,
                    "status": result.status,
                    "error": result.error,
                    "out_dir": result.out_dir,
                }
            )
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _partition_two_stage_rows_balanced(
    *,
    df: pd.DataFrame,
    label_col: str,
    stage1: StageSpec,
    stage2: StageSpec,
    spec: SplitSpec,
    seed: int,
) -> dict:
    if stage1.train_total % 2 != 0 or stage2.train_total % 2 != 0:
        raise ValueError("Both stage train sizes must be even for 1:1 balanced partitioning")

    y = df[label_col].astype(int)
    pos_df = df.loc[y == spec.pos_value].copy()
    neg_df = df.loc[y == spec.neg_value].copy()
    n_stage1_train_each = stage1.train_total // 2
    n_stage2_train_each = stage2.train_total // 2
    n_val_each = spec.val_total // 2
    n_test_each = spec.test_total // 2
    need_each = n_stage1_train_each + n_val_each + n_test_each + n_stage2_train_each + n_val_each + n_test_each
    if len(pos_df) < need_each or len(neg_df) < need_each:
        raise ValueError(
            f"Not enough samples for two-stage balanced partition. Need pos>={need_each}, neg>={need_each}, "
            f"got pos={len(pos_df)}, neg={len(neg_df)}"
        )

    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(pos_df.index.to_numpy(dtype=int))
    neg_idx = rng.permutation(neg_df.index.to_numpy(dtype=int))

    def take(indices: np.ndarray, start: int, n_items: int) -> tuple[np.ndarray, int]:
        return indices[start : start + n_items], start + n_items

    def allocate_stage(indices: np.ndarray, start: int, n_train: int) -> tuple[dict[str, np.ndarray], int]:
        train_idx, start = take(indices, start, n_train)
        val_idx, start = take(indices, start, n_val_each)
        test_idx, start = take(indices, start, n_test_each)
        return {"train": train_idx, "val": val_idx, "test": test_idx}, start

    pos_stage1, pos_cursor = allocate_stage(pos_idx, 0, n_stage1_train_each)
    pos_stage2, pos_cursor = allocate_stage(pos_idx, pos_cursor, n_stage2_train_each)
    neg_stage1, neg_cursor = allocate_stage(neg_idx, 0, n_stage1_train_each)
    neg_stage2, neg_cursor = allocate_stage(neg_idx, neg_cursor, n_stage2_train_each)
    assert pos_cursor <= len(pos_idx)
    assert neg_cursor <= len(neg_idx)

    def build_split(pos_split: np.ndarray, neg_split: np.ndarray, random_state: int) -> pd.DataFrame:
        stage_df = pd.concat([pos_df.loc[pos_split], neg_df.loc[neg_split]], axis=0)
        return stage_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return {
        "partition_seed": int(seed),
        "stage1": {
            "train": build_split(pos_stage1["train"], neg_stage1["train"], seed + 11),
            "val": build_split(pos_stage1["val"], neg_stage1["val"], seed + 13),
            "test": build_split(pos_stage1["test"], neg_stage1["test"], seed + 17),
        },
        "stage2": {
            "train": build_split(pos_stage2["train"], neg_stage2["train"], seed + 19),
            "val": build_split(pos_stage2["val"], neg_stage2["val"], seed + 23),
            "test": build_split(pos_stage2["test"], neg_stage2["test"], seed + 29),
        },
    }


def _apply_stage_view(
    df: pd.DataFrame,
    *,
    label_col: str,
    drift: DriftConfig,
    feature_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    drifted_df, drift_meta = _apply_feature_drift(df, label_col=label_col, drift=drift)
    stage_df = _select_stage_columns(drifted_df, label_col=label_col, feature_cols=feature_cols)
    return stage_df, drift_meta


def _apply_feature_drift(df: pd.DataFrame, *, label_col: str, drift: DriftConfig) -> tuple[pd.DataFrame, dict]:
    out_df = df.copy()
    rename_map = {old_name: new_name for old_name, new_name in drift.renamed_cols}
    if rename_map:
        out_df = out_df.rename(columns=rename_map)

    dropped_present = [col for col in drift.dropped_cols if col in out_df.columns and col not in {label_col, ROW_ID_COL}]
    if dropped_present:
        out_df = out_df.drop(columns=dropped_present)

    added_missing = [col for col in drift.added_cols if col not in out_df.columns and col not in {label_col, ROW_ID_COL}]
    for col in added_missing:
        out_df[col] = np.nan

    if label_col not in out_df.columns:
        raise ValueError(f"label_col={label_col} missing after feature drift application")

    meta = {
        "dropped_cols": list(drift.dropped_cols),
        "dropped_present": dropped_present,
        "added_cols": list(drift.added_cols),
        "added_missing_filled_nan": added_missing,
        "renamed_cols": [{"from": old_name, "to": new_name} for old_name, new_name in drift.renamed_cols],
        "change_note": drift.change_note,
        "prev_hl_out_dir": str(drift.prev_hl_out_dir) if drift.prev_hl_out_dir is not None else "",
    }
    return out_df, meta


def _select_stage_columns(df: pd.DataFrame, *, label_col: str, feature_cols: tuple[str, ...]) -> pd.DataFrame:
    required_cols = list(feature_cols) + [label_col, ROW_ID_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required stage columns: {missing}")
    return df[required_cols].copy()


def _strip_row_id(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    if ROW_ID_COL in out_df.columns:
        out_df = out_df.drop(columns=[ROW_ID_COL])
    return out_df.reset_index(drop=True)


def _sorted_row_ids(df: pd.DataFrame) -> list[int]:
    if ROW_ID_COL not in df.columns:
        return []
    return sorted(int(x) for x in df[ROW_ID_COL].astype(int).tolist())
