"""Model-specific transfer implementations for the redesigned baseline experiment.

The experiment has three independently reported endpoints:

* direct Stage 1 training;
* Stage 1 -> Stage 2 continual adaptation;
* direct Stage 2 training.

Only the continual endpoint may consume fitted Stage 1 state.  No Stage 1 rows
are replayed while adapting to Stage 2.
"""

from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from experiment.modeling import FittedModel
from experiment.modeling.config import CONTINUOUS_MODEL_PARAMS, DEEPTAB_PARAMS, DEEPTAB_TRAINER_PARAMS

from continuous_learning_experiment_common import StageDataBundle


BASELINE_MODEL_NAMES = (
    "MLP",
    "XGBoost",
    "LightGBM",
    "EBM",
    "FT-Transformer",
    "ResNet",
)

STAGE1_DIRECT = "stage1_direct_train1000"
STAGE2_CONTINUAL = "stage2_continual_from_stage1_train40"
STAGE2_DIRECT = "stage2_direct_train40"
REGIME_ORDER = (STAGE1_DIRECT, STAGE2_CONTINUAL, STAGE2_DIRECT)

MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20
BOOSTING_PATIENCE = 20
DEEPTAB_TARGET_LR_FACTOR = 0.1


@dataclass(frozen=True)
class TrainedTrio:
    stage1: FittedModel
    continual: FittedModel
    stage2_direct: FittedModel
    manifests: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class UnionFeatureAdapter:
    """Align SIRS/SOFA drift without ever equating the two feature meanings."""

    label_col: str
    feature_columns: tuple[str, ...]
    sirs_stage1_median: float

    @classmethod
    def from_bundles(cls, stage1: StageDataBundle, stage2: StageDataBundle) -> "UnionFeatureAdapter":
        label = stage1.label_col
        features1 = [col for col in stage1.train_df.columns if col != label]
        features2 = [col for col in stage2.train_df.columns if col != label]
        if "SIRS" not in features1 or "SOFA" in features1:
            raise ValueError("Stage 1 must contain SIRS and exclude SOFA")
        if "SOFA" not in features2 or "SIRS" in features2:
            raise ValueError("Stage 2 must contain SOFA and exclude SIRS")
        union = tuple(features1 + [col for col in features2 if col not in features1])
        median = float(pd.to_numeric(stage1.train_df["SIRS"], errors="coerce").median())
        if not np.isfinite(median):
            raise ValueError("Stage 1 SIRS median is not finite")
        return cls(label_col=label, feature_columns=union, sirs_stage1_median=median)

    def stage1_view(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["SOFA"] = 0.0
        return self._ordered(out)

    def continual_stage2_view(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["SIRS"] = self.sirs_stage1_median
        return self._ordered(out)

    def direct_stage2_view(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Direct Stage 2 is forbidden from consuming a Stage 1-derived fill value.
        # The column is constant and therefore carries no information.
        out = frame.copy()
        out["SIRS"] = 0.0
        return self._ordered(out)

    def _ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = list(self.feature_columns) + [self.label_col]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"Union view is missing columns: {missing}")
        return frame[required].copy()

    def manifest(self) -> dict[str, Any]:
        return {
            "strategy": "stable_union_by_feature_name",
            "feature_columns": list(self.feature_columns),
            "stage1_added_SOFA_constant": 0.0,
            "continual_stage2_added_SIRS_stage1_train_median": self.sirs_stage1_median,
            "direct_stage2_added_SIRS_constant_without_stage1_access": 0.0,
            "true_stage2_SIRS_accessed": False,
            "SIRS_mapped_to_SOFA": False,
        }

    def direct_manifest(self) -> dict[str, Any]:
        """Schema information that contains no fitted Stage 1 statistic."""

        return {
            "strategy": "stable_union_by_predeclared_feature_name",
            "feature_columns": list(self.feature_columns),
            "direct_stage2_added_SIRS_constant": 0.0,
            "stage1_fitted_statistics_accessed": False,
            "true_stage2_SIRS_accessed": False,
            "SIRS_mapped_to_SOFA": False,
        }


class StableNumericPreprocessor:
    """Small all-numeric scaler whose individual feature statistics can evolve."""

    def __init__(self, feature_names: tuple[str, ...] | list[str]) -> None:
        self.feature_names = list(feature_names)
        self.median_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, *, center_overrides: dict[str, float] | None = None) -> "StableNumericPreprocessor":
        values = self._numeric(frame)
        medians = np.nanmedian(values, axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("Cannot fit numeric preprocessor with an all-missing feature")
        filled = np.where(np.isnan(values), medians, values)
        centers = np.mean(filled, axis=0)
        if center_overrides:
            for name, value in center_overrides.items():
                centers[self.feature_names.index(name)] = float(value)
        scales = np.std(filled, axis=0)
        scales[~np.isfinite(scales) | (scales == 0.0)] = 1.0
        self.median_ = medians.astype(float)
        self.center_ = centers.astype(float)
        self.scale_ = scales.astype(float)
        return self

    def adapted_feature(self, frame: pd.DataFrame, feature_name: str) -> "StableNumericPreprocessor":
        self._check_fitted()
        adapted = deepcopy(self)
        values = pd.to_numeric(frame[feature_name], errors="coerce").to_numpy(dtype=float)
        median = float(np.nanmedian(values))
        filled = np.where(np.isnan(values), median, values)
        center = float(np.mean(filled))
        scale = float(np.std(filled))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        idx = self.feature_names.index(feature_name)
        adapted.median_[idx] = median  # type: ignore[index]
        adapted.center_[idx] = center  # type: ignore[index]
        adapted.scale_[idx] = scale  # type: ignore[index]
        return adapted

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        values = self._numeric(frame)
        filled = np.where(np.isnan(values), self.median_, values)
        return ((filled - self.center_) / self.scale_).astype(np.float64, copy=False)

    def _numeric(self, frame: pd.DataFrame) -> np.ndarray:
        selected = frame[self.feature_names]
        converted = selected.apply(pd.to_numeric, errors="coerce")
        return converted.to_numpy(dtype=float)

    def _check_fitted(self) -> None:
        if self.median_ is None or self.center_ is None or self.scale_ is None:
            raise ValueError("StableNumericPreprocessor has not been fitted")

    def manifest(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "feature_names": self.feature_names,
            "median": self.median_.tolist(),  # type: ignore[union-attr]
            "center": self.center_.tolist(),  # type: ignore[union-attr]
            "scale": self.scale_.tolist(),  # type: ignore[union-attr]
        }


class IdentityFramePreprocessor:
    """Preserve pandas feature names for native tabular boosters."""

    def __init__(self, feature_names: tuple[str, ...] | list[str]) -> None:
        self.feature_names = list(feature_names)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.feature_names].copy()


class SafeLightGBMPreprocessor(IdentityFramePreprocessor):
    """Map clinical names to stable JSON-safe names required by LightGBM 4.7."""

    def __init__(self, feature_names: tuple[str, ...] | list[str]) -> None:
        super().__init__(feature_names)
        self.safe_feature_names = [f"feature_{index:02d}" for index in range(len(self.feature_names))]
        self.name_mapping = dict(zip(self.feature_names, self.safe_feature_names, strict=True))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = super().transform(frame)
        return out.rename(columns=self.name_mapping)


class TransferredEBM:
    """Composition of a fixed source EBM and a target residual EBM."""

    def __init__(
        self,
        source_ebm: Any,
        residual_ebm: Any,
        source_features: list[str],
        target_features: list[str],
        source_fill_values: dict[str, float],
    ) -> None:
        self.source_ebm = source_ebm
        self.residual_ebm = residual_ebm
        self.source_features = source_features
        self.target_features = target_features
        self.source_fill_values = source_fill_values
        self.classes_ = np.asarray([0, 1], dtype=int)

    def source_init_score(self, frame: pd.DataFrame) -> np.ndarray:
        source_view = pd.DataFrame(index=frame.index)
        for feature in self.source_features:
            if feature in frame.columns:
                source_view[feature] = frame[feature]
            else:
                source_view[feature] = self.source_fill_values[feature]
        score = self.source_ebm.decision_function(source_view[self.source_features])
        return np.asarray(score, dtype=float).reshape(-1)

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        init_score = self.source_init_score(frame)
        score = self.residual_ebm.decision_function(frame[self.target_features], init_score=init_score)
        return np.asarray(score, dtype=float).reshape(-1)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        init_score = self.source_init_score(frame)
        return np.asarray(
            self.residual_ebm.predict_proba(frame[self.target_features], init_score=init_score),
            dtype=float,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "strategy": "source_raw_score_plus_target_residual_ebm",
            "source_features": self.source_features,
            "target_features": self.target_features,
            "prediction_requires_init_score": True,
        }


def train_model_trio(
    model_name: str,
    stage1: StageDataBundle,
    stage2: StageDataBundle,
    *,
    work_dir: Path,
) -> TrainedTrio:
    if model_name not in BASELINE_MODEL_NAMES:
        raise ValueError(f"Unsupported redesigned baseline: {model_name}")
    if model_name == "MLP":
        return _train_mlp_trio(stage1, stage2)
    if model_name == "XGBoost":
        return _train_xgboost_trio(stage1, stage2)
    if model_name == "LightGBM":
        return _train_lightgbm_trio(stage1, stage2)
    if model_name == "EBM":
        return _train_ebm_trio(stage1, stage2)
    return _train_deeptab_trio(model_name, stage1, stage2, work_dir=work_dir)


def _train_mlp_trio(stage1: StageDataBundle, stage2: StageDataBundle) -> TrainedTrio:
    adapter = UnionFeatureAdapter.from_bundles(stage1, stage2)
    label = stage1.label_col
    train1, val1, _ = (adapter.stage1_view(frame) for frame in (stage1.train_df, stage1.val_df, stage1.test_df))
    train2c, val2c, _ = (
        adapter.continual_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )
    train2d, val2d, _ = (
        adapter.direct_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )

    source_pre = StableNumericPreprocessor(adapter.feature_columns).fit(
        train1.drop(columns=[label]),
        center_overrides={"SIRS": adapter.sirs_stage1_median, "SOFA": 0.0},
    )
    source_estimator, source_summary = _fit_mlp_epochs(
        train1,
        val1,
        label,
        seed=stage1.seed,
        preprocessor=source_pre,
    )
    source_val_x = source_pre.transform(val1.drop(columns=[label]))
    before = source_estimator.predict_proba(source_val_x)[:, 1]
    sofa_idx = list(adapter.feature_columns).index("SOFA")
    source_estimator.coefs_[0][sofa_idx, :] = 0.0
    after = source_estimator.predict_proba(source_val_x)[:, 1]
    zero_diff = float(np.max(np.abs(before - after)))
    if zero_diff > 1e-12:
        raise AssertionError(f"Zeroing the dormant MLP SOFA input changed Stage 1 predictions: {zero_diff}")
    source_hash = _hash_mlp(source_estimator)
    source = FittedModel(
        model_name="MLP",
        family="sklearn",
        estimator=source_estimator,
        preprocessor=source_pre,
        feature_names=list(adapter.feature_columns),
        training_summary={**source_summary, "dormant_SOFA_zero_max_prediction_diff": zero_diff},
    )

    target_pre = source_pre.adapted_feature(train2c.drop(columns=[label]), "SOFA")
    target_estimator, continual_summary = _fit_mlp_epochs(
        train2c,
        val2c,
        label,
        seed=stage1.seed + 1,
        preprocessor=target_pre,
        initial_estimator=source_estimator,
    )
    initial_hash = continual_summary.pop("initial_model_hash")
    if initial_hash != source_hash:
        raise AssertionError("MLP continual training did not start from the exact Stage 1 parameters")
    continual = FittedModel(
        model_name="MLP",
        family="sklearn",
        estimator=target_estimator,
        preprocessor=target_pre,
        feature_names=list(adapter.feature_columns),
        training_summary=continual_summary,
    )

    direct_pre = StableNumericPreprocessor(adapter.feature_columns).fit(
        train2d.drop(columns=[label]), center_overrides={"SIRS": 0.0}
    )
    direct_estimator, direct_summary = _fit_mlp_epochs(
        train2d,
        val2d,
        label,
        seed=stage1.seed + 1,
        preprocessor=direct_pre,
    )
    direct = FittedModel(
        model_name="MLP",
        family="sklearn",
        estimator=direct_estimator,
        preprocessor=direct_pre,
        feature_names=list(adapter.feature_columns),
        training_summary=direct_summary,
    )
    return TrainedTrio(
        stage1=source,
        continual=continual,
        stage2_direct=direct,
        manifests={
            STAGE1_DIRECT: {
                "schema": adapter.manifest(),
                "preprocessor": source_pre.manifest(),
                "final_model_hash": source_hash,
            },
            STAGE2_CONTINUAL: {
                "continuation_strategy": "mlp_partial_fit_parameter_transfer",
                "data_replay": False,
                "schema": adapter.manifest(),
                "source_model_hash": source_hash,
                "initial_target_model_hash": initial_hash,
                "final_model_hash": _hash_mlp(target_estimator),
                "preprocessor": target_pre.manifest(),
            },
            STAGE2_DIRECT: {
                "continuation_strategy": "none_random_initialization",
                "stage1_state_accessed": False,
                "schema": adapter.direct_manifest(),
                "preprocessor": direct_pre.manifest(),
                "final_model_hash": _hash_mlp(direct_estimator),
            },
        },
    )


def _new_mlp(seed: int, train_size: int) -> MLPClassifier:
    params = dict(CONTINUOUS_MODEL_PARAMS["MLP"])
    params["batch_size"] = min(64, max(2, int(train_size)))
    return MLPClassifier(random_state=seed, **params)


def _fit_mlp_epochs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    *,
    seed: int,
    preprocessor: StableNumericPreprocessor,
    initial_estimator: MLPClassifier | None = None,
) -> tuple[MLPClassifier, dict[str, Any]]:
    x_train = preprocessor.transform(train_df.drop(columns=[label_col]))
    y_train = train_df[label_col].astype(int).to_numpy()
    x_val = preprocessor.transform(val_df.drop(columns=[label_col]))
    y_val = val_df[label_col].astype(int).to_numpy()
    estimator = deepcopy(initial_estimator) if initial_estimator is not None else _new_mlp(seed, len(train_df))
    estimator.batch_size = min(64, max(2, int(len(train_df))))
    initial_hash = _hash_mlp(estimator) if initial_estimator is not None else ""
    best_estimator: MLPClassifier | None = None
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    started = time.perf_counter()

    if initial_estimator is not None:
        best_loss = float(log_loss(y_val, estimator.predict_proba(x_val), labels=[0, 1]))
        best_estimator = deepcopy(estimator)

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        if hasattr(estimator, "classes_"):
            estimator.partial_fit(x_train, y_train)
        else:
            estimator.partial_fit(x_train, y_train, classes=np.asarray([0, 1], dtype=int))
        current_loss = float(log_loss(y_val, estimator.predict_proba(x_val), labels=[0, 1]))
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_epoch = epoch
            best_estimator = deepcopy(estimator)
            wait = 0
        else:
            wait += 1
        if wait >= MLP_PATIENCE:
            break
    if best_estimator is None:
        raise RuntimeError("MLP validation selection did not produce a fitted model")
    summary = {
        "seed": int(seed),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "max_epochs": MLP_MAX_EPOCHS,
        "epochs_run": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_log_loss": best_loss,
        "patience": MLP_PATIENCE,
        "termination": "early_stopping" if epoch < MLP_MAX_EPOCHS else "max_epochs",
        "elapsed_seconds": float(time.perf_counter() - started),
        "automatic_balance": False,
        "sample_weight": None,
    }
    if initial_hash:
        summary["initial_model_hash"] = initial_hash
    return best_estimator, summary


def _hash_mlp(estimator: MLPClassifier) -> str:
    digest = hashlib.sha256()
    if not hasattr(estimator, "coefs_"):
        digest.update(b"unfitted")
        return digest.hexdigest()
    for array in [*estimator.coefs_, *estimator.intercepts_]:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _train_xgboost_trio(stage1: StageDataBundle, stage2: StageDataBundle) -> TrainedTrio:
    from xgboost import XGBClassifier

    adapter = UnionFeatureAdapter.from_bundles(stage1, stage2)
    label = stage1.label_col
    train1, val1, _ = (adapter.stage1_view(frame) for frame in (stage1.train_df, stage1.val_df, stage1.test_df))
    train2c, val2c, _ = (
        adapter.continual_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )
    train2d, val2d, _ = (
        adapter.direct_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )
    params = dict(CONTINUOUS_MODEL_PARAMS["XGBoost"])
    params["early_stopping_rounds"] = BOOSTING_PATIENCE

    source_estimator = XGBClassifier(random_state=stage1.seed, **params)
    source_summary = _fit_xgb(source_estimator, train1, val1, label)
    selected_source_booster = _selected_xgb_booster(source_estimator)
    source_hash = hashlib.sha256(bytes(selected_source_booster.save_raw(raw_format="json"))).hexdigest()
    identity = IdentityFramePreprocessor(adapter.feature_columns)
    source = FittedModel(
        model_name="XGBoost", family="sklearn", estimator=source_estimator, preprocessor=identity,
        feature_names=list(adapter.feature_columns), training_summary=source_summary,
    )

    continual_estimator = XGBClassifier(random_state=stage1.seed + 1, **params)
    continual_summary = _fit_xgb(
        continual_estimator, train2c, val2c, label, xgb_model=selected_source_booster
    )
    continual = FittedModel(
        model_name="XGBoost", family="sklearn", estimator=continual_estimator, preprocessor=identity,
        feature_names=list(adapter.feature_columns), training_summary=continual_summary,
    )

    direct_params = {**params, "n_estimators": 400}
    direct_estimator = XGBClassifier(random_state=stage1.seed + 1, **direct_params)
    direct_summary = _fit_xgb(direct_estimator, train2d, val2d, label)
    direct = FittedModel(
        model_name="XGBoost", family="sklearn", estimator=direct_estimator, preprocessor=identity,
        feature_names=list(adapter.feature_columns), training_summary=direct_summary,
    )
    final_booster = continual_estimator.get_booster()
    return TrainedTrio(
        stage1=source,
        continual=continual,
        stage2_direct=direct,
        manifests={
            STAGE1_DIRECT: {"schema": adapter.manifest(), "selected_source_booster_hash": source_hash},
            STAGE2_CONTINUAL: {
                "continuation_strategy": "xgboost_xgb_model",
                "data_replay": False,
                "schema": adapter.manifest(),
                "selected_source_booster_hash": source_hash,
                "final_boosted_rounds": int(final_booster.num_boosted_rounds()),
                "feature_names_equal": list(final_booster.feature_names or []) == list(adapter.feature_columns),
            },
            STAGE2_DIRECT: {
                "continuation_strategy": "none_random_initialization",
                "stage1_state_accessed": False,
                "capacity_matched_n_estimators": 400,
                "schema": adapter.direct_manifest(),
            },
        },
    )


def _fit_xgb(
    estimator: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    *,
    xgb_model: Any = None,
) -> dict[str, Any]:
    features = [col for col in train_df.columns if col != label_col]
    started = time.perf_counter()
    estimator.fit(
        train_df[features],
        train_df[label_col].astype(int).to_numpy(),
        eval_set=[(val_df[features], val_df[label_col].astype(int).to_numpy())],
        verbose=False,
        xgb_model=xgb_model,
    )
    booster = estimator.get_booster()
    return {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "best_iteration": int(getattr(estimator, "best_iteration", booster.num_boosted_rounds() - 1)),
        "best_score": float(getattr(estimator, "best_score", np.nan)),
        "boosted_rounds_in_artifact": int(booster.num_boosted_rounds()),
        "early_stopping_rounds": BOOSTING_PATIENCE,
        "elapsed_seconds": float(time.perf_counter() - started),
        "automatic_balance": False,
        "sample_weight": None,
    }


def _selected_xgb_booster(estimator: Any) -> Any:
    booster = estimator.get_booster()
    best_iteration = getattr(estimator, "best_iteration", None)
    if best_iteration is None or int(best_iteration) + 1 >= booster.num_boosted_rounds():
        return booster.copy()
    return booster[: int(best_iteration) + 1]


def _train_lightgbm_trio(stage1: StageDataBundle, stage2: StageDataBundle) -> TrainedTrio:
    from lightgbm import LGBMClassifier

    adapter = UnionFeatureAdapter.from_bundles(stage1, stage2)
    label = stage1.label_col
    train1, val1, _ = (adapter.stage1_view(frame) for frame in (stage1.train_df, stage1.val_df, stage1.test_df))
    train2c, val2c, _ = (
        adapter.continual_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )
    train2d, val2d, _ = (
        adapter.direct_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )
    params = dict(CONTINUOUS_MODEL_PARAMS["LightGBM"])
    preprocessor = SafeLightGBMPreprocessor(adapter.feature_columns)

    def safe_view(frame: pd.DataFrame) -> pd.DataFrame:
        out = preprocessor.transform(frame.drop(columns=[label]))
        out[label] = frame[label].astype(int).to_numpy()
        return out

    train1, val1 = safe_view(train1), safe_view(val1)
    train2c, val2c = safe_view(train2c), safe_view(val2c)
    train2d, val2d = safe_view(train2d), safe_view(val2d)

    source_estimator = LGBMClassifier(random_state=stage1.seed, **params)
    source_summary = _fit_lgb(source_estimator, train1, val1, label)
    source_hash = hashlib.sha256(source_estimator.booster_.model_to_string().encode("utf-8")).hexdigest()
    source = FittedModel(
        model_name="LightGBM", family="sklearn", estimator=source_estimator, preprocessor=preprocessor,
        feature_names=list(adapter.feature_columns), training_summary=source_summary,
    )

    continual_params = {**params, "n_estimators": 200}
    continual_estimator = LGBMClassifier(random_state=stage1.seed + 1, **continual_params)
    continual_summary = _fit_lgb(
        continual_estimator, train2c, val2c, label, init_model=source_estimator.booster_
    )
    continual = FittedModel(
        model_name="LightGBM", family="sklearn", estimator=continual_estimator, preprocessor=preprocessor,
        feature_names=list(adapter.feature_columns), training_summary=continual_summary,
    )

    direct_params = {**params, "n_estimators": 500}
    direct_estimator = LGBMClassifier(random_state=stage1.seed + 1, **direct_params)
    direct_summary = _fit_lgb(direct_estimator, train2d, val2d, label)
    direct = FittedModel(
        model_name="LightGBM", family="sklearn", estimator=direct_estimator, preprocessor=preprocessor,
        feature_names=list(adapter.feature_columns), training_summary=direct_summary,
    )
    return TrainedTrio(
        stage1=source,
        continual=continual,
        stage2_direct=direct,
        manifests={
            STAGE1_DIRECT: {
                "schema": adapter.manifest(),
                "lightgbm_safe_feature_name_mapping": preprocessor.name_mapping,
                "source_booster_hash": source_hash,
            },
            STAGE2_CONTINUAL: {
                "continuation_strategy": "lightgbm_init_model",
                "data_replay": False,
                "schema": adapter.manifest(),
                "lightgbm_safe_feature_name_mapping": preprocessor.name_mapping,
                "source_booster_hash": source_hash,
                "final_num_trees": int(continual_estimator.booster_.num_trees()),
                "feature_names_equal": (
                    continual_estimator.booster_.feature_name() == source_estimator.booster_.feature_name()
                ),
            },
            STAGE2_DIRECT: {
                "continuation_strategy": "none_random_initialization",
                "stage1_state_accessed": False,
                "capacity_matched_n_estimators": 500,
                "schema": adapter.direct_manifest(),
                "lightgbm_safe_feature_name_mapping": preprocessor.name_mapping,
            },
        },
    )


def _fit_lgb(
    estimator: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    *,
    init_model: Any = None,
) -> dict[str, Any]:
    import lightgbm as lgb

    features = [col for col in train_df.columns if col != label_col]
    started = time.perf_counter()
    estimator.fit(
        train_df[features],
        train_df[label_col].astype(int).to_numpy(),
        eval_X=val_df[features],
        eval_y=val_df[label_col].astype(int).to_numpy(),
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(BOOSTING_PATIENCE, verbose=False), lgb.log_evaluation(0)],
        init_model=init_model,
    )
    return {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "best_iteration": int(estimator.best_iteration_ or estimator.booster_.num_trees()),
        "num_trees_in_artifact": int(estimator.booster_.num_trees()),
        "early_stopping_rounds": BOOSTING_PATIENCE,
        "elapsed_seconds": float(time.perf_counter() - started),
        "automatic_balance": False,
        "sample_weight": None,
    }


def _train_ebm_trio(stage1: StageDataBundle, stage2: StageDataBundle) -> TrainedTrio:
    from interpret.glassbox import ExplainableBoostingClassifier

    label = stage1.label_col
    features1 = [col for col in stage1.train_df.columns if col != label]
    features2 = [col for col in stage2.train_df.columns if col != label]
    fill_values = {
        feature: float(pd.to_numeric(stage1.train_df[feature], errors="coerce").median())
        for feature in features1
    }
    params = {"n_jobs": 1}

    started = time.perf_counter()
    source_estimator = ExplainableBoostingClassifier(random_state=stage1.seed, **params)
    source_estimator.fit(stage1.train_df[features1], stage1.train_df[label].astype(int).to_numpy())
    source_summary = {
        "train_rows": int(len(stage1.train_df)),
        "val_rows_reserved": int(len(stage1.val_df)),
        "validation_strategy": "EBM internal validation; external val not used for fitting",
        "elapsed_seconds": float(time.perf_counter() - started),
        "automatic_balance": False,
        "sample_weight": None,
    }
    source = FittedModel(
        model_name="EBM", family="ebm", estimator=source_estimator, preprocessor=None,
        feature_names=features1, training_summary=source_summary,
    )

    def source_view(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for feature in features1:
            out[feature] = frame[feature] if feature in frame.columns else fill_values[feature]
        return out[features1]

    source_train_score = np.asarray(source_estimator.decision_function(source_view(stage2.train_df)), dtype=float)
    source_val_score = np.asarray(source_estimator.decision_function(source_view(stage2.val_df)), dtype=float)
    source_test_score = np.asarray(source_estimator.decision_function(source_view(stage2.test_df)), dtype=float)
    started = time.perf_counter()
    residual_estimator = ExplainableBoostingClassifier(random_state=stage1.seed + 1, **params)
    residual_estimator.fit(
        stage2.train_df[features2],
        stage2.train_df[label].astype(int).to_numpy(),
        init_score=source_train_score,
    )
    composite = TransferredEBM(
        source_estimator, residual_estimator, features1, features2, fill_values
    )
    continual_summary = {
        "train_rows": int(len(stage2.train_df)),
        "val_rows_reserved": int(len(stage2.val_df)),
        "validation_strategy": "EBM internal validation; external val not used for fitting",
        "elapsed_seconds": float(time.perf_counter() - started),
        "source_train_raw_score_range": [float(source_train_score.min()), float(source_train_score.max())],
        "source_val_raw_score_range": [float(source_val_score.min()), float(source_val_score.max())],
        "source_test_raw_score_range": [float(source_test_score.min()), float(source_test_score.max())],
        "automatic_balance": False,
        "sample_weight": None,
    }
    continual = FittedModel(
        model_name="EBM", family="ebm", estimator=composite, preprocessor=None,
        feature_names=features2, training_summary=continual_summary,
    )

    started = time.perf_counter()
    direct_estimator = ExplainableBoostingClassifier(random_state=stage1.seed + 1, **params)
    direct_estimator.fit(stage2.train_df[features2], stage2.train_df[label].astype(int).to_numpy())
    direct_summary = {
        "train_rows": int(len(stage2.train_df)),
        "val_rows_reserved": int(len(stage2.val_df)),
        "validation_strategy": "EBM internal validation; external val not used for fitting",
        "elapsed_seconds": float(time.perf_counter() - started),
        "automatic_balance": False,
        "sample_weight": None,
    }
    direct = FittedModel(
        model_name="EBM", family="ebm", estimator=direct_estimator, preprocessor=None,
        feature_names=features2, training_summary=direct_summary,
    )
    return TrainedTrio(
        stage1=source,
        continual=continual,
        stage2_direct=direct,
        manifests={
            STAGE1_DIRECT: {"feature_columns": features1},
            STAGE2_CONTINUAL: {
                "continuation_strategy": "ebm_init_score_residual",
                "data_replay": False,
                "source_feature_columns": features1,
                "target_feature_columns": features2,
                "source_fill_values_from_stage1_train": fill_values,
                "fit_uses_source_raw_init_score": True,
                "prediction_uses_source_raw_init_score": True,
                "true_stage2_SIRS_accessed": False,
            },
            STAGE2_DIRECT: {
                "continuation_strategy": "none_random_initialization",
                "stage1_state_accessed": False,
                "feature_columns": features2,
            },
        },
    )


def _train_deeptab_trio(
    model_name: str,
    stage1: StageDataBundle,
    stage2: StageDataBundle,
    *,
    work_dir: Path,
) -> TrainedTrio:
    adapter = UnionFeatureAdapter.from_bundles(stage1, stage2)
    label = stage1.label_col
    train1, val1, _ = (adapter.stage1_view(frame) for frame in (stage1.train_df, stage1.val_df, stage1.test_df))
    train2c, val2c, _ = (
        adapter.continual_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )
    train2d, val2d, _ = (
        adapter.direct_stage2_view(frame) for frame in (stage2.train_df, stage2.val_df, stage2.test_df)
    )
    source, source_summary = _fit_deeptab(
        model_name, train1, val1, label, seed=stage1.seed,
        lr=float(DEEPTAB_PARAMS[model_name]["lr"]), initial_state=None, work_dir=work_dir,
    )
    zero_diff = _zero_inactive_deeptab_feature(source, model_name, val1, label, "SOFA")
    source.training_summary.update(source_summary)
    source.training_summary["dormant_SOFA_zero_max_prediction_diff"] = zero_diff
    source_state = _cpu_state_dict(source.estimator._task_model.state_dict())
    source_hash = _hash_torch_state(source_state)

    continual, continual_summary = _fit_deeptab(
        model_name,
        train2c,
        val2c,
        label,
        seed=stage1.seed + 1,
        lr=float(DEEPTAB_PARAMS[model_name]["lr"]) * DEEPTAB_TARGET_LR_FACTOR,
        initial_state=source_state,
        work_dir=work_dir,
    )
    continual.training_summary.update(continual_summary)
    initial_hash = continual_summary["initial_model_hash"]
    if initial_hash != source_hash:
        raise AssertionError(f"{model_name} did not start Stage 2 from the exact Stage 1 state")

    direct, direct_summary = _fit_deeptab(
        model_name, train2d, val2d, label, seed=stage1.seed + 1,
        lr=float(DEEPTAB_PARAMS[model_name]["lr"]), initial_state=None, work_dir=work_dir,
    )
    direct.training_summary.update(direct_summary)
    return TrainedTrio(
        stage1=source,
        continual=continual,
        stage2_direct=direct,
        manifests={
            STAGE1_DIRECT: {
                "schema": adapter.direct_manifest(),
                "final_model_state_hash": source_hash,
                "dormant_SOFA_zero_max_prediction_diff": zero_diff,
                "checkpointing": "disabled_in_memory_best_state",
            },
            STAGE2_CONTINUAL: {
                "continuation_strategy": "deeptab_in_memory_state_transfer_and_finetune",
                "data_replay": False,
                "schema": adapter.manifest(),
                "source_model_state_hash": source_hash,
                "initial_target_model_state_hash": initial_hash,
                "final_model_state_hash": _hash_torch_state(
                    _cpu_state_dict(continual.estimator._task_model.state_dict())
                ),
                "target_lr_factor": DEEPTAB_TARGET_LR_FACTOR,
                "checkpointing": "disabled_in_memory_best_state",
            },
            STAGE2_DIRECT: {
                "continuation_strategy": "none_random_initialization",
                "stage1_state_accessed": False,
                "schema": adapter.manifest(),
                "final_model_state_hash": _hash_torch_state(
                    _cpu_state_dict(direct.estimator._task_model.state_dict())
                ),
                "checkpointing": "disabled_in_memory_best_state",
            },
        },
    )


def _new_deeptab_estimator(model_name: str, seed: int, train_size: int, lr: float) -> Any:
    from deeptab.configs import FTTransformerConfig, PreprocessingConfig, ResNetConfig, TrainerConfig
    from deeptab.models import FTTransformerClassifier, ResNetClassifier

    batch_size = max(2, min(128, int(train_size)))
    trainer_config = TrainerConfig(
        **{
            **DEEPTAB_TRAINER_PARAMS,
            "batch_size": batch_size,
            "lr": lr,
            # This value is required by the config but the custom Lightning
            # trainer below has checkpointing disabled and never creates it.
            "checkpoint_path": "unused_in_memory_checkpoint",
        }
    )
    preprocessing_config = PreprocessingConfig(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        treat_all_integers_as_numerical=True,
    )
    if model_name == "FT-Transformer":
        return FTTransformerClassifier(
            model_config=FTTransformerConfig(**DEEPTAB_PARAMS[model_name]["model"]),
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=seed,
        )
    return ResNetClassifier(
        model_config=ResNetConfig(**DEEPTAB_PARAMS[model_name]["model"]),
        preprocessing_config=preprocessing_config,
        trainer_config=trainer_config,
        random_state=seed,
    )


def _fit_deeptab(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    *,
    seed: int,
    lr: float,
    initial_state: dict[str, Any] | None,
    work_dir: Path,
) -> tuple[FittedModel, dict[str, Any]]:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping

    features = [col for col in train_df.columns if col != label_col]
    x_train = train_df[features]
    y_train = train_df[label_col].astype(int).to_numpy()
    x_val = val_df[features]
    y_val = val_df[label_col].astype(int).to_numpy()
    estimator = _new_deeptab_estimator(model_name, seed, len(train_df), lr)
    estimator.build_model(
        x_train,
        y_train,
        X_val=x_val,
        y_val=y_val,
        random_state=seed,
        batch_size=max(2, min(128, len(train_df))),
        shuffle=True,
        stratify=True,
        lr=lr,
        weight_decay=float(DEEPTAB_TRAINER_PARAMS["weight_decay"]),
        class_weight=None,
        balanced_sampler=False,
        sample_weight=None,
        loss_fct="bce",
    )
    initial_hash = ""
    if initial_state is not None:
        estimator._task_model.load_state_dict(initial_state, strict=True)
        initial_hash = _hash_torch_state(_cpu_state_dict(estimator._task_model.state_dict()))

    best_callback = _InMemoryBestState()
    early_stopping = EarlyStopping(
        monitor=str(DEEPTAB_TRAINER_PARAMS["monitor"]),
        mode=str(DEEPTAB_TRAINER_PARAMS["mode"]),
        patience=int(DEEPTAB_TRAINER_PARAMS["patience"]),
        min_delta=0.0,
        verbose=False,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=int(DEEPTAB_TRAINER_PARAMS["max_epochs"]),
        callbacks=[early_stopping, best_callback],
        accelerator="auto",
        devices=1,
        precision="32-true",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(work_dir),
    )
    estimator._task_model.train()
    estimator._task_model.estimator.train()
    started = time.perf_counter()
    trainer.fit(estimator._task_model, datamodule=estimator._data_module)
    if best_callback.best_state is None:
        raise RuntimeError(f"{model_name} did not record an in-memory validation state")
    estimator._task_model.load_state_dict(best_callback.best_state, strict=True)
    estimator._task_model.eval()
    estimator._estimator = estimator._task_model.estimator
    estimator._trainer = trainer
    estimator._best_model_path = None
    estimator.is_fitted_ = True
    summary: dict[str, Any] = {
        "seed": int(seed),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "max_epochs": int(DEEPTAB_TRAINER_PARAMS["max_epochs"]),
        "epochs_run": int(trainer.current_epoch),
        "best_epoch": int(best_callback.best_epoch),
        "best_val_loss": float(best_callback.best_score),
        "patience": int(DEEPTAB_TRAINER_PARAMS["patience"]),
        "learning_rate": float(lr),
        "termination": "early_stopping" if int(early_stopping.stopped_epoch) > 0 else "max_epochs",
        "elapsed_seconds": float(time.perf_counter() - started),
        "automatic_balance": False,
        "sample_weight": None,
        "checkpoint_created": False,
    }
    if initial_hash:
        summary["initial_model_hash"] = initial_hash
    fitted = FittedModel(
        model_name=model_name,
        family="deeptab",
        estimator=estimator,
        preprocessor=None,
        feature_names=features,
        training_summary=summary.copy(),
    )
    return fitted, summary


class _InMemoryBestState(Callback):
    """Lightning callback that restores the best validation state without disk I/O."""

    def __init__(self) -> None:
        super().__init__()
        self.best_score = float("inf")
        self.best_epoch = -1
        self.best_state: dict[str, Any] | None = None

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("val_loss")
        if metric is None:
            return
        score = float(metric.detach().cpu())
        if np.isfinite(score) and score < self.best_score:
            self.best_score = score
            self.best_epoch = int(trainer.current_epoch)
            self.best_state = _cpu_state_dict(pl_module.state_dict())


def _zero_inactive_deeptab_feature(
    fitted: FittedModel,
    model_name: str,
    val_df: pd.DataFrame,
    label_col: str,
    feature_name: str,
) -> float:
    import torch

    estimator = fitted.estimator
    num_features = list(estimator._data_module.num_feature_info)
    cat_features = list(estimator._data_module.cat_feature_info)
    if cat_features or num_features != fitted.feature_names:
        raise ValueError(
            f"Expected an all-numeric DeepTab schema in input order; num={num_features}, cat={cat_features}"
        )
    idx = num_features.index(feature_name)
    before = estimator.predict_proba(val_df[fitted.feature_names])[:, 1]
    architecture = estimator._task_model.estimator
    with torch.no_grad():
        if model_name == "FT-Transformer":
            module = architecture.embedding_layer.num_embeddings[idx]
            for parameter in module.parameters():
                parameter.zero_()
        elif bool(getattr(architecture.hparams, "use_embeddings", False)):
            module = architecture.embedding_layer.num_embeddings[idx]
            for parameter in module.parameters():
                parameter.zero_()
        else:
            architecture.initial_layer.weight[:, idx].zero_()
    after = estimator.predict_proba(val_df[fitted.feature_names])[:, 1]
    max_diff = float(np.max(np.abs(before - after)))
    if max_diff > 1e-6:
        raise AssertionError(
            f"Zeroing dormant {model_name}/{feature_name} parameters changed Stage 1 predictions: {max_diff}"
        )
    return max_diff


def _cpu_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _hash_torch_state(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def frames_for_model(
    model_name: str,
    stage1: StageDataBundle,
    stage2: StageDataBundle,
) -> dict[str, pd.DataFrame]:
    """Return the exact test view consumed by each fitted endpoint."""

    if model_name == "EBM":
        return {
            STAGE1_DIRECT: stage1.test_df.copy(),
            STAGE2_CONTINUAL: stage2.test_df.copy(),
            STAGE2_DIRECT: stage2.test_df.copy(),
        }
    adapter = UnionFeatureAdapter.from_bundles(stage1, stage2)
    return {
        STAGE1_DIRECT: adapter.stage1_view(stage1.test_df),
        STAGE2_CONTINUAL: adapter.continual_stage2_view(stage2.test_df),
        STAGE2_DIRECT: adapter.direct_stage2_view(stage2.test_df),
    }
