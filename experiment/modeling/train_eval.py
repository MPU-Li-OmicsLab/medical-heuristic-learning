"""普通对比模型的共享训练、预测和评估逻辑。

预处理器只在训练集上拟合，验证集只用于 DeepTab 的模型选择，测试集只做
最终评估。所有指标统一调用 hl.metrics.compute_metrics。
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hl.metrics import compute_metrics

from .models import build_model


@dataclass
class FittedModel:
    """Small container for a fitted estimator and its train-only transforms."""

    model_name: str
    family: str
    estimator: Any
    preprocessor: Any = None
    feature_names: list[str] = field(default_factory=list)
    training_summary: dict[str, Any] = field(default_factory=dict)


class CorelsBinarizer:
    """Train-only conversion from mixed tabular columns to binary predicates."""

    def __init__(self) -> None:
        self.specs: list[dict[str, Any]] = []
        self.feature_names_: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "CorelsBinarizer":
        candidates: list[dict[str, Any]] = []
        for col in frame.columns:
            series = frame[col]
            missing = series.isna()
            if missing.any():
                candidates.append({"column": col, "kind": "missing", "name": f"{col} is missing"})
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                values = np.sort(numeric.dropna().unique())
                if len(values) <= 2 and set(float(x) for x in values).issubset({0.0, 1.0}):
                    candidates.append({"column": col, "kind": "binary", "name": f"{col} = 1"})
                    continue
                cuts = sorted(set(float(x) for x in numeric.quantile([0.25, 0.5, 0.75]).dropna().tolist()))
                if not cuts:
                    continue
                lower = -math.inf
                for idx, upper in enumerate(cuts + [math.inf]):
                    if idx == 0:
                        name = f"{col} <= {upper:.8g}"
                    elif math.isinf(upper):
                        name = f"{col} > {lower:.8g}"
                    else:
                        name = f"{lower:.8g} < {col} <= {upper:.8g}"
                    candidates.append({"column": col, "kind": "interval", "lower": lower, "upper": upper, "name": name})
                    lower = upper
            else:
                for value in sorted(series.dropna().astype(str).unique().tolist()):
                    candidates.append({"column": col, "kind": "category", "value": value, "name": f"{col} = {value}"})

        self.specs = candidates
        matrix = self.transform(frame)
        keep = [idx for idx in range(matrix.shape[1]) if 0 < int(matrix[:, idx].sum()) < len(frame)]
        self.specs = [self.specs[idx] for idx in keep]
        self.feature_names_ = [str(spec["name"]) for spec in self.specs]
        if not self.specs:
            raise ValueError("CORELS binarization produced no non-constant predicates")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for spec in self.specs:
            series = frame[spec["column"]]
            kind = spec["kind"]
            if kind == "missing":
                values = series.isna().to_numpy()
            elif kind == "binary":
                values = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() == 1
            elif kind == "category":
                values = series.astype(str).to_numpy() == str(spec["value"])
            else:
                numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                lower = float(spec["lower"])
                upper = float(spec["upper"])
                values = np.isfinite(numeric)
                if not math.isinf(lower):
                    values &= numeric > lower
                if not math.isinf(upper):
                    values &= numeric <= upper
            columns.append(np.asarray(values, dtype=np.uint8))
        if not columns:
            return np.zeros((len(frame), 0), dtype=np.uint8)
        matrix = np.column_stack(columns).astype(np.uint8, copy=False)
        if not np.isin(matrix, [0, 1]).all():
            raise ValueError("CORELS matrix contains values outside {0, 1}")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def manifest(self) -> list[dict[str, Any]]:
        def finite_or_text(value: Any) -> Any:
            if isinstance(value, float) and math.isinf(value):
                return "-inf" if value < 0 else "inf"
            return value

        return [{key: finite_or_text(value) for key, value in spec.items()} for spec in self.specs]


def _is_categorical(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [col for col in frame.columns if _is_categorical(frame[col])]
    num_cols = [col for col in frame.columns if col not in cat_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    seed: int,
    *,
    checkpoint_dir: Path | None = None,
    variant: str = "standard",
    stage: int = 1,
    continue_from: FittedModel | None = None,
) -> FittedModel:
    """Fit one model without sample weights or model-level class balancing."""

    x_train = train_df.drop(columns=[label_col]).copy()
    y_train = train_df[label_col].astype(int).to_numpy()
    x_val = val_df.drop(columns=[label_col]).copy()
    y_val = val_df[label_col].astype(int).to_numpy()
    labels, counts = np.unique(y_train, return_counts=True)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"Training labels must contain 0 and 1, got {labels.tolist()}")

    family, estimator = build_model(
        model_name,
        seed,
        train_size=len(train_df),
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        stage=stage,
    )
    if continue_from is not None and model_name in {"LogisticRegression", "MLP"}:
        if continue_from.model_name != model_name or continue_from.family != "sklearn":
            raise ValueError(f"Cannot continue {model_name} from {continue_from.model_name}/{continue_from.family}")
        estimator = continue_from.estimator
    started = time.perf_counter()
    preprocessor: Any = None

    if family == "deeptab":
        deep_checkpoint_dir = checkpoint_dir or Path("model_checkpoints")
        deep_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        estimator.fit(
            x_train,
            y_train,
            X_val=x_val,
            y_val=y_val,
            random_state=seed,
            class_weight=None,
            balanced_sampler=False,
            sample_weight=None,
            loss_fct="bce",
            checkpoint_path=str(checkpoint_dir or Path("model_checkpoints")),
            # DeepTab 2.0 currently constructs Lightning's ModelCheckpoint with
            # dirpath=None, so Lightning resolves it from default_root_dir.
            default_root_dir=str(deep_checkpoint_dir.parent),
            accelerator="auto",
            devices=1,
            precision="32-true",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
    elif family == "corels":
        preprocessor = CorelsBinarizer()
        x_binary = preprocessor.fit_transform(x_train)
        estimator.fit(
            x_binary,
            y_train.astype(np.uint8),
            features=preprocessor.feature_names_,
            prediction_name=label_col,
        )
    elif family == "ebm":
        estimator.fit(x_train, y_train)
    else:
        if continue_from is not None and model_name in {"LogisticRegression", "MLP"}:
            preprocessor = continue_from.preprocessor
            x_transformed = preprocessor.transform(x_train)
        else:
            preprocessor = _build_preprocessor(x_train)
            x_transformed = preprocessor.fit_transform(x_train)
        fit_kwargs: dict[str, Any] = {}
        if family == "aplr":
            fit_kwargs["X_names"] = list(preprocessor.get_feature_names_out())
        if continue_from is not None and model_name == "XGBoost":
            fit_kwargs["xgb_model"] = continue_from.estimator.get_booster()
        if continue_from is not None and model_name == "LightGBM":
            fit_kwargs["init_model"] = getattr(continue_from.estimator, "booster_", None)
        estimator.fit(x_transformed, y_train, **fit_kwargs)

    elapsed = time.perf_counter() - started
    summary = {
        "seed": int(seed),
        "train_rows": int(len(train_df)),
        "class_counts": {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)},
        "automatic_balance": False,
        "sample_weight": None,
        "elapsed_seconds": float(elapsed),
        "best_epoch": _best_epoch(estimator),
        "termination": _termination_reason(estimator, family),
    }
    return FittedModel(
        model_name=model_name,
        family=family,
        estimator=estimator,
        preprocessor=preprocessor,
        feature_names=list(x_train.columns),
        training_summary=summary,
    )


def _transform(fitted_model: FittedModel, data_df: pd.DataFrame) -> Any:
    frame = data_df[fitted_model.feature_names].copy()
    if fitted_model.family in {"deeptab", "ebm"}:
        return frame
    return fitted_model.preprocessor.transform(frame)


def predict_positive_probability(fitted_model: FittedModel, data_df: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability, or CORELS hard predictions as 0/1."""

    transformed = _transform(fitted_model, data_df)
    estimator = fitted_model.estimator
    if fitted_model.family == "corels":
        return np.asarray(estimator.predict(transformed), dtype=float).reshape(-1)
    if fitted_model.family == "aplr":
        probabilities = np.asarray(estimator.predict_class_probabilities(transformed), dtype=float)
    else:
        probabilities = np.asarray(estimator.predict_proba(transformed), dtype=float)
    if probabilities.ndim == 1:
        return probabilities.reshape(-1)
    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    positive = np.flatnonzero(classes.astype(int) == 1)
    index = int(positive[0]) if len(positive) else probabilities.shape[1] - 1
    return probabilities[:, index].reshape(-1)


def predict_model(fitted_model: FittedModel, data_df: pd.DataFrame) -> np.ndarray:
    """Predict integer labels using the fixed 0.5 positive-class threshold."""

    probabilities = predict_positive_probability(fitted_model, data_df)
    return (probabilities >= 0.5).astype(int)


def evaluate_model(fitted_model: FittedModel, test_df: pd.DataFrame, label_col: str) -> dict[str, float]:
    y_true = test_df[label_col].astype(int).to_numpy()
    y_pred = predict_model(fitted_model, test_df.drop(columns=[label_col]))
    return compute_metrics(y_true, y_pred)


def save_fitted_model(fitted_model: FittedModel, output_dir: Path) -> Path:
    """Persist a fitted model with the public save API of its implementation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if fitted_model.family == "deeptab":
        model_path = output_dir / "model.deeptab"
        fitted_model.estimator.save(model_path)
    elif fitted_model.family == "corels":
        model_path = output_dir / "model.corels"
        fitted_model.estimator.save(str(model_path))
        (output_dir / "rulelist.txt").write_text(str(fitted_model.estimator.rl()), encoding="utf-8")
        (output_dir / "predicate_manifest.json").write_text(
            json.dumps(fitted_model.preprocessor.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        joblib.dump(fitted_model.preprocessor, output_dir / "preprocessor.joblib")
    else:
        model_path = output_dir / "model.joblib"
        joblib.dump(fitted_model, model_path)
    estimator = fitted_model.estimator
    get_params = getattr(estimator, "get_params", None)
    resolved_params = get_params(deep=True) if callable(get_params) else {}
    (output_dir / "resolved_model_config.json").write_text(
        json.dumps(
            {
                "model_name": fitted_model.model_name,
                "family": fitted_model.family,
                "estimator_class": f"{type(estimator).__module__}.{type(estimator).__name__}",
                "parameters": resolved_params,
                "feature_names": fitted_model.feature_names,
                "training_summary": fitted_model.training_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment_manifest(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if fitted_model.family == "ebm":
        (output_dir / "explainability_summary.json").write_text(
            json.dumps(
                {
                    "feature_names": list(getattr(estimator, "feature_names_in_", fitted_model.feature_names)),
                    "term_names": list(getattr(estimator, "term_names_", [])),
                    "best_iteration": getattr(estimator, "best_iteration_", None),
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
    return model_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _environment_manifest() -> dict[str, Any]:
    packages = (
        "numpy", "pandas", "scikit-learn", "torch", "deeptab",
        "interpret-core", "aplr", "corels", "xgboost", "lightgbm",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        import torch

        torch_runtime = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except ImportError:
        torch_runtime = {"cuda_available": False, "cuda_device_count": 0}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_runtime": torch_runtime,
        "git": _git_manifest(),
    }


@lru_cache(maxsize=1)
def _git_manifest() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        return {"commit": commit, "worktree_dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "worktree_dirty": None, "status": []}


def _best_epoch(estimator: Any) -> int | str:
    for owner_name in ("trainer_", "_trainer", "trainer"):
        owner = getattr(estimator, owner_name, None)
        if owner is not None:
            callback = getattr(owner, "checkpoint_callback", None)
            best_path = getattr(callback, "best_model_path", "") if callback is not None else ""
            if best_path and Path(best_path).exists():
                try:
                    import torch

                    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
                    if "epoch" in checkpoint:
                        return int(checkpoint["epoch"])
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
            for attr in ("best_epoch", "current_epoch"):
                value = getattr(owner, attr, None)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        pass
    return ""


def _termination_reason(estimator: Any, family: str) -> str:
    if family != "deeptab":
        return "fit_completed"
    trainer = getattr(estimator, "_trainer", None)
    if trainer is None:
        return "fit_completed"
    callback = getattr(trainer, "early_stopping_callback", None)
    if int(getattr(callback, "stopped_epoch", 0) or 0) > 0:
        return "early_stopping"
    if int(getattr(trainer, "current_epoch", 0)) >= int(getattr(trainer, "max_epochs", 1)) - 1:
        return "max_epochs"
    return "fit_completed"
