from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PredictFunction = Callable[[dict[str, Any]], int]
BatchPredictFunction = Callable[[pd.DataFrame], list[int]]


def load_model(model_path: str | Path) -> PredictFunction:
    """Load the original single-row predictor from an exported MHL model.

    The model file is executable Python code and is executed while loading.
    Only load artifacts from a trusted source.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"MHL model file not found: {path}")

    spec = importlib.util.spec_from_file_location("final_heuristic_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create a module spec for MHL model: {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Failed to execute MHL model file: {path}") from exc

    predict_fn = getattr(module, "predict", None)
    if not callable(predict_fn):
        raise RuntimeError(f"Callable `predict(features)` not found in MHL model: {path}")
    return predict_fn


def load_batch_model(model_path: str | Path) -> BatchPredictFunction:
    """Load an MHL model and assemble a predictor for multiple DataFrame rows.

    The returned function preserves input row order and returns one integer
    prediction per row. The input DataFrame must contain model features only;
    callers are responsible for removing labels and all other preprocessing.
    """
    predict_one = load_model(model_path)

    def predict_batch(data: pd.DataFrame) -> list[int]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Batch predictor input must be a pandas DataFrame.")
        if not data.columns.is_unique:
            raise ValueError("Batch predictor input must have unique column names.")

        records = data.to_dict(orient="records")
        return [int(predict_one(features)) for features in records]

    return predict_batch
