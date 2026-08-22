from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable


PredictFunction = Callable[[dict[str, Any]], int]


def load_model(model_path: str | Path) -> PredictFunction:
    """Load the ``predict(features)`` function from an exported MHL model.

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
