from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


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


def main() -> None:
    data_path = Path("./data/YHD_bicarbonate.csv")
    model_path = Path("./example_out/final_heuristic_model.py")
    label_col = "hospital_expire_flag"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found: {model_path}. Run `example_training.py` first to generate it."
        )

    data = pd.read_csv(data_path)
    infer_df = data.tail(5).copy()
    feature_cols = [c for c in infer_df.columns if c != label_col]

    predict_fn = _load_predict_fn(model_path)

    predictions: list[int] = []
    for _, row in infer_df.iterrows():
        features = {col: row[col] for col in feature_cols}
        predictions.append(int(predict_fn(features)))

    result_df = infer_df[[label_col]].copy()
    result_df.insert(0, "row_index", infer_df.index)
    result_df["prediction"] = predictions

    print("Inference on the last 5 rows of ./data/YHD_bicarbonate.csv")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
