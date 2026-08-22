from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl import load_model


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

    predict_fn = load_model(model_path)

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
