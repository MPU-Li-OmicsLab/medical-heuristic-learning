from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl import load_batch_model, load_model


def main() -> None:
    data_path = Path("./data/YHD_bicarbonate.csv")
    model_path = Path("./example_out/final_heuristic_model.py")
    label_col = "hospital_expire_flag"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found: {model_path}. Run `example_training.py` first to generate it."
        )

    data = pd.read_csv(data_path)

    # Prepare feature-only data before calling either prediction interface.
    feature_data = data.drop(columns=[label_col])

    # Load the original final predictor for one-row inference.
    predict_one = load_model(model_path)
    single_features = feature_data.iloc[-1].to_dict()
    single_prediction = int(predict_one(single_features))

    print("Inference on one row from ./data/YHD_bicarbonate.csv")
    print(f"prediction={single_prediction}")

    # Pass a feature-only DataFrame to the assembled batch predictor.
    predict_batch = load_batch_model(model_path)
    infer_df = data.tail(5).copy()
    infer_features = feature_data.tail(5)
    predictions = predict_batch(infer_features)

    result_df = infer_df[[label_col]].copy()
    result_df.insert(0, "row_index", infer_df.index)
    result_df["prediction"] = predictions

    print("Inference on the last 5 rows of ./data/YHD_bicarbonate.csv")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
