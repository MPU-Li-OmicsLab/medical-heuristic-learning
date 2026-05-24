from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl.config import LLMConfig
from hl.continuous_learning import ContinuousLearningConfig, DriftConfig, run_continuous_learning


def main() -> None:
    data_path = Path("./data/YHD_bicarbonate.csv")
    prev_out_dir = Path("./example_out")
    label_col = "hospital_expire_flag"
    dropped_feature = "wbc"

    if not prev_out_dir.exists():
        raise FileNotFoundError(
            f"Previous HL output directory not found: {prev_out_dir}. "
            "Run `example_training.py` first to generate `./example_out`."
        )

    data = pd.read_csv(data_path)
    if dropped_feature not in data.columns:
        raise ValueError(f"Expected drift feature `{dropped_feature}` not found in {data_path}")

    train_df = data.iloc[:500].copy()
    test_df = data.iloc[500:1000].copy()

    # Simulate the new environment where the `wbc` feature is no longer available.
    train_df = train_df.drop(columns=[dropped_feature])
    test_df = test_df.drop(columns=[dropped_feature])

    llm_cfg = LLMConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model_name="deepseek-v4-pro",
        temperature=0.0,
    )

    continuous_cfg = ContinuousLearningConfig(
        output_dir=None,
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=(
            "Continuous learning on the YHD bicarbonate dataset. "
            f"The previously trained heuristic system is stored in {prev_out_dir}. "
            f"In the new environment, feature `{dropped_feature}` is no longer available, "
            "so the rule system must adapt while preserving useful prior logic."
        ),
        drift=DriftConfig(
            dropped_cols=(dropped_feature,),
            added_cols=(),
            renamed_cols=(),
            change_note=(
                "The new dataset version no longer provides the `wbc` measurement, "
                "so the heuristic system must remove this dependency and adapt the rule logic."
            ),
            prev_hl_out_dir=prev_out_dir,
        )
    )

    result = run_continuous_learning(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        llm_cfg=llm_cfg,
        continuous_cfg=continuous_cfg,
    )

    print(f"continuous_learning_out_dir={result.out_dir}")
    print(f"continuous_learning_final_model={result.final_model_path}")


if __name__ == "__main__":
    main()
