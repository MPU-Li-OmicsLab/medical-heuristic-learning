from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl.config import LLMConfig, RunConfig
from hl.orchestrator import run_heuristic_learning


def main() -> None:
    data = pd.read_csv("./data/YHD_bicarbonate.csv")
    label_col = "hospital_expire_flag"

    train_df = data.iloc[:500].copy()
    test_df = data.iloc[500:1000].copy()

    # Complete steps
    run_cfg = RunConfig(
        output_dir=Path("./example_out"),
        run_univariate_probe=True,
        run_knowledge_probe=True,
        run_v0_generation=True,
        run_iterations=True,
        task_description=(
            "Binary classification on a clinical tabular dataset. "
            f"Predict {label_col} (in-hospital mortality flag) from the provided features. "
            "Optimize metrics by the configured metric priority."
        ),
    )

    # LLM config
    llm_cfg = LLMConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        # Set this environment variable in your shell before running the script:
        # export DEEPSEEK_API_KEY=sk-xxxxxx
        model_name="deepseek-v4-pro",
        temperature=0.3,
    )
    run_heuristic_learning(train_df=train_df, test_df=test_df, label_col=label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)


if __name__ == "__main__":
    main()
