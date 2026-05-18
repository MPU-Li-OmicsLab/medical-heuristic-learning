from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

from config import LLMConfig, RunConfig
from run import run_heuristic_learning


def main() -> None:
    data = pd.read_csv("/data/yk/HL/data/YHD_bicarbonate.csv")
    label_col = "hospital_expire_flag"

    train_df = data.iloc[:500].copy()
    test_df = data.iloc[500:1000].copy()

    output_dir = Path(os.getenv("HL_OUTPUT_DIR", "/data/yk/HL/out_example_v4pro"))
    run_cfg = RunConfig(output_dir=output_dir)
    llm_cfg = LLMConfig()
    run_heuristic_learning(train_df=train_df, test_df=test_df, label_col=label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)


if __name__ == "__main__":
    main()
