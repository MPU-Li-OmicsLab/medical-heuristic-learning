from __future__ import annotations

import pandas as pd
from pathlib import Path

from config import LLMConfig, RunConfig
from run import run_heuristic_learning


def main() -> None:
    data = pd.read_csv("/data/yk/HL/data/YHD_bicarbonate.csv")
    label_col = "hospital_expire_flag"

    train_df = data.iloc[:500].copy()
    test_df = data.iloc[500:].copy()

    run_cfg = RunConfig(output_dir=Path("/data/yk/HL/out_example"))
    llm_cfg = LLMConfig()
    run_heuristic_learning(train_df=train_df, test_df=test_df, label_col=label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)


if __name__ == "__main__":
    main()
