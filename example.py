from __future__ import annotations

import argparse
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import LLMConfig, RunConfig
from run import run_heuristic_learning


def _default_output_dir() -> Path:
    base = Path(__file__).resolve().parent / "out"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / stamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-baselines", action="store_true")
    parser.add_argument("--skip-univariate", action="store_true")
    parser.add_argument("--skip-knowledge", action="store_true")
    parser.add_argument("--skip-v0", action="store_true")
    parser.add_argument("--skip-iterations", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv("/data/yk/HL/data/YHD_bicarbonate.csv")
    label_col = "hospital_expire_flag"

    train_df = data.iloc[:500].copy()
    test_df = data.iloc[500:1000].copy()

    output_dir_arg = (args.output_dir or "").strip()
    output_dir_env = (os.getenv("HL_OUTPUT_DIR", "") or "").strip()
    output_dir: Path | None = None
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    elif output_dir_env:
        output_dir = Path(output_dir_env)

    not_from_scratch = any(
        [
            bool(args.skip_univariate),
            bool(args.skip_knowledge),
            bool(args.skip_v0),
            bool(args.skip_iterations),
        ]
    )
    if output_dir is None and not_from_scratch:
        raise RuntimeError("未配置输出目录，但当前不是从头运行。请设置 --output-dir 或环境变量 HL_OUTPUT_DIR 指向已有输出目录。")
    if output_dir is None:
        output_dir = _default_output_dir()

    run_cfg = RunConfig(
        output_dir=output_dir,
        train_baselines=bool(args.train_baselines),
        run_univariate_probe=not bool(args.skip_univariate),
        run_knowledge_probe=not bool(args.skip_knowledge),
        run_v0_generation=not bool(args.skip_v0),
        run_iterations=not bool(args.skip_iterations),
    )
    llm_cfg = LLMConfig()
    run_heuristic_learning(train_df=train_df, test_df=test_df, label_col=label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)


if __name__ == "__main__":
    main()
