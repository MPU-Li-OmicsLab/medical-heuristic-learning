from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import LLMConfig, RunConfig
from run import run_heuristic_learning


def _default_output_dir() -> Path:
    base = Path(__file__).resolve().parent / "out"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / stamp


def main() -> None:
    data = pd.read_csv("/data/yk/HL/data/YHD_bicarbonate.csv")
    label_col = "hospital_expire_flag"

    train_df = data.iloc[:500].copy()
    test_df = data.iloc[500:1000].copy()

    mode = (os.getenv("HL_MODE", "scratch") or "scratch").strip().lower()

    run_univariate_probe = True
    run_knowledge_probe = True
    run_v0_generation = True
    run_iterations = True

    if mode in {"iterate_only", "iter", "iterate"}:
        run_univariate_probe = False
        run_knowledge_probe = False
        run_v0_generation = False
        run_iterations = True

    not_from_scratch = (
        mode not in {"scratch", "from_scratch"}
        or (not run_univariate_probe)
        or (not run_knowledge_probe)
        or (not run_v0_generation)
        or (not run_iterations)
    )

    output_dir_env = (os.getenv("HL_OUTPUT_DIR", "") or "").strip()
    if not_from_scratch and not output_dir_env:
        raise RuntimeError(
            "This run is not from scratch, but HL_OUTPUT_DIR is not set. "
            "Set HL_OUTPUT_DIR to an existing output directory to reuse previous artifacts."
        )

    output_dir = Path(output_dir_env) if output_dir_env else _default_output_dir()

    run_cfg = RunConfig(
        output_dir=output_dir,
        run_univariate_probe=run_univariate_probe,
        run_knowledge_probe=run_knowledge_probe,
        run_v0_generation=run_v0_generation,
        run_iterations=run_iterations,
        task_description=(
            "Binary classification on a clinical tabular dataset. "
            f"Predict {label_col} (in-hospital mortality flag) from the provided features. "
            "Optimize metrics by the configured metric priority."
        ),
    )
    llm_cfg = LLMConfig()
    run_heuristic_learning(train_df=train_df, test_df=test_df, label_col=label_col, run_cfg=run_cfg, llm_cfg=llm_cfg)


if __name__ == "__main__":
    main()
