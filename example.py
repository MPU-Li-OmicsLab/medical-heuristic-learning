from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl.config import LLMConfig, RunConfig
from hl.orchestrator import run_heuristic_learning


def main() -> None:
    data = pd.read_csv("/data/yk/HL/data/YHD_bicarbonate.csv")
    label_col = "hospital_expire_flag"

    train_df = data.iloc[:500].copy()
    test_df = data.iloc[500:1000].copy()

    mode = "scratch"
    output_dir: Path | None = None

    run_univariate_probe = True
    run_knowledge_probe = True
    run_v0_generation = True
    run_iterations = True

    if mode == "iterate_only":
        run_univariate_probe = False
        run_knowledge_probe = False
        run_v0_generation = False
        run_iterations = True

    not_from_scratch = mode != "scratch" or not (
        run_univariate_probe and run_knowledge_probe and run_v0_generation and run_iterations
    )
    if not_from_scratch and output_dir is None:
        raise RuntimeError("This run is not from scratch; please set output_dir to an existing output directory.")

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
