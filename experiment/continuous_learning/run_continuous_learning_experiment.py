from __future__ import annotations

import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from continuous_learning_experiment_common import DEFAULT_SEEDS, MIMIC_CSV_PATH, MIMIC_LABEL_COL, write_results_csv
from run_continuous_learning_baselines import run_baseline_experiments
from run_continuous_learning_hl import run_hl_experiments


def main() -> None:
    print(
        f"Running combined continuous learning experiment on MIMIC from {MIMIC_CSV_PATH} "
        f"with label={MIMIC_LABEL_COL} and seeds={DEFAULT_SEEDS}.",
        flush=True,
    )
    hl_results = run_hl_experiments()
    baseline_results = run_baseline_experiments()
    all_results = list(hl_results) + list(baseline_results)
    out_csv = SCRIPT_DIR / "continuous_results.csv"
    write_results_csv(out_csv, all_results)
    print(f"continuous_results_csv={out_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
