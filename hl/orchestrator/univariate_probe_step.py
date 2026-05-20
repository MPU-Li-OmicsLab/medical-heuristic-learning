from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl.config import RunConfig
from hl.probes.univariate import run_univariate_probe


def run_univariate_probe_task(
    *,
    train_df: pd.DataFrame,
    label_col: str,
    run_cfg: RunConfig,
    univariate_path: Path,
    feature_cols: list[str],
) -> tuple[list[str], list[str], str]:
    univariate_df: pd.DataFrame | None = None
    if run_cfg.run_univariate_probe:
        univariate_df = run_univariate_probe(train_df=train_df, label_col=label_col)
        univariate_df.to_csv(univariate_path, index=False)
    elif univariate_path.exists():
        try:
            univariate_df = pd.read_csv(univariate_path)
        except Exception:
            univariate_df = None

    if univariate_df is not None and len(univariate_df) > 0:
        topk = min(run_cfg.univariate_top_k, len(univariate_df))
        top_features = univariate_df.head(topk)["feature"].tolist()
        report_features = list(top_features)
        univariate_summary = univariate_df.head(topk).to_string(index=False)
        return top_features, report_features, univariate_summary

    return list(feature_cols), list(feature_cols), ""
