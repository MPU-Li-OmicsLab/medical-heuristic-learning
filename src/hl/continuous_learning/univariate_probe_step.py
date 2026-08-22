from __future__ import annotations

from pathlib import Path

import pandas as pd

from hl.config import RunConfig
from hl.continuous_learning.config import DriftConfig
from hl.probes.univariate import run_univariate_probe
from hl.utils.io import write_text
from hl.utils.progress import log_progress


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _format_univariate_summary(df: pd.DataFrame, top_k: int) -> str:
    if df is None or df.empty:
        return ""
    try:
        return df.head(min(int(top_k), len(df))).to_string(index=False)
    except Exception:
        return ""


def _filter_previous_probe(prev_df: pd.DataFrame, drift: DriftConfig) -> pd.DataFrame:
    out_df = prev_df.copy()
    if "feature" in out_df.columns:
        out_df = out_df.loc[~out_df["feature"].astype(str).isin(list(drift.dropped_cols))].copy()
        for old_name, new_name in drift.renamed_cols:
            mask = out_df["feature"].astype(str) == str(old_name)
            out_df.loc[mask, "feature"] = str(new_name)
    return out_df


def run_univariate_probe_task(
    *,
    train_df: pd.DataFrame,
    label_col: str,
    run_cfg: RunConfig,
    univariate_path: Path,
    feature_cols: list[str],
    drift: DriftConfig,
) -> tuple[list[str], list[str], str]:
    prev_path = drift.prev_hl_out_dir / "probe_univariate_results.csv" if drift.prev_hl_out_dir is not None else Path("__missing__")
    prev_df = _read_csv_if_exists(prev_path)
    if prev_df is not None:
        log_progress("HL-CL-U", f"Loaded previous univariate probe from {prev_path}.")
        write_text(univariate_path.parent / "probe_univariate_results_prev.csv", prev_df.to_csv(index=False))
    else:
        log_progress("HL-CL-U", "No previous univariate probe file is available under drift context.")
        write_text(univariate_path.parent / "probe_univariate_results_prev.csv", "")

    univariate_df: pd.DataFrame | None = None
    if run_cfg.run_univariate_probe:
        log_progress("HL-CL-U", "Computing updated univariate probe results.")
        new_df = run_univariate_probe(train_df=train_df, label_col=label_col)
        if prev_df is None:
            univariate_df = new_df
        else:
            filtered_prev = _filter_previous_probe(prev_df, drift)
            added_rows = (
                new_df.loc[new_df["feature"].astype(str).isin(list(drift.added_cols))].copy()
                if "feature" in new_df.columns
                else pd.DataFrame(columns=new_df.columns)
            )
            univariate_df = pd.concat([filtered_prev, added_rows], axis=0, ignore_index=True)
            if univariate_df.empty:
                univariate_df = new_df
            elif "p_value" in univariate_df.columns and "missing_rate" in univariate_df.columns:
                univariate_df = univariate_df.sort_values(
                    by=["p_value", "missing_rate"],
                    ascending=[True, True],
                    na_position="last",
                )
        univariate_df.to_csv(univariate_path, index=False)
        log_progress("HL-CL-U", f"Saved updated univariate probe results to {univariate_path}.")
    elif univariate_path.exists():
        log_progress("HL-CL-U", f"Reusing existing univariate probe file: {univariate_path}.")
        univariate_df = _read_csv_if_exists(univariate_path)
    else:
        log_progress("HL-CL-U", "Univariate probe is disabled and no cached file was found.")

    if univariate_df is not None and len(univariate_df) > 0 and "feature" in univariate_df.columns:
        topk = min(run_cfg.univariate_top_k, len(univariate_df))
        top_features = univariate_df.head(topk)["feature"].astype(str).tolist()
        report_features = list(top_features)
        univariate_summary = _format_univariate_summary(univariate_df, run_cfg.univariate_top_k)
        log_progress("HL-CL-U", f"Prepared top-{topk} updated univariate features.")
        return top_features, report_features, univariate_summary

    log_progress("HL-CL-U", "Falling back to all feature columns because no univariate summary is available.")
    return list(feature_cols), list(feature_cols), ""
