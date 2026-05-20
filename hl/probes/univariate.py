from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr


@dataclass(frozen=True)
class UnivariateResult:
    feature: str
    feature_type: str
    n_valid: int
    method: str
    statistic: float
    p_value: float
    direction: str
    missing_rate: float
    pointbiserial_r: float | None = None
    pointbiserial_p: float | None = None
    mwu_u: float | None = None
    mwu_p: float | None = None
    chi2_stat: float | None = None
    chi2_p: float | None = None
    binned_or_q4_rel_to_q1: str | None = None


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def _binned_odds_ratios_q4(x: pd.Series, y: pd.Series) -> str:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return ""
    try:
        df["bin"] = pd.qcut(df["x"], q=4, duplicates="drop")
    except Exception:
        return ""

    bins = list(df["bin"].cat.categories)
    if len(bins) < 2:
        return ""

    y1_q1 = int(((df["bin"] == bins[0]) & (df["y"] == 1)).sum())
    y0_q1 = int(((df["bin"] == bins[0]) & (df["y"] == 0)).sum())

    parts: list[str] = []
    for idx, b in enumerate(bins, start=1):
        y1 = int(((df["bin"] == b) & (df["y"] == 1)).sum())
        y0 = int(((df["bin"] == b) & (df["y"] == 0)).sum())
        if idx == 1:
            parts.append("Q1=1.00")
            continue
        or_val = _odds_ratio(y1, y0, y1_q1, y0_q1)
        parts.append(f"Q{idx}={or_val:.2f}")
    return "; ".join(parts)


def run_univariate_probe(train_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    y = train_df[label_col].astype(int)
    results: list[UnivariateResult] = []

    for col in train_df.columns:
        if col == label_col:
            continue
        s = train_df[col]
        missing_rate = float(s.isna().mean())
        n_valid = int(s.notna().sum())

        unique_vals = s.dropna().unique()
        is_binary = len(unique_vals) <= 2 and set(map(_safe_float, unique_vals)).issubset({0.0, 1.0})
        is_numeric = pd.api.types.is_numeric_dtype(s)

        if is_numeric and not is_binary:
            x = pd.to_numeric(s, errors="coerce")
            df = pd.DataFrame({"x": x, "y": y}).dropna()
            method = "pointbiserial"
            stat = float("nan")
            p = float("nan")
            direction = ""
            r_val: float | None = None
            p_r_val: float | None = None
            mwu_p: float | None = None
            mwu_u: float | None = None

            if not df.empty:
                try:
                    r, p_r = pointbiserialr(df["y"].to_numpy(), df["x"].to_numpy())
                    r_val = float(r)
                    p_r_val = float(p_r)
                    stat = float(r)
                    p = float(p_r)
                    direction = "pos" if r_val >= 0 else "neg"
                except Exception:
                    pass

                try:
                    x1 = df.loc[df["y"] == 1, "x"].to_numpy()
                    x0 = df.loc[df["y"] == 0, "x"].to_numpy()
                    if len(x1) > 0 and len(x0) > 0:
                        u, p_u = mannwhitneyu(x1, x0, alternative="two-sided")
                        mwu_u = float(u)
                        mwu_p = float(p_u)
                        if np.isnan(p) or mwu_p < p:
                            method = "mwu"
                            p = mwu_p
                            stat = mwu_u if mwu_u is not None else float("nan")
                except Exception:
                    pass

            binned_or = _binned_odds_ratios_q4(x, y)
            results.append(
                UnivariateResult(
                    feature=col,
                    feature_type="continuous",
                    n_valid=n_valid,
                    method=method,
                    statistic=stat,
                    p_value=p,
                    pointbiserial_p=p_r_val,
                    mwu_u=mwu_u,
                    direction=direction,
                    missing_rate=missing_rate,
                    pointbiserial_r=r_val,
                    mwu_p=mwu_p,
                    binned_or_q4_rel_to_q1=binned_or,
                )
            )
            continue

        df = pd.DataFrame({"x": s, "y": y}).dropna()
        method = "chi2"
        stat = float("nan")
        p = float("nan")
        direction = ""
        chi2_stat: float | None = None
        chi2_p: float | None = None

        if not df.empty:
            try:
                ctab = pd.crosstab(df["x"], df["y"])
                if ctab.shape[0] >= 2 and ctab.shape[1] == 2:
                    chi2, p_chi2, _, _ = chi2_contingency(ctab)
                    stat = float(chi2)
                    p = float(p_chi2)
                    chi2_stat = stat
                    chi2_p = p
                elif ctab.shape[0] == 1 and ctab.shape[1] == 2:
                    stat = 0.0
                    p = 1.0
                    chi2_stat = stat
                    chi2_p = p
            except Exception:
                pass

        results.append(
            UnivariateResult(
                feature=col,
                feature_type="binary" if is_binary else "categorical",
                n_valid=n_valid,
                method=method,
                statistic=stat,
                p_value=p,
                chi2_stat=chi2_stat,
                chi2_p=chi2_p,
                direction=direction,
                missing_rate=missing_rate,
            )
        )

    out_df = pd.DataFrame([r.__dict__ for r in results])
    out_df = out_df.sort_values(by=["p_value", "missing_rate"], ascending=[True, True], na_position="last")
    out_df.insert(0, "rank", range(1, len(out_df) + 1))
    return out_df
