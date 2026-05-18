from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ErrorSample:
    idx: int
    y_true: int
    y_pred: int
    kind: str
    features: dict


def collect_errors(
    df: pd.DataFrame,
    label_col: str,
    y_pred: np.ndarray,
    max_error_samples: int,
    random_seed: int,
    feature_cols: list[str] | None = None,
) -> list[ErrorSample]:
    y_true = df[label_col].astype(int).to_numpy()
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return []

    fp = [i for i in wrong if y_true[i] == 0 and y_pred[i] == 1]
    fn = [i for i in wrong if y_true[i] == 1 and y_pred[i] == 0]

    rng = np.random.default_rng(random_seed)

    keep: list[int] = []
    if max_error_samples <= 0:
        keep = list(wrong)
    else:
        half = max_error_samples // 2
        n_fp = min(len(fp), half)
        n_fn = min(len(fn), max_error_samples - n_fp)
        if n_fn < half:
            n_fp = min(len(fp), max_error_samples - n_fn)
        keep.extend(list(rng.choice(fp, size=n_fp, replace=False)) if n_fp > 0 else [])
        keep.extend(list(rng.choice(fn, size=n_fn, replace=False)) if n_fn > 0 else [])

    samples: list[ErrorSample] = []
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]
    for i in keep:
        row = df.iloc[int(i)]
        features = {c: row[c] for c in feature_cols}
        kind = "FP" if (y_true[i] == 0 and y_pred[i] == 1) else "FN"
        samples.append(ErrorSample(idx=int(i), y_true=int(y_true[i]), y_pred=int(y_pred[i]), kind=kind, features=features))
    return samples


def format_error_report(samples: list[ErrorSample], max_details: int = 40) -> str:
    if not samples:
        return "无错误样本。"
    fp = sum(1 for s in samples if s.kind == "FP")
    fn = sum(1 for s in samples if s.kind == "FN")
    lines = [f"错误样本数={len(samples)} (FP={fp}, FN={fn})", ""]
    shown = samples[: max(0, max_details)]
    for s in shown:
        lines.append(f"- idx={s.idx} kind={s.kind} y_true={s.y_true} y_pred={s.y_pred}")
        lines.append(f"  features={s.features}")
    if len(shown) < len(samples):
        lines.append("")
        lines.append(f"（仅展示前 {len(shown)} 条错误样本详情，其余已省略）")
    return "\n".join(lines)
