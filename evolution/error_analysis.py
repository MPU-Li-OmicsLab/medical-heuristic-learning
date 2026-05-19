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

    rng = np.random.default_rng(random_seed)
    is_binary = set(np.unique(y_true).tolist()).issubset({0, 1})

    if max_error_samples > 0 and len(wrong) > max_error_samples:
        keep = list(rng.choice(wrong, size=max_error_samples, replace=False))
    else:
        keep = [int(i) for i in wrong]

    samples: list[ErrorSample] = []
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]
    for i in keep:
        row = df.iloc[int(i)]
        features = {c: row[c] for c in feature_cols}
        if is_binary:
            kind = "FP" if (y_true[i] == 0 and y_pred[i] == 1) else "FN"
        else:
            kind = "ERR"
        samples.append(ErrorSample(idx=int(i), y_true=int(y_true[i]), y_pred=int(y_pred[i]), kind=kind, features=features))
    return samples


def format_error_report(samples: list[ErrorSample], max_details: int = 40) -> str:
    if not samples:
        return "No error samples."
    fp = sum(1 for s in samples if s.kind == "FP")
    fn = sum(1 for s in samples if s.kind == "FN")
    other = len(samples) - fp - fn
    if fp + fn > 0 and other == 0:
        header = f"Error samples={len(samples)} (FP={fp}, FN={fn})"
    else:
        header = f"Error samples={len(samples)}"
    lines = [header, ""]
    shown = samples[: max(0, max_details)]
    for s in shown:
        lines.append(f"- idx={s.idx} kind={s.kind} y_true={s.y_true} y_pred={s.y_pred}")
        lines.append(f"  features={s.features}")
    if len(shown) < len(samples):
        lines.append("")
        lines.append(f"(Showing only the first {len(shown)} samples; the rest are omitted.)")
    return "\n".join(lines)
