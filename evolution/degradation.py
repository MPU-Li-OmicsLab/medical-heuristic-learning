from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DegradationResult:
    degraded_indices: list[int]


def detect_degradation(y_true: np.ndarray, y_pred_old: np.ndarray, y_pred_new: np.ndarray) -> DegradationResult:
    y_true = np.asarray(y_true).astype(int)
    y_pred_old = np.asarray(y_pred_old).astype(int)
    y_pred_new = np.asarray(y_pred_new).astype(int)
    old_correct = y_pred_old == y_true
    new_wrong = y_pred_new != y_true
    degraded = np.where(old_correct & new_wrong)[0].tolist()
    return DegradationResult(degraded_indices=degraded)


def format_degradation_warning(indices: list[int], max_items: int = 20) -> str:
    if not indices:
        return "无退化案例。"
    shown = indices[:max_items]
    more = len(indices) - len(shown)
    msg = f"退化案例数={len(indices)}，示例索引={shown}"
    if more > 0:
        msg += f"（另有 {more} 条未展示）"
    return msg


def collect_degradation_examples(
    df: pd.DataFrame,
    label_col: str,
    degraded_indices: list[int],
    y_pred_old: np.ndarray,
    y_pred_new: np.ndarray,
    feature_cols: list[str],
    max_samples: int,
    random_seed: int,
) -> list[dict]:
    if not degraded_indices:
        return []
    rng = np.random.default_rng(random_seed)
    k = min(max_samples, len(degraded_indices))
    chosen = list(rng.choice(np.asarray(degraded_indices, dtype=int), size=k, replace=False)) if k > 0 else []
    y_true = df[label_col].astype(int).to_numpy()

    examples: list[dict] = []
    for i in chosen:
        row = df.iloc[int(i)]
        feats = {c: row[c] for c in feature_cols}
        examples.append(
            {
                "idx": int(i),
                "y_true": int(y_true[int(i)]),
                "y_pred_old": int(y_pred_old[int(i)]),
                "y_pred_new": int(y_pred_new[int(i)]),
                "features": feats,
            }
        )
    return examples
