from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    classes = np.unique(y_true)
    is_binary = classes.size <= 2
    f1 = float(f1_score(y_true, y_pred, average=("binary" if is_binary else "macro"), zero_division=0))

    sensitivity = float("nan")
    specificity = float("nan")
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if is_binary:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "ACC": acc,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def generate_metric_description(metric_priority: list[str] | tuple[str, ...]) -> str:
    metrics = [m.strip() for m in metric_priority if m and m.strip()]
    if not metrics:
        return "This optimization focuses on overall predictive performance."
    if len(metrics) == 1:
        return f"This optimization prioritizes {metrics[0]}."
    if len(metrics) == 2:
        return f"This optimization prioritizes {metrics[0]} first, then {metrics[1]}."
    return (
        f"This optimization prioritizes {metrics[0]} first, then {metrics[1]}. "
        f"Finally, it considers {metrics[2]} and {metrics[3]}. "
        "Focus on balancing false negatives and false positives."
    )
