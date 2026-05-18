from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    auc = float("nan")
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, np.asarray(y_score)))
        except Exception:
            auc = float("nan")

    return {
        "ACC": acc,
        "F1": f1,
        "AUC": auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def generate_metric_description(metric_priority: list[str] | tuple[str, ...]) -> str:
    metrics = [m.strip() for m in metric_priority if m and m.strip()]
    if not metrics:
        return "本次优化关注整体预测表现。"
    if len(metrics) == 1:
        return f"本次优化首要关注 {metrics[0]} 值。"
    return f"本次优化首要关注 {metrics[0]} 值，其次为 {metrics[1]}。请优先改进假阴性与假阳性之间的平衡。"
