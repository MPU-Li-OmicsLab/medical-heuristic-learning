from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from hl.metrics import compute_metrics


def train_and_eval_baselines(
    train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str, out_dir: Path, random_seed: int = 42
) -> dict[str, dict[str, float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    feature_cols = [c for c in train_df.columns if c != label_col]
    x_train = train_df[feature_cols].astype(float)
    x_test = test_df[feature_cols].astype(float)
    fill = x_train.median(numeric_only=True)
    x_train = x_train.fillna(fill)
    x_test = x_test.fillna(fill)
    y_train = train_df[label_col].astype(int).to_numpy()
    y_test = test_df[label_col].astype(int).to_numpy()

    results: dict[str, dict[str, float]] = {}

    lr = LogisticRegression(max_iter=4000)
    lr.fit(x_train, y_train)
    y_score = lr.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    results["baseline_lr"] = compute_metrics(y_test, y_pred, y_score=y_score)
    (out_dir / "baseline_lr.pkl").write_bytes(pickle.dumps(lr))

    dt = DecisionTreeClassifier(random_state=random_seed, max_depth=4)
    dt.fit(x_train, y_train)
    y_score = dt.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    results["baseline_dt"] = compute_metrics(y_test, y_pred, y_score=y_score)
    (out_dir / "baseline_dt.pkl").write_bytes(pickle.dumps(dt))

    return results

