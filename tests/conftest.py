from __future__ import annotations

import pandas as pd
import pytest


def _synthetic_frame(risk_values: list[float]) -> pd.DataFrame:
    size = len(risk_values)
    return pd.DataFrame(
        {
            "risk_score": risk_values,
            "age": [30 + (index * 3) % 45 for index in range(size)],
            "ward": ["A" if index % 2 == 0 else "B" for index in range(size)],
            "binary_marker": [index % 2 for index in range(size)],
            "target": [int(value >= 0.0) for value in risk_values],
        }
    )


@pytest.fixture
def binary_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balanced, deterministic frames with no dependency on repository CSVs."""

    train_df = _synthetic_frame([-2.0, -1.6, -1.2, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
    val_df = _synthetic_frame([-1.8, -1.0, -0.3, -0.05, 0.05, 0.3, 1.0, 1.8])
    return train_df, val_df
