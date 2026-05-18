from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

