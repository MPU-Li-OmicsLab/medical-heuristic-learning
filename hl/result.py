from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    """Common artifact paths produced by an MHL run."""

    out_dir: Path
    heuristic_path: Path
    final_model_path: Path
