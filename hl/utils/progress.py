from __future__ import annotations

from datetime import datetime


def log_progress(component: str, message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] [{component}] {message}", flush=True)
