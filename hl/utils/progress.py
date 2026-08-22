from __future__ import annotations

import logging


_LOGGER = logging.getLogger("hl")


def log_progress(component: str, message: str) -> None:
    _LOGGER.info("[%s] %s", component, message)
