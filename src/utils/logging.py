"""Logger estandarizado con rich."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str = "tps", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = RichHandler(show_time=True, show_level=True, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
