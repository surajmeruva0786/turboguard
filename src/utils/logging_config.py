"""Consistent logging setup for CLI scripts."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a single stdout handler.

    Safe to call repeatedly (e.g. once per module import) — handlers are
    only attached once per logger name.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
