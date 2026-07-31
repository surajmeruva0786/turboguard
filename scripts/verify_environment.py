#!/usr/bin/env python
"""Sanity-check that the environment has everything TurboGuard needs.

Usage: python scripts/verify_environment.py
"""
from __future__ import annotations

import importlib
import shutil
import sys

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "xgboost",
    "torch",
    "pywt",
    "shap",
    "matplotlib",
    "plotly",
    "streamlit",
    "yaml",
]

MIN_PYTHON = (3, 11)
MIN_FREE_DISK_GB = 5


def check_python_version() -> bool:
    ok = sys.version_info[:2] >= MIN_PYTHON
    logger.info("Python %s.%s.%s%s", *sys.version_info[:3], "" if ok else " (< 3.11 required)")
    return ok


def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            logger.info("  %-12s OK (%s)", pkg, version)
        except ImportError:
            logger.error("  %-12s MISSING", pkg)
            all_ok = False
    return all_ok


def check_disk_space() -> bool:
    free_gb = shutil.disk_usage(".").free / (1024**3)
    ok = free_gb >= MIN_FREE_DISK_GB
    logger.info("Free disk space: %.1f GB (%s)", free_gb, "OK" if ok else f"< {MIN_FREE_DISK_GB} GB required")
    return ok


def check_torch_backend() -> None:
    try:
        import torch

        logger.info("torch CUDA available: %s", torch.cuda.is_available())
    except ImportError:
        pass


def main() -> None:
    logger.info("Checking Python version...")
    py_ok = check_python_version()
    logger.info("Checking required packages...")
    pkg_ok = check_packages()
    logger.info("Checking disk space...")
    disk_ok = check_disk_space()
    check_torch_backend()

    if py_ok and pkg_ok and disk_ok:
        logger.info("Environment OK.")
        sys.exit(0)
    logger.error("Environment check FAILED - see messages above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
