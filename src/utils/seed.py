"""Deterministic seeding across numpy, python's random, and (optionally) torch."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed all RNGs used across the pipeline for reproducibility.

    Parameters
    ----------
    seed:
        Seed value applied to Python's ``random``, NumPy, and PyTorch (if
        installed).
    deterministic_torch:
        If PyTorch is available, also request deterministic algorithms.
        This can slow down GPU training but is required for byte-for-byte
        reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
