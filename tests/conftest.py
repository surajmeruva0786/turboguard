"""Session-scoped fixtures that make the test suite self-contained.

`data/processed/*/features.parquet` and `runs/random_forest_cwru/model.joblib`
are gitignored (regenerable — see docs/ROADMAP.md's data note), but the
dashboard tests load them from disk at their default paths. On a fresh
checkout (a clean CI runner, or anyone who just cloned the repo) neither
exists yet, so those tests fail with ``FileNotFoundError`` even though the
pipeline itself is fine — this was caught by reproducing a fresh checkout
locally (see git history for the fix commit). This fixture generates the
missing artifacts once per test session, the same way README section 16
documents, so ``pytest -q`` works right after
``git clone && pip install -r requirements.txt`` with no manual setup step.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


@pytest.fixture(scope="session", autouse=True)
def ensure_pipeline_fixtures() -> None:
    python = sys.executable
    cwru_features = REPO_ROOT / "data" / "processed" / "cwru" / "features.parquet"
    ims_features = REPO_ROOT / "data" / "processed" / "ims" / "features.parquet"
    rf_model = REPO_ROOT / "runs" / "random_forest_cwru" / "model.joblib"

    if not cwru_features.exists():
        _run(
            [python, "-m", "src.preprocessing.run", "--dataset", "cwru",
             "--input_dir", "data/raw/cwru/synthetic", "--output_dir", "data/processed/cwru"]
        )
        _run(
            [python, "-m", "src.features.extract", "--dataset", "cwru",
             "--input_dir", "data/processed/cwru", "--output_dir", "data/processed/cwru"]
        )

    if not ims_features.exists():
        _run(
            [python, "-m", "src.preprocessing.run", "--dataset", "ims",
             "--input_dir", "data/raw/ims/synthetic", "--output_dir", "data/processed/ims"]
        )
        _run(
            [python, "-m", "src.features.extract", "--dataset", "ims",
             "--input_dir", "data/processed/ims", "--output_dir", "data/processed/ims"]
        )

    if not rf_model.exists():
        _run(
            [python, "-m", "src.training.train_classical", "--model", "random_forest",
             "--dataset", "cwru", "--output_dir", "runs/random_forest_cwru"]
        )
