#!/usr/bin/env python
"""End-to-end pipeline orchestration (README section 16, roadmap step 113):
synthetic data -> preprocessing -> feature extraction -> classical training
-> deep-model smoke training -> RUL/cross-condition/cross-dataset
evaluation -> XAI report.

Runs the exact same CLI commands documented individually across the
README, chained together with fail-fast subprocess calls so a single
command reproduces the whole pipeline from a clean checkout.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --skip-synthetic --skip-deep
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Reuse the already-committed synthetic datasets instead of regenerating them.",
    )
    parser.add_argument(
        "--skip-deep", action="store_true",
        help="Skip the deep-model smoke-training steps (CNN + hybrid); classical/RUL/XAI still run.",
    )
    args = parser.parse_args()
    python = sys.executable

    if not args.skip_synthetic:
        run([python, "scripts/generate_synthetic.py", "--output_dir", "data/raw"])

    for dataset in ("cwru", "ims"):
        run(
            [
                python, "-m", "src.preprocessing.run",
                "--dataset", dataset,
                "--input_dir", f"data/raw/{dataset}/synthetic",
                "--output_dir", f"data/processed/{dataset}",
            ]
        )
        run(
            [
                python, "-m", "src.features.extract",
                "--dataset", dataset,
                "--input_dir", f"data/processed/{dataset}",
                "--output_dir", f"data/processed/{dataset}",
            ]
        )

    run(
        [
            python, "-m", "src.training.train_classical",
            "--model", "random_forest", "--dataset", "cwru",
            "--output_dir", "runs/random_forest_cwru",
        ]
    )

    if not args.skip_deep:
        run(
            [
                python, "-m", "src.training.train_deep",
                "--model", "turboguard_cnn", "--dataset_fault", "cwru",
                "--epochs", "3", "--batch_size", "4", "--warmup_epochs", "1",
                "--output_dir", "runs/cnn_smoke",
            ]
        )
        run(
            [
                python, "-m", "src.training.train_deep",
                "--model", "turboguard_hybrid", "--dataset_fault", "cwru", "--dataset_rul", "ims",
                "--epochs", "3", "--batch_size", "4", "--warmup_epochs", "1",
                "--output_dir", "runs/hybrid_smoke",
            ]
        )

    run([python, "-m", "src.evaluation.evaluate_rul", "--output_dir", "results/rul_ims"])
    run(
        [
            python, "-m", "src.evaluation.cross_condition",
            "--model", "random_forest", "--dataset", "cwru",
            "--train_loads", "0", "1", "2", "--test_load", "3",
            "--output_dir", "results/cross_condition",
        ]
    )
    run(
        [
            python, "-m", "src.evaluation.cross_dataset",
            "--model", "random_forest", "--train_dataset", "cwru", "--test_dataset", "ims",
            "--output_dir", "results/cross_dataset",
        ]
    )
    run(
        [
            python, "-m", "src.xai.explain",
            "--model_dir", "runs/random_forest_cwru", "--processed_dir", "data/processed/cwru",
            "--sample_idx", "0", "--output_dir", "results/xai/sample_0",
        ]
    )

    print("\nFull pipeline complete.")


if __name__ == "__main__":
    main()
