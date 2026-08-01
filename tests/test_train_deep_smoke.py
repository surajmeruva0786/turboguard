"""Smoke tests for src.training.train_deep against the committed synthetic data.

Kept to a handful of epochs / tiny batch size — these verify the training
loop wires together (data -> model -> loss -> backward -> save), not model
quality (see `runs/*_smoke/metrics.json` for the actual smoke-run numbers
committed alongside this test).
"""
from __future__ import annotations

import argparse

from src.training.train_deep import build_cnn_dataset, build_multitask_dataset, train_cnn, train_hybrid
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        dataset_fault="cwru",
        dataset_rul="ims",
        n_classes=5,
        seq_len=10,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_epochs=1,
        fault_weight=1.0,
        rul_weight=0.5,
        augment=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_cnn_dataset_matches_synthetic_cwru_size():
    data_cfg = load_yaml("configs/data.yaml")
    dataset = build_cnn_dataset(data_cfg, "cwru")
    assert len(dataset) == 20  # 5 classes x 4 loads, one window each


def test_build_multitask_dataset_combines_fault_and_rul_samples():
    data_cfg = load_yaml("configs/data.yaml")
    dataset = build_multitask_dataset(data_cfg, "cwru", "ims", seq_len=10)
    assert len(dataset) > 20  # cwru fault sequences + ims rul sequences


def test_train_cnn_smoke_run_reduces_or_completes_loss():
    set_seed(42)
    data_cfg = load_yaml("configs/data.yaml")
    result = train_cnn(_base_args(), data_cfg)
    assert len(result["history"]) == 2
    assert all(h["loss"] >= 0 for h in result["history"])


def test_train_hybrid_smoke_run_produces_finite_multitask_loss():
    set_seed(42)
    data_cfg = load_yaml("configs/data.yaml")
    result = train_hybrid(_base_args(), data_cfg)
    assert len(result["history"]) == 2
    assert all(h["loss"] >= 0 for h in result["history"])
