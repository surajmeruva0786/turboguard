#!/usr/bin/env python
"""Train a deep model: TurboGuard-CNN (single-task) or TurboGuard-Hybrid
(multi-task fault + RUL), per README sections 9.2/9.3/10.

Usage (README section 16):
    python -m src.training.train_deep \\
        --model turboguard_hybrid \\
        --dataset_fault cwru \\
        --dataset_rul ims \\
        --epochs 120 \\
        --batch_size 32 \\
        --output_dir runs/hybrid_multitask

For ``--model turboguard_cnn``, only ``--dataset_fault`` is used (single
task, single-window). AdamW + cosine-annealing-with-warmup schedule, AMP on
CUDA, and (for the hybrid model) masked multi-task Huber+CE loss.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.cwru_loader import load_cwru_dataset
from src.data.dataset import (
    FaultWindowDataset,
    SequenceFaultDataset,
    SequenceRULDataset,
    collate_sequence_samples,
)
from src.data.ims_loader import load_ims_dataset
from src.models.turboguard_cnn import TurboGuardCNN
from src.models.turboguard_hybrid import TurboGuardHybrid
from src.training.augmentation import augment_batch
from src.utils.io import ensure_dir, load_yaml, save_json, save_yaml
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def cosine_warmup_lr(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """LR multiplier: linear warmup then cosine decay to 0, per README section 10."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _resolve_dir(data_cfg: dict, dataset: str) -> str:
    entry = data_cfg[dataset]
    return entry["synthetic_dir"] if entry["source"] == "synthetic" else entry["real_dir"]


def build_cnn_dataset(data_cfg: dict, dataset: str) -> FaultWindowDataset:
    records = load_cwru_dataset(_resolve_dir(data_cfg, dataset), source=data_cfg[dataset]["source"])
    window_cfg = data_cfg["window"]
    return FaultWindowDataset(
        records,
        window_seconds=window_cfg["window_seconds"],
        overlap=window_cfg["overlap"],
        target_rate=window_cfg.get("target_sample_rate_hz"),
        n_channels=window_cfg["n_channels"],
    )


def build_multitask_dataset(data_cfg: dict, dataset_fault: str, dataset_rul: str, seq_len: int) -> ConcatDataset:
    window_cfg = data_cfg["window"]
    fault_records = load_cwru_dataset(
        _resolve_dir(data_cfg, dataset_fault), source=data_cfg[dataset_fault]["source"]
    )
    fault_ds = SequenceFaultDataset(
        fault_records,
        seq_len=seq_len,
        window_seconds=window_cfg["window_seconds"],
        overlap=window_cfg["overlap"],
        target_rate=window_cfg.get("target_sample_rate_hz"),
        n_channels=window_cfg["n_channels"],
    )
    rul_records = load_ims_dataset(
        _resolve_dir(data_cfg, dataset_rul), source=data_cfg[dataset_rul]["source"]
    )
    rul_ds = SequenceRULDataset(
        rul_records,
        seq_len=seq_len,
        n_channels=window_cfg["n_channels"],
        window_seconds=window_cfg["window_seconds"],
        target_rate=window_cfg.get("target_sample_rate_hz"),
    )
    return ConcatDataset([fault_ds, rul_ds])


def train_cnn(args, data_cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_cnn_dataset(data_cfg, args.dataset_fault)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)

    model = TurboGuardCNN(n_channels=data_cfg["window"]["n_channels"], n_classes=args.n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: cosine_warmup_lr(e, args.epochs, args.warmup_epochs)
    )

    history = []
    model.train()
    for epoch in range(args.epochs):
        epoch_loss, n_batches = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if args.augment:
                x = augment_batch(x)
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history.append({"epoch": epoch, "loss": epoch_loss / max(1, n_batches)})
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, args.epochs, history[-1]["loss"])

    return {"history": history, "model": model, "n_samples": len(dataset)}


def train_hybrid(args, data_cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_multitask_dataset(data_cfg, args.dataset_fault, args.dataset_rul, args.seq_len)
    loader = DataLoader(
        dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True, collate_fn=collate_sequence_samples
    )

    model = TurboGuardHybrid(n_channels=data_cfg["window"]["n_channels"], n_classes=args.n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: cosine_warmup_lr(e, args.epochs, args.warmup_epochs)
    )

    history = []
    model.train()
    for epoch in range(args.epochs):
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            windows = batch["windows"].to(device)
            fault_labels = batch["fault_labels"].to(device)
            rul_targets = batch["rul_targets"].to(device)
            has_fault = batch["has_fault"].to(device)
            has_rul = batch["has_rul"].to(device)
            if args.augment:
                windows = augment_batch(windows)

            optimizer.zero_grad()
            fault_logits, rul_pred = model(windows)

            loss = torch.tensor(0.0, device=device)
            if has_fault.any():
                loss = loss + args.fault_weight * torch.nn.functional.cross_entropy(
                    fault_logits[has_fault], fault_labels[has_fault]
                )
            if has_rul.any():
                loss = loss + args.rul_weight * torch.nn.functional.huber_loss(
                    rul_pred[has_rul], rul_targets[has_rul]
                )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history.append({"epoch": epoch, "loss": epoch_loss / max(1, n_batches)})
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, args.epochs, history[-1]["loss"])

    return {"history": history, "model": model, "n_samples": len(dataset)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["turboguard_cnn", "turboguard_hybrid"], required=True)
    parser.add_argument("--dataset_fault", default="cwru")
    parser.add_argument("--dataset_rul", default="ims")
    parser.add_argument("--data_config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--fault_weight", type=float, default=1.0)
    parser.add_argument("--rul_weight", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    data_cfg = load_yaml(args.data_config)

    start = time.time()
    if args.model == "turboguard_cnn":
        result = train_cnn(args, data_cfg)
    else:
        result = train_hybrid(args, data_cfg)
    elapsed = time.time() - start

    output_dir = ensure_dir(args.output_dir)
    torch.save(result["model"].state_dict(), output_dir / "model.pt")
    save_json(
        {
            "final_loss": result["history"][-1]["loss"],
            "history": result["history"],
            "n_samples": result["n_samples"],
            "elapsed_seconds": elapsed,
        },
        output_dir / "metrics.json",
    )
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    save_yaml(args_dict | {"data_config_resolved": data_cfg}, output_dir / "config.yaml")
    logger.info("Saved model + metrics + config -> %s", output_dir)


if __name__ == "__main__":
    main()
