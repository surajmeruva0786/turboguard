"""PyTorch ``Dataset`` classes bridging the loaders/preprocessing to the models.

Two shapes are needed (README section 9):

- Single-window ``(n_channels, window_samples)`` for the plain CNN and for
  classical feature extraction.
- Sequence-of-``K`` windows ``(K, n_channels, window_samples)`` for the
  hybrid CNN+Bi-LSTM multi-task model.

Note on synthetic CWRU data: each committed synthetic file is exactly one
1-second window (kept short to keep the repo small), so there is only ever
one *distinct* window per file. :class:`SequenceFaultDataset` pads short
recordings up to ``seq_len`` by repeating the last window rather than
failing — documented, not hidden — real multi-second CWRU recordings
naturally yield ``seq_len`` distinct windows instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cwru_loader import CWRURecord, align_channels
from src.data.ims_loader import IMSRecord
from src.preprocessing.conditioning import condition_signal
from src.preprocessing.windowing import window_signal

FAULT_CLASSES = ["healthy", "inner_race", "outer_race", "ball", "compound"]


def _condition_and_window(
    signal: np.ndarray,
    sample_rate: float,
    target_rate: float | None,
    window_seconds: float,
    overlap: float,
    n_channels: int,
) -> np.ndarray:
    if target_rate and target_rate != sample_rate:
        signal = condition_signal(signal, sample_rate, target_rate)
        sample_rate = target_rate
    signal = align_channels(signal, tuple(range(signal.shape[0])), n_channels=n_channels)
    return window_signal(signal, sample_rate, window_seconds, overlap)


class FaultWindowDataset(Dataset):
    """Single-window fault-classification dataset (for TurboGuard-CNN)."""

    def __init__(
        self,
        records: list[CWRURecord],
        window_seconds: float = 1.0,
        overlap: float = 0.5,
        target_rate: float | None = None,
        n_channels: int = 3,
        class_names: list[str] = FAULT_CLASSES,
    ):
        self.class_names = class_names
        self.samples: list[tuple[np.ndarray, int]] = []
        for r in records:
            windows = _condition_and_window(
                r.signal, r.sample_rate_hz, target_rate, window_seconds, overlap, n_channels
            )
            label_idx = class_names.index(r.health_state)
            for w in windows:
                self.samples.append((w.astype(np.float32), label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        window, label = self.samples[idx]
        return torch.from_numpy(window), label


def _pad_to_length(windows: np.ndarray, seq_len: int) -> np.ndarray:
    if len(windows) >= seq_len:
        return windows[:seq_len]
    if len(windows) == 0:
        raise ValueError("Recording produced zero windows; cannot build a sequence.")
    pad = np.repeat(windows[-1:], seq_len - len(windows), axis=0)
    return np.concatenate([windows, pad], axis=0)


@dataclass
class SequenceSample:
    windows: np.ndarray  # (seq_len, n_channels, window_samples)
    fault_label: int | None  # None if this sample has no fault supervision (RUL-only)
    rul_target: float | None  # None if this sample has no RUL supervision (fault-only)


class SequenceFaultDataset(Dataset):
    """Sequence-of-windows dataset for fault classification (hybrid model, CWRU side)."""

    def __init__(
        self,
        records: list[CWRURecord],
        seq_len: int = 10,
        window_seconds: float = 1.0,
        overlap: float = 0.5,
        target_rate: float | None = None,
        n_channels: int = 3,
        class_names: list[str] = FAULT_CLASSES,
    ):
        self.class_names = class_names
        self.samples: list[SequenceSample] = []
        for r in records:
            windows = _condition_and_window(
                r.signal, r.sample_rate_hz, target_rate, window_seconds, overlap, n_channels
            )
            seq = _pad_to_length(windows, seq_len).astype(np.float32)
            label_idx = class_names.index(r.health_state)
            self.samples.append(SequenceSample(windows=seq, fault_label=label_idx, rul_target=None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SequenceSample:
        return self.samples[idx]


class SequenceRULDataset(Dataset):
    """Sequence-of-windows dataset for RUL regression (hybrid model, IMS side).

    Each sample is a sliding window of ``seq_len`` *consecutive snapshots*
    from one bearing's run-to-failure trajectory (snapshots are already
    ~1 window each — see module docstring), with the RUL target taken at
    the last timestep of the sequence.
    """

    def __init__(self, records: list[IMSRecord], seq_len: int = 10, stride: int = 1, n_channels: int = 3):
        by_bearing: dict[int, list[IMSRecord]] = {}
        for r in records:
            by_bearing.setdefault(r.bearing_id, []).append(r)

        self.samples: list[SequenceSample] = []
        for bearing_records in by_bearing.values():
            bearing_records = sorted(bearing_records, key=lambda r: r.snapshot_index)
            signals = [align_channels(r.signal, (), n_channels=n_channels).astype(np.float32) for r in bearing_records]
            for end in range(seq_len - 1, len(signals), stride):
                start = end - seq_len + 1
                seq = np.stack(signals[start : end + 1])
                self.samples.append(
                    SequenceSample(
                        windows=seq, fault_label=None, rul_target=float(bearing_records[end].rul_cycles)
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SequenceSample:
        return self.samples[idx]


def collate_sequence_samples(batch: list[SequenceSample]) -> dict[str, torch.Tensor]:
    """Collate a batch of :class:`SequenceSample`, masking absent labels/targets
    (rather than using a fill value that could look like a valid label) so the
    multi-task loss can select only the supervised terms per sample."""
    windows = torch.from_numpy(np.stack([s.windows for s in batch]))
    has_fault = torch.tensor([s.fault_label is not None for s in batch])
    has_rul = torch.tensor([s.rul_target is not None for s in batch])
    fault_labels = torch.tensor([s.fault_label if s.fault_label is not None else -1 for s in batch])
    rul_targets = torch.tensor(
        [s.rul_target if s.rul_target is not None else 0.0 for s in batch], dtype=torch.float32
    )
    return {
        "windows": windows,
        "fault_labels": fault_labels,
        "rul_targets": rul_targets,
        "has_fault": has_fault,
        "has_rul": has_rul,
    }
