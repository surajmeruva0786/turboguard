import torch
from src.data.cwru_loader import load_cwru_dataset
from src.data.dataset import (
    FAULT_CLASSES,
    FaultWindowDataset,
    SequenceFaultDataset,
    SequenceRULDataset,
    collate_sequence_samples,
)
from src.data.ims_loader import load_ims_dataset

CWRU_DIR = "data/raw/cwru/synthetic"
IMS_DIR = "data/raw/ims/synthetic"


def test_fault_window_dataset_basic():
    records = load_cwru_dataset(CWRU_DIR, source="synthetic")
    ds = FaultWindowDataset(records)
    assert len(ds) == 20  # 1 window per 1s synthetic file
    window, label = ds[0]
    assert isinstance(window, torch.Tensor)
    assert window.shape == (3, 12000)
    assert 0 <= label < len(FAULT_CLASSES)


def test_sequence_fault_dataset_pads_short_recordings():
    records = load_cwru_dataset(CWRU_DIR, source="synthetic")
    ds = SequenceFaultDataset(records, seq_len=10)
    sample = ds[0]
    assert sample.windows.shape == (10, 3, 12000)
    # padded sequence should repeat the (only) window
    import numpy as np

    assert np.array_equal(sample.windows[0], sample.windows[-1])
    assert sample.fault_label is not None
    assert sample.rul_target is None


def test_sequence_rul_dataset_sliding_windows_and_targets():
    records = load_ims_dataset(IMS_DIR, source="synthetic")
    ds = SequenceRULDataset(records, seq_len=5, stride=1)
    # 2 bearings x (20 - 5 + 1) = 2 x 16 = 32 samples
    assert len(ds) == 32
    sample = ds[0]
    assert sample.windows.shape[0] == 5
    assert sample.fault_label is None
    assert sample.rul_target is not None


def test_sequence_rul_dataset_targets_decrease_along_trajectory():
    records = load_ims_dataset(IMS_DIR, source="synthetic", bearing_id=1)
    ds = SequenceRULDataset(records, seq_len=5, stride=1)
    targets = [ds[i].rul_target for i in range(len(ds))]
    assert targets == sorted(targets, reverse=True)
    assert targets[-1] == 0.0


def test_collate_sequence_samples_masks_absent_labels():
    fault_records = load_cwru_dataset(CWRU_DIR, source="synthetic")
    rul_records = load_ims_dataset(IMS_DIR, source="synthetic", bearing_id=1)

    fault_ds = SequenceFaultDataset(fault_records, seq_len=5)
    rul_ds = SequenceRULDataset(rul_records, seq_len=5)

    batch = [fault_ds[0], fault_ds[1], rul_ds[0], rul_ds[1]]
    collated = collate_sequence_samples(batch)

    assert collated["windows"].shape[0] == 4
    assert collated["has_fault"].tolist() == [True, True, False, False]
    assert collated["has_rul"].tolist() == [False, False, True, True]
