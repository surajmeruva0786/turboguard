from src.data.ims_loader import load_ims_dataset

SYNTHETIC_DIR = "data/raw/ims/synthetic"


def test_load_ims_synthetic_all_bearings():
    records = load_ims_dataset(SYNTHETIC_DIR, source="synthetic")
    assert len(records) == 40  # 2 bearings x 20 snapshots
    assert {r.bearing_id for r in records} == {1, 2}


def test_load_ims_synthetic_single_bearing_sorted_by_snapshot():
    records = load_ims_dataset(SYNTHETIC_DIR, source="synthetic", bearing_id=1)
    assert len(records) == 20
    indices = [r.snapshot_index for r in records]
    assert indices == sorted(indices)


def test_rul_cycles_decreases_to_zero_at_last_snapshot():
    records = load_ims_dataset(SYNTHETIC_DIR, source="synthetic", bearing_id=1)
    assert records[-1].rul_cycles == 0
    assert records[0].rul_cycles == len(records) - 1


def test_health_indicator_increases_over_trajectory():
    records = load_ims_dataset(SYNTHETIC_DIR, source="synthetic", bearing_id=2)
    his = [r.health_indicator for r in records]
    assert his[-1] > his[0]
    assert his[-1] == max(his)


def test_signal_shape():
    records = load_ims_dataset(SYNTHETIC_DIR, source="synthetic", bearing_id=1)
    for r in records:
        assert r.signal.shape[0] == 3  # triaxial
        assert r.signal.shape[1] > 0
