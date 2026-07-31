import os

from src.utils.io import ensure_dir, load_json, load_yaml, save_json, save_yaml
from src.utils.reports import recommend_from_rul
from src.utils.seed import set_seed


def test_set_seed_reproducible():
    import random

    import numpy as np

    set_seed(123, deterministic_torch=False)
    a = (random.random(), np.random.rand())
    set_seed(123, deterministic_torch=False)
    b = (random.random(), np.random.rand())
    assert a == b


def test_yaml_roundtrip(tmp_path):
    data = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    path = tmp_path / "sub" / "cfg.yaml"
    save_yaml(data, path)
    assert load_yaml(path) == data


def test_json_roundtrip_with_numpy(tmp_path):
    import numpy as np

    data = {"acc": np.float64(0.97), "arr": np.array([1, 2, 3])}
    path = tmp_path / "metrics.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded["acc"] == 0.97
    assert loaded["arr"] == [1, 2, 3]


def test_ensure_dir_creates_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result == target
    assert os.path.isdir(target)


def test_recommend_from_rul_urgency_buckets():
    assert recommend_from_rul("A1", 2.0, lead_time_cycles=20).urgency == "immediate"
    assert recommend_from_rul("A1", 10.0, lead_time_cycles=20).urgency == "urgent"
    assert recommend_from_rul("A1", 30.0, lead_time_cycles=20).urgency == "plan_soon"
    assert recommend_from_rul("A1", 100.0, lead_time_cycles=20).urgency == "routine"
