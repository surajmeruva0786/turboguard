import numpy as np

from src.evaluation.classification import classification_metrics


def test_perfect_predictions_give_accuracy_and_f1_of_one():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    result = classification_metrics(y_true, y_pred, class_names=["a", "b", "c"])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert np.array_equal(np.array(result["confusion_matrix"]), np.eye(3, dtype=int) * 2)


def test_per_class_metrics_reflect_confusion():
    # class "a" always confused with "b"; "b" and "c" always correct
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 1, 1, 1, 2, 2])
    result = classification_metrics(y_true, y_pred, class_names=["a", "b", "c"])
    assert result["per_class"]["a"]["recall"] == 0.0
    assert result["per_class"]["c"]["recall"] == 1.0
    assert result["per_class"]["a"]["support"] == 2


def test_roc_auc_present_when_proba_given():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 100)
    proba = np.zeros((100, 3))
    for i, label in enumerate(y_true):
        proba[i, label] = 0.9
        others = [c for c in range(3) if c != label]
        proba[i, others] = 0.05
    y_pred = proba.argmax(axis=1)
    result = classification_metrics(y_true, y_pred, y_proba=proba, class_names=["h", "f1", "f2"])
    assert result["roc_auc_ovr"] is not None
    assert result["roc_auc_ovr"] > 0.9


def test_confusion_matrix_shape_matches_class_count():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    result = classification_metrics(y_true, y_pred, class_names=["a", "b", "c", "d", "e"])
    cm = np.array(result["confusion_matrix"])
    assert cm.shape == (5, 5)
