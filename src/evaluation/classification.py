"""Fault-classification evaluation metrics (README section 11.2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix,
    and one-vs-rest ROC-AUC (if class probabilities are given)."""
    labels = list(range(len(class_names))) if class_names else None

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    result: dict = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }

    names = class_names or [str(i) for i in sorted(set(y_true) | set(y_pred))]
    result["per_class"] = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for name, p, r, f, s in zip(names, precision, recall, f1, support)
    }

    if y_proba is not None:
        try:
            result["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels))
        except ValueError:
            result["roc_auc_ovr"] = None

    return result
