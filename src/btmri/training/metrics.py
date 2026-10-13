"""Classification metrics.

Accuracy alone is misleading on imbalanced data (glioma is 46% of slices), so we also report
balanced accuracy and macro F1 (each class counts equally), per-class sensitivity (recall)
and specificity, which are the numbers clinicians care about, and ROC-AUC.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Scalar metrics that can be averaged across folds and used as ``train.monitor``.
SCALAR_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "loss")


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str]
) -> dict:
    """``y_true``: (N,) class indices. ``y_prob``: (N, C) predicted probabilities."""
    labels = list(range(len(class_names)))
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels))
    except ValueError:  # a class is missing from y_true (only happens on tiny debug runs)
        macro_auc = float("nan")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, name in enumerate(class_names):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        per_class[name] = {
            "precision": float(precision[i]),
            "sensitivity": float(recall[i]),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "macro_auc": macro_auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
