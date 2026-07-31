"""Classification and probability metrics for labeled held-out data."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predictions = (clipped >= threshold).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "rows": len(labels),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, clipped)),
        "average_precision": float(average_precision_score(labels, clipped)),
        "brier_score": float(brier_score_loss(labels, clipped)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }
