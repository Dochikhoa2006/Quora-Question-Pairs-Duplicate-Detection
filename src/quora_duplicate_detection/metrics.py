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


def calibration_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Return fixed-width reliability bins and count-weighted calibration error."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be one-dimensional with the same shape")
    if len(labels) == 0:
        raise ValueError("calibration requires at least one labeled row")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")

    bin_ids = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    counts = np.bincount(bin_ids, minlength=bins)
    probability_sums = np.bincount(bin_ids, weights=probabilities, minlength=bins)
    label_sums = np.bincount(bin_ids, weights=labels, minlength=bins)
    reliability = []
    weighted_error = 0.0
    for index, count in enumerate(counts):
        mean_probability = float(probability_sums[index] / count) if count else None
        observed_rate = float(label_sums[index] / count) if count else None
        if count:
            weighted_error += count * abs(mean_probability - observed_rate)
        reliability.append(
            {
                "lower_bound": index / bins,
                "upper_bound": (index + 1) / bins,
                "count": int(count),
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
            }
        )
    return {
        "binning": "equal_width",
        "bin_count": bins,
        "expected_calibration_error": float(weighted_error / len(labels)),
        "reliability_bins": reliability,
    }


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    calibration = calibration_summary(labels, probabilities)
    if len(np.unique(labels)) != 2:
        raise ValueError("evaluation requires both label classes")
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
        "calibration": calibration,
    }
