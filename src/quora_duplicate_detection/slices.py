"""Aggregate labeled error analysis without emitting question text or row IDs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quora_duplicate_detection.data import PAIR_COLUMNS


def analyze_slices(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    min_rows: int = 20,
) -> dict[str, Any]:
    """Summarize predefined question-length and missing-text slices.

    Groups below ``min_rows`` are omitted entirely, including their counts.
    This is an aggregate disclosure guard, not differential privacy.
    """
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 2:
        raise ValueError("min_rows must be an integer of at least 2")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    labels = frame["is_duplicate"].to_numpy()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(frame),):
        raise ValueError("probabilities must have one value per row")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")

    lengths = np.minimum(
        frame[PAIR_COLUMNS[0]].str.split().str.len().to_numpy(),
        frame[PAIR_COLUMNS[1]].str.split().str.len().to_numpy(),
    )
    definitions = {
        "shortest_question_tokens": {
            "0": lengths == 0,
            "1_to_5": (lengths >= 1) & (lengths <= 5),
            "6_to_15": (lengths >= 6) & (lengths <= 15),
            "16_plus": lengths >= 16,
        },
        "empty_question": {
            "yes": lengths == 0,
            "no": lengths > 0,
        },
    }
    predictions = probabilities >= threshold
    output: dict[str, Any] = {}
    for dimension, groups in definitions.items():
        shown: dict[str, Any] = {}
        for name, mask in groups.items():
            count = int(mask.sum())
            if count < min_rows:
                continue
            observed = labels[mask]
            predicted = predictions[mask]
            positives = observed == 1
            negatives = ~positives
            shown[name] = {
                "rows": count,
                "positive_rate": float(positives.mean()),
                "mean_probability": float(probabilities[mask].mean()),
                "brier_score": float(np.square(probabilities[mask] - observed).mean()),
                "error_rate": float((predicted != positives).mean()),
                "false_positive_rate": (
                    float(predicted[negatives].mean()) if negatives.any() else None
                ),
                "false_negative_rate": (
                    float((~predicted[positives]).mean()) if positives.any() else None
                ),
            }
        output[dimension] = shown
    return {
        "minimum_rows": min_rows,
        "dimensions": output,
    }
