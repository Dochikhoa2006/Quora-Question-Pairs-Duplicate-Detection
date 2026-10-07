from __future__ import annotations

import numpy as np
import pytest

from quora_duplicate_detection.metrics import calibration_summary, evaluate_probabilities


def test_reliability_bins_include_one_and_keep_empty_bins_explicit() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.0, 0.5, 0.5, 1.0])
    summary = calibration_summary(labels, probabilities, bins=2)

    assert [item["count"] for item in summary["reliability_bins"]] == [1, 3]
    assert summary["reliability_bins"][0]["mean_probability"] == 0.0
    assert summary["reliability_bins"][1]["observed_positive_rate"] == pytest.approx(2 / 3)
    assert summary["expected_calibration_error"] == 0.0

    sparse = calibration_summary(np.array([0, 1]), np.array([0.0, 1.0]), bins=4)
    assert sparse["expected_calibration_error"] == 0.0
    assert sparse["reliability_bins"][1]["count"] == 0
    assert sparse["reliability_bins"][1]["mean_probability"] is None
    assert sparse["reliability_bins"][1]["observed_positive_rate"] is None


@pytest.mark.parametrize("probabilities", [np.array([np.nan, 0.5]), np.array([-0.1, 1.1])])
def test_calibration_rejects_invalid_probabilities(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite values between 0 and 1"):
        calibration_summary(np.array([0, 1]), probabilities)


def test_calibration_rejects_invalid_labels_and_bin_count() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        calibration_summary(np.array([0, 2]), np.array([0.1, 0.9]))
    with pytest.raises(ValueError, match="positive integer"):
        calibration_summary(np.array([0, 1]), np.array([0.1, 0.9]), bins=0)


def test_evaluation_includes_calibration_without_changing_probability_metrics() -> None:
    report = evaluate_probabilities(np.array([0, 1]), np.array([0.25, 0.75]), threshold=0.5)
    assert report["brier_score"] == pytest.approx(0.0625)
    assert report["calibration"]["expected_calibration_error"] == pytest.approx(0.25)
    assert sum(item["count"] for item in report["calibration"]["reliability_bins"]) == 2
