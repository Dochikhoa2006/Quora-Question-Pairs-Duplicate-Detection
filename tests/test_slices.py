from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quora_duplicate_detection.slices import analyze_slices


def test_slice_report_contains_only_aggregate_metrics_and_suppresses_small_groups() -> None:
    frame = pd.DataFrame(
        {
            "question1": ["", "one two", "one two", "one two", "one two"],
            "question2": ["private text", "three four", "three four", "three four", "three four"],
            "is_duplicate": [1, 0, 0, 1, 1],
        }
    )
    probabilities = np.array([0.9, 0.8, 0.2, 0.1, 0.9])
    report = analyze_slices(frame, probabilities, threshold=0.5, min_rows=3)

    assert set(report["dimensions"]["empty_question"]) == {"no"}
    ordinary = report["dimensions"]["empty_question"]["no"]
    assert ordinary["rows"] == 4
    assert ordinary["error_rate"] == 0.5
    assert ordinary["false_positive_rate"] == 0.5
    assert ordinary["false_negative_rate"] == 0.5
    assert report["dimensions"]["shortest_question_tokens"]["1_to_5"]["rows"] == 4
    assert "private text" not in str(report)


def test_slice_report_uses_null_rate_when_class_absent() -> None:
    frame = pd.DataFrame({"question1": ["a", "b"], "question2": ["c", "d"], "is_duplicate": [1, 1]})
    result = analyze_slices(frame, np.array([0.8, 0.4]), threshold=0.5, min_rows=2)
    assert result["dimensions"]["empty_question"]["no"]["false_positive_rate"] is None


def test_slice_report_rejects_invalid_minimum_and_probability_shape() -> None:
    frame = pd.DataFrame({"question1": ["a", "b"], "question2": ["c", "d"], "is_duplicate": [0, 1]})
    with pytest.raises(ValueError, match="at least 2"):
        analyze_slices(frame, np.array([0.1, 0.9]), threshold=0.5, min_rows=1)
    with pytest.raises(ValueError, match="one value per row"):
        analyze_slices(frame, np.array([0.1]), threshold=0.5)
