from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quora_duplicate_detection.data import (
    dataset_fingerprint,
    normalize_question,
    split_labeled_pairs,
    validate_pairs,
)


def test_normalize_question_handles_unicode_whitespace_and_missing() -> None:
    assert normalize_question("  CAFÉ\tQuestion  ") == "café question"
    assert normalize_question(None) == ""
    assert normalize_question(np.nan) == ""


def test_validate_pairs_fills_missing_questions_and_rejects_bad_labels() -> None:
    frame = pd.DataFrame(
        {
            "question1": [None, "A"],
            "question2": ["B", np.nan],
            "is_duplicate": [0, 1],
        }
    )
    clean = validate_pairs(frame, labeled=True)
    assert clean["question1"].tolist() == ["", "a"]
    assert clean["question2"].tolist() == ["b", ""]

    frame.loc[0, "is_duplicate"] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_pairs(frame, labeled=True)


def test_validate_pairs_rejects_missing_columns_and_one_class() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_pairs(pd.DataFrame({"question1": ["a"]}), labeled=False)
    with pytest.raises(ValueError, match="both classes"):
        validate_pairs(
            pd.DataFrame(
                {
                    "question1": ["a", "b"],
                    "question2": ["c", "d"],
                    "is_duplicate": [1, 1],
                }
            ),
            labeled=True,
        )


def test_split_is_deterministic_stratified_and_complete(pair_frame: pd.DataFrame) -> None:
    first = split_labeled_pairs(
        pair_frame,
        validation_size=0.2,
        test_size=0.2,
        random_seed=7,
    )
    second = split_labeled_pairs(
        pair_frame,
        validation_size=0.2,
        test_size=0.2,
        random_seed=7,
    )
    assert first.assignments.equals(second.assignments)
    assert set(first.assignments["split"]) == {"train", "validation", "test"}
    assert all(
        part["is_duplicate"].nunique() == 2 for part in (first.train, first.validation, first.test)
    )


def test_dataset_fingerprint_changes_with_content(pair_frame: pd.DataFrame) -> None:
    original = dataset_fingerprint(pair_frame)
    changed = pair_frame.copy()
    changed.loc[0, "question1"] = "changed"
    assert dataset_fingerprint(changed) != original


def test_too_small_dataset_cannot_be_stratified() -> None:
    tiny = pd.DataFrame(
        {
            "question1": ["a", "b"],
            "question2": ["a", "c"],
            "is_duplicate": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="provide more examples"):
        split_labeled_pairs(tiny, validation_size=0.2, test_size=0.2, random_seed=42)
