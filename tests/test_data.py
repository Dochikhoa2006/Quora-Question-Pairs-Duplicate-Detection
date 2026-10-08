from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from quora_duplicate_detection.data import (
    _question_components,
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


def test_question_disjoint_split_keeps_transitive_questions_together() -> None:
    rows = []
    for component in range(30):
        rows.extend(
            [
                (f"question {component} a", f"question {component} b", 1),
                (f"question {component} b", f"question {component} c", 0),
            ]
        )
    frame = pd.DataFrame(rows, columns=["question1", "question2", "is_duplicate"])
    first = split_labeled_pairs(
        frame,
        validation_size=0.2,
        test_size=0.2,
        random_seed=7,
        strategy="question_disjoint",
    )
    second = split_labeled_pairs(
        frame,
        validation_size=0.2,
        test_size=0.2,
        random_seed=7,
        strategy="question_disjoint",
    )
    assert first.assignments.equals(second.assignments)
    question_sets = [
        set(part["question1"]) | set(part["question2"])
        for part in (first.train, first.validation, first.test)
    ]
    assert all(question_sets[i].isdisjoint(question_sets[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert all(
        part["is_duplicate"].nunique() == 2 for part in (first.train, first.validation, first.test)
    )


def test_question_disjoint_split_rejects_single_component() -> None:
    frame = pd.DataFrame(
        [("a", "b", 1), ("b", "c", 0), ("c", "d", 1)],
        columns=["question1", "question2", "is_duplicate"],
    )
    with pytest.raises(ValueError, match="at least three components"):
        split_labeled_pairs(
            frame,
            validation_size=0.2,
            test_size=0.2,
            random_seed=7,
            strategy="question_disjoint",
        )


def _previous_grouped_assignments(
    frame: pd.DataFrame, seed: int, validation_size: float, test_size: float
) -> list[str]:
    """Reference row-level search used by split replay version 1."""
    groups = _question_components(frame)
    labels = frame["is_duplicate"].to_numpy()
    indices = np.arange(len(frame))
    generator = np.random.default_rng(seed)
    best = None
    best_error = float("inf")
    for _ in range(128):
        outer = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=int(generator.integers(0, 2**32 - 1))
        )
        development, test = next(outer.split(indices, labels, groups))
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size / (1 - test_size),
            random_state=int(generator.integers(0, 2**32 - 1)),
        )
        train_relative, validation_relative = next(
            inner.split(development, labels[development], groups[development])
        )
        parts = (development[train_relative], development[validation_relative], test)
        if any(len(np.unique(labels[part])) != 2 for part in parts):
            continue
        sizes = np.array([len(part) / len(frame) for part in parts])
        rates = np.array([labels[part].mean() for part in parts])
        targets = np.array([1 - validation_size - test_size, validation_size, test_size])
        error = float(np.square(sizes - targets).sum() + np.square(rates - labels.mean()).sum())
        if error < best_error:
            best, best_error = parts, error
    assert best is not None
    assignments = np.empty(len(frame), dtype=object)
    for name, part in zip(("train", "validation", "test"), best, strict=True):
        assignments[part] = name
    return assignments.tolist()


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
@pytest.mark.parametrize("validation_size,test_size", [(0.2, 0.2), (0.15, 0.15)])
def test_component_scoring_preserves_version_one_assignments(
    seed: int, validation_size: float, test_size: float
) -> None:
    rows = []
    for component in range(40):
        for member in range(1 + component % 5):
            rows.append(
                (
                    f"anchor {component}",
                    f"leaf {component} {member}",
                    (component + member) % 2,
                )
            )
    frame = pd.DataFrame(rows, columns=["question1", "question2", "is_duplicate"])
    optimized = split_labeled_pairs(
        frame,
        validation_size=validation_size,
        test_size=test_size,
        random_seed=seed,
        strategy="question_disjoint",
    )
    assert optimized.assignments["split"].tolist() == _previous_grouped_assignments(
        frame, seed, validation_size, test_size
    )
