"""Input validation, normalization, fingerprints, and deterministic splits."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PAIR_COLUMNS = ("question1", "question2")
LABEL_COLUMN = "is_duplicate"


def normalize_question(value: object) -> str:
    """Normalize a question without requiring language-specific resources."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def validate_pairs(frame: pd.DataFrame, *, labeled: bool) -> pd.DataFrame:
    required = set(PAIR_COLUMNS)
    if labeled:
        required.add(LABEL_COLUMN)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    clean = frame.copy()
    for column in PAIR_COLUMNS:
        clean[column] = clean[column].map(normalize_question)

    if labeled:
        if clean[LABEL_COLUMN].isna().any():
            raise ValueError("is_duplicate contains missing values")
        numeric = pd.to_numeric(clean[LABEL_COLUMN], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError("is_duplicate must contain only 0 and 1")
        clean[LABEL_COLUMN] = numeric.astype(np.int8)
        if clean[LABEL_COLUMN].nunique() != 2:
            raise ValueError("labeled data must contain both classes")

    return clean


def load_pairs(path: str | Path, *, labeled: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    return validate_pairs(frame, labeled=labeled)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    columns = [*PAIR_COLUMNS]
    if LABEL_COLUMN in frame:
        columns.append(LABEL_COLUMN)
    row_hashes = pd.util.hash_pandas_object(frame[columns], index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    assignments: pd.DataFrame


def split_labeled_pairs(
    frame: pd.DataFrame,
    *,
    validation_size: float,
    test_size: float,
    random_seed: int,
    strategy: str = "pair_stratified",
) -> DataSplits:
    """Create reproducible partitions with an untouched test set."""
    indices = np.arange(len(frame))
    labels = frame[LABEL_COLUMN].to_numpy()
    if strategy == "question_disjoint":
        train_idx, validation_idx, test_idx = _question_disjoint_indices(
            frame, labels, validation_size, test_size, random_seed
        )
    elif strategy == "pair_stratified":
        train_idx, validation_idx, test_idx = _pair_stratified_indices(
            indices, labels, validation_size, test_size, random_seed
        )
    else:
        raise ValueError(f"unsupported split strategy: {strategy}")

    assignments = pd.DataFrame({"row_number": indices, "split": ""})
    assignments.loc[train_idx, "split"] = "train"
    assignments.loc[validation_idx, "split"] = "validation"
    assignments.loc[test_idx, "split"] = "test"

    return DataSplits(
        train=frame.iloc[train_idx].reset_index(drop=True),
        validation=frame.iloc[validation_idx].reset_index(drop=True),
        test=frame.iloc[test_idx].reset_index(drop=True),
        assignments=assignments.sort_values("row_number").reset_index(drop=True),
    )


def _pair_stratified_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        development_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels,
        )
        relative_validation_size = validation_size / (1.0 - test_size)
        train_idx, validation_idx = train_test_split(
            development_idx,
            test_size=relative_validation_size,
            random_state=random_seed,
            stratify=labels[development_idx],
        )
    except ValueError as exc:
        raise ValueError(
            "unable to create stratified splits; provide more examples of each class"
        ) from exc
    return train_idx, validation_idx, test_idx


def _question_components(frame: pd.DataFrame) -> np.ndarray:
    """Union every pair that shares a normalized question, including transitive links."""
    parent = np.arange(len(frame))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    seen: dict[str, int] = {}
    for row, pair in enumerate(frame[list(PAIR_COLUMNS)].itertuples(index=False, name=None)):
        for question in pair:
            previous = seen.setdefault(question, row)
            parent[root(row)] = root(previous)
    return np.fromiter((root(i) for i in range(len(frame))), dtype=np.int64)


def _question_disjoint_indices(
    frame: pd.DataFrame,
    labels: np.ndarray,
    validation_size: float,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = _question_components(frame)
    if len(np.unique(groups)) < 3:
        raise ValueError("question-disjoint splitting requires at least three components")

    indices = np.arange(len(frame))
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_error = float("inf")
    generator = np.random.default_rng(random_seed)
    # Sample group partitions, then choose the valid one closest to the requested
    # row fractions and class balance. A fixed seed makes the search reproducible.
    for _ in range(128):
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=int(generator.integers(0, 2**32 - 1)),
        )
        development_idx, test_idx = next(outer.split(indices, labels, groups))
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size / (1 - test_size),
            random_state=int(generator.integers(0, 2**32 - 1)),
        )
        try:
            train_relative, validation_relative = next(
                inner.split(development_idx, labels[development_idx], groups[development_idx])
            )
        except ValueError:
            continue
        train_idx = development_idx[train_relative]
        validation_idx = development_idx[validation_relative]
        parts = (train_idx, validation_idx, test_idx)
        if any(len(part) == 0 or len(np.unique(labels[part])) != 2 for part in parts):
            continue
        sizes = np.array([len(part) / len(frame) for part in parts])
        targets = np.array([1 - validation_size - test_size, validation_size, test_size])
        rates = np.array([labels[part].mean() for part in parts])
        error = float(np.square(sizes - targets).sum() + np.square(rates - labels.mean()).sum())
        if error < best_error:
            best, best_error = parts, error
    if best is None:
        raise ValueError(
            "unable to create question-disjoint splits with both classes in each partition; "
            "provide more independent question components"
        )
    return best
