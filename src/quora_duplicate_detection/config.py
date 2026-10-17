"""Validated, serializable training configuration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    split_strategy: str = "pair_stratified"
    threshold_metric: str = "f1"
    logistic_c: float = 1.0
    max_iter: int = 1000
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    tfidf_max_features: int = 50_000
    tfidf_min_df: int = 2
    semantic_batch_size: int = 64

    def __post_init__(self) -> None:
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed < 2**32
        ):
            raise ValueError("random_seed must be an integer between 0 and 2**32 - 1")
        if self.split_strategy not in {"pair_stratified", "question_disjoint"}:
            raise ValueError("split_strategy must be pair_stratified or question_disjoint")
        if not _finite_number(self.validation_size) or not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if not _finite_number(self.test_size) or not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size + test_size must be less than 1")
        if self.threshold_metric not in {"f1", "accuracy", "balanced_accuracy"}:
            raise ValueError("threshold_metric must be f1, accuracy, or balanced_accuracy")
        if not _finite_number(self.logistic_c) or self.logistic_c <= 0:
            raise ValueError("logistic_c must be finite and positive")
        if not _positive_int(self.max_iter):
            raise ValueError("max_iter must be a positive integer")
        if (
            not _positive_int(self.char_ngram_min)
            or not _positive_int(self.char_ngram_max)
            or self.char_ngram_min > self.char_ngram_max
        ):
            raise ValueError("character n-gram bounds must be positive integers in order")
        if not _positive_int(self.tfidf_max_features) or not _positive_int(self.tfidf_min_df):
            raise ValueError("TF-IDF settings must be positive integers")
        if not _positive_int(self.semantic_batch_size):
            raise ValueError("semantic_batch_size must be a positive integer")

    @classmethod
    def from_json(cls, path: str | Path | None) -> TrainingConfig:
        if path is None:
            return cls()
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
