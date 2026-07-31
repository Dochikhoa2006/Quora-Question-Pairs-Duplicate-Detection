"""Validated, serializable training configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    threshold_metric: str = "f1"
    logistic_c: float = 1.0
    max_iter: int = 1000
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    tfidf_max_features: int = 50_000
    tfidf_min_df: int = 2
    semantic_batch_size: int = 64

    def __post_init__(self) -> None:
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size + test_size must be less than 1")
        if self.threshold_metric not in {"f1", "accuracy", "balanced_accuracy"}:
            raise ValueError("threshold_metric must be f1, accuracy, or balanced_accuracy")
        if self.logistic_c <= 0:
            raise ValueError("logistic_c must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not 1 <= self.char_ngram_min <= self.char_ngram_max:
            raise ValueError("character n-gram bounds are invalid")
        if self.tfidf_max_features < 1 or self.tfidf_min_df < 1:
            raise ValueError("TF-IDF settings must be positive")
        if self.semantic_batch_size < 1:
            raise ValueError("semantic_batch_size must be positive")

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
