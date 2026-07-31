"""Probability calibration, learned score fusion, and threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ThresholdMetric = Literal["f1", "accuracy", "balanced_accuracy"]


@dataclass
class ScoreFusion:
    component_names: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float

    @classmethod
    def fit(
        cls,
        *,
        lexical_scores: np.ndarray,
        labels: np.ndarray,
        semantic_scores: np.ndarray | None,
        logistic_c: float,
        max_iter: int,
        random_seed: int,
    ) -> ScoreFusion:
        components = [np.asarray(lexical_scores, dtype=np.float64)]
        names = ["lexical_probability"]
        if semantic_scores is not None:
            components.append(np.asarray(semantic_scores, dtype=np.float64))
            names.append("semantic_cosine")
        matrix = np.column_stack(components)
        if not np.isfinite(matrix).all():
            raise ValueError("component scores contain non-finite values")

        classifier = LogisticRegression(
            C=logistic_c,
            max_iter=max_iter,
            random_state=random_seed,
            solver="lbfgs",
        )
        classifier.fit(matrix, labels)
        return cls(
            component_names=tuple(names),
            coefficients=np.asarray(classifier.coef_[0], dtype=np.float64),
            intercept=float(classifier.intercept_[0]),
        )

    def predict_proba(
        self,
        *,
        lexical_scores: np.ndarray,
        semantic_scores: np.ndarray | None,
    ) -> np.ndarray:
        components = [np.asarray(lexical_scores, dtype=np.float64)]
        if "semantic_cosine" in self.component_names:
            if semantic_scores is None:
                raise ValueError("this artifact requires semantic scores")
            components.append(np.asarray(semantic_scores, dtype=np.float64))
        elif semantic_scores is not None:
            raise ValueError("this artifact was trained without semantic scores")
        matrix = np.column_stack(components)
        if matrix.shape[1] != len(self.component_names):
            raise ValueError("component score shape does not match the fusion model")
        return expit(matrix @ self.coefficients + self.intercept)

    def validate(self) -> None:
        if self.component_names not in {
            ("lexical_probability",),
            ("lexical_probability", "semantic_cosine"),
        }:
            raise ValueError("unsupported fusion component list")
        if self.coefficients.shape != (len(self.component_names),):
            raise ValueError("fusion coefficients have an invalid shape")
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercept):
            raise ValueError("fusion parameters contain non-finite values")


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric: ThresholdMetric,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")

    metric_functions = {
        "f1": lambda observed, predicted: f1_score(observed, predicted, zero_division=0),
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }
    scorer = metric_functions[metric]
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.01, 0.99, 197):
        predictions = (probabilities >= threshold).astype(np.int8)
        score = float(scorer(labels, predictions))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, best_score
