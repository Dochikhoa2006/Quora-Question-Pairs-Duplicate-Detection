from __future__ import annotations

import numpy as np
import pytest

from quora_duplicate_detection.ensemble import ScoreFusion, select_threshold
from quora_duplicate_detection.metrics import evaluate_probabilities


def test_fusion_learns_mixed_score_scales() -> None:
    lexical = np.array([0.05, 0.2, 0.7, 0.95])
    semantic = np.array([-0.8, -0.1, 0.4, 0.9])
    labels = np.array([0, 0, 1, 1])
    fusion = ScoreFusion.fit(
        lexical_scores=lexical,
        semantic_scores=semantic,
        labels=labels,
        logistic_c=1.0,
        max_iter=500,
        random_seed=42,
    )
    probabilities = fusion.predict_proba(
        lexical_scores=lexical,
        semantic_scores=semantic,
    )
    assert fusion.component_names == ("lexical_probability", "semantic_cosine")
    assert np.all(np.diff(probabilities) > 0)


def test_threshold_direction_is_greater_than_or_equal() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold, score = select_threshold(labels, probabilities, metric="f1")
    report = evaluate_probabilities(labels, probabilities, threshold=threshold)
    assert score == 1.0
    assert report["f1"] == 1.0
    assert (probabilities >= threshold).tolist() == [False, False, True, True]


def test_score_and_metric_shape_validation() -> None:
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="same shape"):
        select_threshold(labels, np.array([0.1]), metric="accuracy")
    with pytest.raises(ValueError, match="same shape"):
        evaluate_probabilities(labels, np.array([0.1]), threshold=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_probabilities(labels, np.array([0.1, 0.9]), threshold=1.0)


def test_fusion_requires_the_components_it_was_trained_with() -> None:
    fusion = ScoreFusion(
        component_names=("lexical_probability", "semantic_cosine"),
        coefficients=np.array([1.0, 1.0]),
        intercept=0.0,
    )
    with pytest.raises(ValueError, match="requires semantic"):
        fusion.predict_proba(lexical_scores=np.array([0.5]), semantic_scores=None)

    lexical_only = ScoreFusion(
        component_names=("lexical_probability",),
        coefficients=np.array([1.0]),
        intercept=0.0,
    )
    with pytest.raises(ValueError, match="trained without semantic"):
        lexical_only.predict_proba(
            lexical_scores=np.array([0.5]),
            semantic_scores=np.array([0.2]),
        )
