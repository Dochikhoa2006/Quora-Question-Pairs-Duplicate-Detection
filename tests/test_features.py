from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quora_duplicate_detection.features import FEATURE_NAMES, PairFeatureExtractor


def test_features_are_symmetric_finite_and_documented(pair_frame: pd.DataFrame) -> None:
    extractor = PairFeatureExtractor(min_df=1, max_features=2_000)
    features = extractor.fit_transform(pair_frame)
    swapped = pair_frame.rename(columns={"question1": "question2", "question2": "question1"})
    swapped_features = extractor.transform(swapped)

    assert features.shape == (len(pair_frame), len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features, swapped_features, atol=1e-12)


def test_empty_and_missing_questions_do_not_divide_by_zero() -> None:
    frame = pd.DataFrame(
        {
            "question1": [None, "", "Question 42"],
            "question2": ["", None, "Question 42"],
        }
    )
    extractor = PairFeatureExtractor(min_df=1)
    features = extractor.fit_transform(frame)

    assert np.isfinite(features).all()
    assert features[0, FEATURE_NAMES.index("character_length_ratio")] == 1.0
    assert features[2, FEATURE_NAMES.index("numeric_token_overlap")] == 1.0
    assert features[2, FEATURE_NAMES.index("exact_normalized_match")] == 1.0


def test_tiny_empty_corpus_uses_safe_tfidf_fallback() -> None:
    frame = pd.DataFrame({"question1": [None, ""], "question2": ["", None]})
    extractor = PairFeatureExtractor(min_df=2)
    features = extractor.fit_transform(frame)
    assert features.shape == (2, len(FEATURE_NAMES))


def test_unfitted_and_invalid_feature_state_is_rejected() -> None:
    extractor = PairFeatureExtractor()
    frame = pd.DataFrame({"question1": ["a"], "question2": ["b"]})
    with pytest.raises(RuntimeError, match="not fitted"):
        extractor.transform(frame)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = extractor.vocabulary
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = extractor.idf
    with pytest.raises(ValueError, match="does not match"):
        PairFeatureExtractor.from_state(
            vocabulary={"a": 0},
            idf=[],
            ngram_range=(3, 5),
            max_features=10,
            min_df=1,
        )
