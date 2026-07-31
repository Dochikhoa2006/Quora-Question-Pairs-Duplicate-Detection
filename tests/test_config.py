from __future__ import annotations

import json

import pytest

from quora_duplicate_detection.config import TrainingConfig


def test_config_round_trip_and_unknown_keys(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"random_seed": 9, "tfidf_min_df": 1}), encoding="utf-8")
    config = TrainingConfig.from_json(path)
    assert config.random_seed == 9
    assert config.to_dict()["tfidf_min_df"] == 1
    assert TrainingConfig.from_json(None) == TrainingConfig()

    path.write_text(json.dumps({"unknown": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration"):
        TrainingConfig.from_json(path)

    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        TrainingConfig.from_json(path)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"validation_size": 0}, "validation_size"),
        ({"test_size": 1}, "test_size"),
        ({"validation_size": 0.6, "test_size": 0.5}, "less than 1"),
        ({"threshold_metric": "made-up"}, "threshold_metric"),
        ({"logistic_c": 0}, "logistic_c"),
        ({"max_iter": 0}, "max_iter"),
        ({"char_ngram_min": 5, "char_ngram_max": 3}, "n-gram"),
        ({"tfidf_max_features": 0}, "TF-IDF"),
        ({"tfidf_min_df": 0}, "TF-IDF"),
        ({"semantic_batch_size": 0}, "semantic_batch_size"),
    ],
)
def test_invalid_config_is_rejected(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**settings)
