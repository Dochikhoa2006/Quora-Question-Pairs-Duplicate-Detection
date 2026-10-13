from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quora_duplicate_detection import cli, pipeline
from quora_duplicate_detection.compare import compare_artifacts
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.pipeline import train_pipeline


@pytest.fixture()
def comparison_inputs(tmp_path, pair_frame: pd.DataFrame, monkeypatch):
    training_path = tmp_path / "train.csv"
    external_path = tmp_path / "external.csv"
    pair_frame.to_csv(training_path, index=False)
    external = pair_frame.copy()
    for column in ("question1", "question2"):
        external[column] = "external " + external[column]
    external.to_csv(external_path, index=False)

    def fake_semantic_scores(frame, *, semantic_model, batch_size):
        if semantic_model is None:
            return None
        return np.asarray(frame["question1"].str.len(), dtype=np.float64) / 100

    monkeypatch.setattr(pipeline, "_semantic_scores", fake_semantic_scores)
    config = TrainingConfig(tfidf_min_df=1, tfidf_max_features=2_000)
    lexical_dir = tmp_path / "lexical"
    hybrid_dir = tmp_path / "hybrid"
    train_pipeline(data_path=training_path, output_dir=lexical_dir, config=config)
    train_pipeline(
        data_path=training_path,
        output_dir=hybrid_dir,
        config=config,
        semantic_model="stub-model",
    )
    return training_path, external_path, lexical_dir, hybrid_dir


def test_compare_artifacts_is_paired_reproducible_and_aggregate(
    comparison_inputs, tmp_path
) -> None:
    _, external_path, lexical_dir, hybrid_dir = comparison_inputs
    options = {
        "data_path": external_path,
        "models": {"lexical": lexical_dir, "hybrid": hybrid_dir},
        "baseline": "lexical",
        "bootstrap_samples": 100,
        "random_seed": 17,
    }
    first = compare_artifacts(**options, output_path=tmp_path / "first.json")
    second = compare_artifacts(**options, output_path=tmp_path / "second.json")
    assert first == second
    assert json.loads((tmp_path / "first.json").read_text(encoding="utf-8")) == first
    assert first["rows"] > 0
    assert first["models"]["lexical"]["components"] == ["lexical_probability"]
    assert "semantic_cosine" in first["models"]["hybrid"]["components"]
    difference = first["comparisons"]["hybrid"]["candidate_minus_baseline"]
    assert difference["brier_score"] == pytest.approx(
        first["models"]["hybrid"]["metrics"]["brier_score"]
        - first["models"]["lexical"]["metrics"]["brier_score"]
    )
    bounds = first["comparisons"]["hybrid"]["paired_bootstrap_percentiles"]
    assert all(value["p2_5"] <= value["p97_5"] for value in bounds.values())
    assert "question1" not in json.dumps(first)
    first_question = pd.read_csv(external_path).loc[0, "question1"]
    assert first_question not in json.dumps(first)


def test_compare_rejects_exact_training_data_and_does_not_write(
    comparison_inputs, tmp_path
) -> None:
    training_path, _, lexical_dir, hybrid_dir = comparison_inputs
    output = tmp_path / "leaky.json"
    with pytest.raises(ValueError, match="matches training data"):
        compare_artifacts(
            data_path=training_path,
            output_path=output,
            models={"lexical": lexical_dir, "hybrid": hybrid_dir},
            baseline="lexical",
            bootstrap_samples=100,
        )
    assert not output.exists()


def test_compare_cli_parses_labeled_artifacts(comparison_inputs, tmp_path, capsys) -> None:
    _, external_path, lexical_dir, hybrid_dir = comparison_inputs
    output = tmp_path / "comparison.json"
    cli.main(
        [
            "compare-artifacts",
            "--data",
            str(external_path),
            "--output",
            str(output),
            "--model",
            f"lexical={lexical_dir}",
            "--model",
            f"hybrid={hybrid_dir}",
            "--baseline",
            "lexical",
            "--bootstrap-samples",
            "100",
        ]
    )
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text(encoding="utf-8"))


def test_compare_validation_rejects_invalid_setup_before_reading_data(tmp_path) -> None:
    base = {"data_path": tmp_path / "missing.csv", "output_path": tmp_path / "unused.json"}
    with pytest.raises(ValueError, match="at least two models"):
        compare_artifacts(**base, models={"only": tmp_path}, baseline="only")
    with pytest.raises(ValueError, match="unknown model label"):
        compare_artifacts(
            **base,
            models={"a": tmp_path, "b": tmp_path},
            baseline="a",
            semantic_models={"other": "model"},
        )
    with pytest.raises(ValueError, match="bootstrap_samples"):
        compare_artifacts(
            **base,
            models={"a": tmp_path, "b": tmp_path},
            baseline="a",
            bootstrap_samples=10,
        )
    with pytest.raises(ValueError, match="unique LABEL=PATH"):
        cli._labeled_paths(["a=one", "a=two"], option="--model")
