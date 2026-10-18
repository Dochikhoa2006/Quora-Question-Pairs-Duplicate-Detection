from __future__ import annotations

import json

import pandas as pd
import pytest

from quora_duplicate_detection import pipeline as pipeline_module
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.lexical import LexicalModel
from quora_duplicate_detection.pipeline import (
    create_predictions,
    create_predictions_chunked,
    evaluate_labeled_file,
    train_pipeline,
)


def test_end_to_end_lexical_training_and_prediction(
    tmp_path,
    pair_frame: pd.DataFrame,
) -> None:
    training_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    artifact_dir = tmp_path / "artifact"
    submission_csv = tmp_path / "submission.csv"
    pair_frame.to_csv(training_csv, index=False)
    pair_frame.drop(columns=["is_duplicate"]).iloc[:7].to_csv(test_csv, index=False)

    evaluation = train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(
            validation_size=0.2,
            test_size=0.2,
            tfidf_max_features=2_000,
            tfidf_min_df=1,
        ),
    )
    predictions = create_predictions(
        input_path=test_csv,
        artifact_dir=artifact_dir,
        output_path=submission_csv,
    )

    assert evaluation["methodology"]["competition_test_labels_used"] is False
    assert evaluation["test"]["slices"]["minimum_rows"] == 20
    assert (
        sum(bin_["count"] for bin_ in evaluation["test"]["calibration"]["reliability_bins"])
        == evaluation["split_sizes"]["test"]
    )
    assert set(predictions.columns) == {"id", "is_duplicate"}
    assert predictions["is_duplicate"].between(0, 1).all()
    assert not any(artifact_dir.glob("*.pkl"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision"]["comparison"] == "probability >= threshold"
    assert manifest["training"]["split_replay_version"] == 1


def test_grouped_training_records_split_protocol(tmp_path, pair_frame: pd.DataFrame) -> None:
    training_csv = tmp_path / "train.csv"
    pair_frame.to_csv(training_csv, index=False)
    artifact_dir = tmp_path / "artifact"
    report = train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(
            split_strategy="question_disjoint",
            validation_size=0.2,
            test_size=0.2,
            tfidf_max_features=2_000,
            tfidf_min_df=1,
        ),
    )
    assert report["methodology"]["split_strategy"] == "question_disjoint"
    assignments = pd.read_csv(artifact_dir / "split_assignments.csv")
    pair_frame = pair_frame.assign(split=assignments["split"])
    partitions = [
        set(part["question1"]) | set(part["question2"]) for _, part in pair_frame.groupby("split")
    ]
    assert len(partitions) == 3
    assert all(partitions[i].isdisjoint(partitions[j]) for i, j in ((0, 1), (0, 2), (1, 2)))


@pytest.mark.parametrize("keep_id", [True, False])
def test_chunked_predictions_match_regular_output(
    tmp_path, pair_frame: pd.DataFrame, keep_id: bool
) -> None:
    training_csv = tmp_path / "train.csv"
    input_csv = tmp_path / "input.csv"
    artifact_dir = tmp_path / "artifact"
    regular_csv = tmp_path / "regular.csv"
    chunked_csv = tmp_path / "chunked.csv"
    pair_frame.to_csv(training_csv, index=False)
    columns = ["question1", "question2"]
    if keep_id:
        columns.insert(0, "id")
    pair_frame[columns].iloc[:11].to_csv(input_csv, index=False)
    train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(tfidf_min_df=1),
    )

    create_predictions(
        input_path=input_csv,
        artifact_dir=artifact_dir,
        output_path=regular_csv,
        include_binary_prediction=True,
    )
    rows = create_predictions_chunked(
        input_path=input_csv,
        artifact_dir=artifact_dir,
        output_path=chunked_csv,
        chunk_size=3,
        include_binary_prediction=True,
    )
    assert rows == 11
    pd.testing.assert_frame_equal(pd.read_csv(regular_csv), pd.read_csv(chunked_csv))
    assert next(iter(pd.read_csv(chunked_csv).columns)) == ("id" if keep_id else "test_id")


def test_chunked_predictions_preserve_existing_output_on_later_failure(
    tmp_path, pair_frame: pd.DataFrame, monkeypatch
) -> None:
    training_csv = tmp_path / "train.csv"
    input_csv = tmp_path / "input.csv"
    artifact_dir = tmp_path / "artifact"
    output_csv = tmp_path / "output.csv"
    pair_frame.to_csv(training_csv, index=False)
    pair_frame.iloc[:7].to_csv(input_csv, index=False)
    train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(tfidf_min_df=1),
    )
    output_csv.write_text("previous result\n", encoding="utf-8")
    original_predict = LexicalModel.predict_proba
    calls = 0

    def fail_second_chunk(self, frame):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("later chunk failed")
        return original_predict(self, frame)

    monkeypatch.setattr(LexicalModel, "predict_proba", fail_second_chunk)
    with pytest.raises(RuntimeError, match="later chunk failed"):
        create_predictions_chunked(
            input_path=input_csv,
            artifact_dir=artifact_dir,
            output_path=output_csv,
            chunk_size=3,
        )
    assert output_csv.read_text(encoding="utf-8") == "previous result\n"
    assert not list(tmp_path.glob(".qqdup-*.csv"))


def test_regular_predictions_preserve_existing_output_if_replace_fails(
    tmp_path, pair_frame: pd.DataFrame, monkeypatch
) -> None:
    training_csv = tmp_path / "train.csv"
    input_csv = tmp_path / "input.csv"
    artifact_dir = tmp_path / "artifact"
    output_csv = tmp_path / "output.csv"
    pair_frame.to_csv(training_csv, index=False)
    pair_frame.iloc[:3].to_csv(input_csv, index=False)
    train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(tfidf_min_df=1),
    )
    output_csv.write_text("previous result\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(pipeline_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        create_predictions(input_path=input_csv, artifact_dir=artifact_dir, output_path=output_csv)
    assert output_csv.read_text(encoding="utf-8") == "previous result\n"
    assert not list(tmp_path.glob(".qqdup-*.csv"))


def test_evaluation_preserves_existing_report_if_replace_fails(
    tmp_path, pair_frame: pd.DataFrame, monkeypatch
) -> None:
    training_csv = tmp_path / "train.csv"
    artifact_dir = tmp_path / "artifact"
    output_json = tmp_path / "evaluation.json"
    pair_frame.to_csv(training_csv, index=False)
    train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(tfidf_min_df=1),
    )
    output_json.write_text('{"previous": true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(pipeline_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        evaluate_labeled_file(
            data_path=training_csv, artifact_dir=artifact_dir, output_path=output_json
        )
    assert output_json.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert not list(tmp_path.glob(".qqdup-*.json"))


def test_chunked_predictions_reject_invalid_chunk_size(tmp_path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        create_predictions_chunked(
            input_path=tmp_path / "missing.csv",
            artifact_dir=tmp_path / "missing",
            output_path=tmp_path / "unused.csv",
            chunk_size=0,
        )


@pytest.mark.parametrize("identifier", ["id", "test_id"])
def test_predictions_preserve_mixed_identifier_text(
    tmp_path, pair_frame: pd.DataFrame, identifier: str
) -> None:
    training_csv = tmp_path / "train.csv"
    input_csv = tmp_path / "input.csv"
    artifact_dir = tmp_path / "artifact"
    output_csv = tmp_path / "output.csv"
    regular_csv = tmp_path / "regular.csv"
    pair_frame.to_csv(training_csv, index=False)
    input_frame = pair_frame[["question1", "question2"]].iloc[:3].copy()
    input_frame.insert(0, identifier, ["001", "002", "A03"])
    input_frame.to_csv(input_csv, index=False)
    train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(tfidf_min_df=1),
    )

    create_predictions_chunked(
        input_path=input_csv,
        artifact_dir=artifact_dir,
        output_path=output_csv,
        chunk_size=1,
    )
    create_predictions(input_path=input_csv, artifact_dir=artifact_dir, output_path=regular_csv)
    for path in (output_csv, regular_csv):
        assert pd.read_csv(path, dtype={identifier: "string"})[identifier].tolist() == [
            "001",
            "002",
            "A03",
        ]
