from __future__ import annotations

import json

import pandas as pd

from quora_duplicate_detection import cli


def test_cli_train_predict_evaluate_and_inspect(
    tmp_path,
    pair_frame: pd.DataFrame,
    capsys,
) -> None:
    training_csv = tmp_path / "train.csv"
    artifact_dir = tmp_path / "artifact"
    prediction_csv = tmp_path / "predictions.csv"
    evaluation_json = tmp_path / "external.json"
    pair_frame.to_csv(training_csv, index=False)

    cli.main(["train", "--data", str(training_csv), "--output", str(artifact_dir)])
    train_output = json.loads(capsys.readouterr().out)
    assert train_output["methodology"]["competition_test_labels_used"] is False

    cli.main(
        [
            "predict",
            "--input",
            str(training_csv),
            "--artifact",
            str(artifact_dir),
            "--output",
            str(prediction_csv),
            "--include-binary-prediction",
        ]
    )
    predict_output = json.loads(capsys.readouterr().out)
    assert predict_output["rows"] == len(pair_frame)
    assert "prediction" in pd.read_csv(prediction_csv)

    chunked_csv = tmp_path / "chunked.csv"
    cli.main(
        [
            "predict",
            "--input",
            str(training_csv),
            "--artifact",
            str(artifact_dir),
            "--output",
            str(chunked_csv),
            "--include-binary-prediction",
            "--chunk-size",
            "7",
        ]
    )
    assert json.loads(capsys.readouterr().out)["rows"] == len(pair_frame)
    pd.testing.assert_frame_equal(pd.read_csv(prediction_csv), pd.read_csv(chunked_csv))

    cli.main(
        [
            "evaluate",
            "--data",
            str(training_csv),
            "--artifact",
            str(artifact_dir),
            "--output",
            str(evaluation_json),
            "--slice-min-rows",
            "20",
        ]
    )
    external_output = json.loads(capsys.readouterr().out)
    assert external_output["rows"] == len(pair_frame)
    assert external_output["calibration"]["bin_count"] == 10
    assert external_output["slices"]["minimum_rows"] == 20
    assert "no" in external_output["slices"]["dimensions"]["empty_question"]
    assert evaluation_json.is_file()

    cli.main(["inspect", "--artifact", str(artifact_dir)])
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["artifact_type"] == "quora_duplicate_detector"

    cli.main(["inspect", "--artifact", str(artifact_dir), "--summary"])
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "artifact_type": "quora_duplicate_detector",
        "schema_version": manifest["schema_version"],
        "model_kind": "lexical",
        "training_rows": len(pair_frame),
        "split_strategy": manifest["training"]["config"]["split_strategy"],
        "components": ["lexical_probability"],
        "threshold": manifest["decision"]["threshold"],
        "threshold_metric": manifest["decision"]["selection_metric"],
        "dataset_fingerprint_sha256": manifest["training"]["dataset_fingerprint_sha256"],
    }


def test_cli_semantic_command_delegates_without_running_a_model(
    tmp_path,
    pair_frame: pd.DataFrame,
    monkeypatch,
    capsys,
) -> None:
    training_csv = tmp_path / "train.csv"
    pair_frame.to_csv(training_csv, index=False)
    called = {}

    def fake_train(frame, **kwargs):
        called.update(kwargs)
        return {"objective": "CoSENTLoss", "rows": len(frame)}

    monkeypatch.setattr(cli, "train_semantic_model", fake_train)
    cli.main(
        [
            "train-semantic",
            "--data",
            str(training_csv),
            "--output",
            str(tmp_path / "semantic"),
            "--lora",
            "--base-revision",
            "c" * 40,
            "--resume",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["objective"] == "CoSENTLoss"
    assert called["use_lora"] is True
    assert called["base_revision"] == "c" * 40
    assert called["resume"] is True
