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

    cli.main(
        [
            "evaluate",
            "--data",
            str(training_csv),
            "--artifact",
            str(artifact_dir),
            "--output",
            str(evaluation_json),
        ]
    )
    external_output = json.loads(capsys.readouterr().out)
    assert external_output["rows"] == len(pair_frame)
    assert evaluation_json.is_file()

    cli.main(["inspect", "--artifact", str(artifact_dir)])
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["artifact_type"] == "quora_duplicate_detector"


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
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["objective"] == "CoSENTLoss"
    assert called["use_lora"] is True
