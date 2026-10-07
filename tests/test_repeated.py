from __future__ import annotations

import json

import pandas as pd
import pytest

from quora_duplicate_detection import cli
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.repeated import evaluate_repeated


def test_repeated_grouped_report_is_reproducible_and_aggregate(
    tmp_path, pair_frame: pd.DataFrame
) -> None:
    input_path = tmp_path / "pairs.csv"
    pair_frame.to_csv(input_path, index=False)
    config = TrainingConfig(tfidf_min_df=1, tfidf_max_features=2_000)
    first = evaluate_repeated(
        data_path=input_path,
        output_path=tmp_path / "first.json",
        config=config,
        repeats=3,
    )
    second = evaluate_repeated(
        data_path=input_path,
        output_path=tmp_path / "second.json",
        config=config,
        repeats=3,
    )

    assert first == second
    assert json.loads((tmp_path / "first.json").read_text(encoding="utf-8")) == first
    assert first["config"]["split_strategy"] == "question_disjoint"
    assert [run["seed"] for run in first["runs"]] == [42, 43, 44]
    assert all(sum(run["split_sizes"].values()) == len(pair_frame) for run in first["runs"])
    assert len({run["split_assignments_sha256"] for run in first["runs"]}) == 3
    assert first["summary"]["log_loss"]["sample_std"] >= 0
    assert "question1" not in json.dumps(first)
    assert config.split_strategy == "pair_stratified"


def test_repeated_cli_writes_report(tmp_path, pair_frame: pd.DataFrame, capsys) -> None:
    input_path = tmp_path / "pairs.csv"
    output_path = tmp_path / "report.json"
    pair_frame.to_csv(input_path, index=False)
    cli.main(
        [
            "evaluate-repeated",
            "--data",
            str(input_path),
            "--output",
            str(output_path),
            "--repeats",
            "3",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output_path.read_text(encoding="utf-8"))
    assert printed["protocol"] == "repeated_question_disjoint_holdout"


def test_repeated_rejects_insufficient_runs_without_writing(tmp_path) -> None:
    output_path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="between 3 and 100"):
        evaluate_repeated(
            data_path=tmp_path / "missing.csv",
            output_path=output_path,
            config=TrainingConfig(),
            repeats=2,
        )
    assert not output_path.exists()


def test_repeated_split_failure_does_not_write_partial_report(tmp_path) -> None:
    input_path = tmp_path / "connected.csv"
    output_path = tmp_path / "report.json"
    pd.DataFrame(
        [
            ("a", "b", 1),
            ("b", "c", 0),
            ("c", "d", 1),
        ],
        columns=["question1", "question2", "is_duplicate"],
    ).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="at least three components"):
        evaluate_repeated(
            data_path=input_path,
            output_path=output_path,
            config=TrainingConfig(),
            repeats=3,
        )
    assert not output_path.exists()
