from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quora_duplicate_detection import cli, pipeline, repeated
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.repeated import _test_overlap_summary, evaluate_repeated


def test_test_overlap_summary_uses_set_jaccard_and_row_exposure() -> None:
    masks = [
        pd.Series([True, True, False, False]).to_numpy(),
        pd.Series([False, True, True, False]).to_numpy(),
        pd.Series([False, False, True, False]).to_numpy(),
    ]
    summary = _test_overlap_summary(masks)
    assert summary["rows_tested_at_least_once"] == 3
    assert summary["fraction_tested_at_least_once"] == 0.75
    assert summary["rows_never_tested"] == 1
    assert summary["test_appearances_per_row"] == {
        "minimum": 0,
        "maximum": 2,
        "mean": 1.25,
    }
    assert summary["pairwise_test_jaccard"]["minimum"] == 0
    assert summary["pairwise_test_jaccard"]["maximum"] == 0.5
    assert summary["pairwise_test_jaccard"]["mean"] == pytest.approx(5 / 18)


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
    assert first["test_overlap"]["rows_tested_at_least_once"] <= len(pair_frame)
    assert first["test_overlap"]["rows_never_tested"] >= 0
    assert 0 <= first["test_overlap"]["pairwise_test_jaccard"]["mean"] <= 1
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


def test_repeated_rejects_seed_overflow_before_loading_data(tmp_path) -> None:
    output_path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="supported seed range"):
        evaluate_repeated(
            data_path=tmp_path / "missing.csv",
            output_path=output_path,
            config=TrainingConfig(random_seed=2**32 - 2),
            repeats=3,
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


def test_paired_hybrid_comparison_uses_same_seeded_splits(
    tmp_path, pair_frame: pd.DataFrame, monkeypatch
) -> None:
    input_path = tmp_path / "pairs.csv"
    output_path = tmp_path / "paired.json"
    pair_frame.to_csv(input_path, index=False)

    def fake_semantic_scores(frame, *, semantic_model, batch_size):
        if semantic_model is None:
            return None
        return np.asarray(frame["question1"].str.len(), dtype=np.float64) / 100

    monkeypatch.setattr(pipeline, "_semantic_scores", fake_semantic_scores)
    report = evaluate_repeated(
        data_path=input_path,
        output_path=output_path,
        config=TrainingConfig(tfidf_min_df=1, tfidf_max_features=2_000),
        repeats=3,
        semantic_model="stub-model",
        compare_lexical=True,
    )

    assert report["paired_comparison"]["same_split_per_seed"] is True
    assert report["paired_comparison"]["difference"] == "hybrid_minus_lexical"
    assert len(report["runs"]) == 3
    for run in report["runs"]:
        assert run["hybrid_minus_lexical"]["log_loss"] == pytest.approx(
            run["test_metrics"]["log_loss"] - run["lexical_test_metrics"]["log_loss"]
        )
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert "question1" not in json.dumps(report)


def test_paired_comparison_requires_semantic_model_before_loading_data(tmp_path) -> None:
    output_path = tmp_path / "unused.json"
    with pytest.raises(ValueError, match="requires a semantic_model"):
        evaluate_repeated(
            data_path=tmp_path / "missing.csv",
            output_path=output_path,
            config=TrainingConfig(),
            compare_lexical=True,
        )
    assert not output_path.exists()


def test_paired_comparison_rejects_mismatched_split_evidence(
    tmp_path, pair_frame: pd.DataFrame, monkeypatch
) -> None:
    input_path = tmp_path / "pairs.csv"
    output_path = tmp_path / "paired.json"
    pair_frame.to_csv(input_path, index=False)
    original_train = repeated.train_pipeline

    def altered_lexical_manifest(*, output_dir, semantic_model=None, **kwargs):
        report = original_train(output_dir=output_dir, semantic_model=None, **kwargs)
        if semantic_model is None:
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["split_assignments.csv"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return report

    monkeypatch.setattr(repeated, "train_pipeline", altered_lexical_manifest)
    with pytest.raises(RuntimeError, match="identical split assignments"):
        evaluate_repeated(
            data_path=input_path,
            output_path=output_path,
            config=TrainingConfig(tfidf_min_df=1),
            repeats=3,
            semantic_model="stub-model",
            compare_lexical=True,
        )
    assert not output_path.exists()
