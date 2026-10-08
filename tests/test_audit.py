from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from quora_duplicate_detection import cli
from quora_duplicate_detection.audit import audit_split
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.pipeline import train_pipeline


@pytest.fixture()
def grouped_artifact(tmp_path):
    rows = []
    for component in range(30):
        rows.extend(
            [
                (f"private {component} a", f"private {component} b", 1),
                (f"private {component} b", f"private {component} c", 0),
            ]
        )
    data_path = tmp_path / "pairs.csv"
    artifact_dir = tmp_path / "artifact"
    pd.DataFrame(rows, columns=["question1", "question2", "is_duplicate"]).to_csv(
        data_path, index=False
    )
    train_pipeline(
        data_path=data_path,
        output_dir=artifact_dir,
        config=TrainingConfig(
            split_strategy="question_disjoint",
            validation_size=0.2,
            test_size=0.2,
            tfidf_min_df=1,
        ),
    )
    return data_path, artifact_dir


def test_audit_split_reports_aggregate_grouped_evidence(grouped_artifact, tmp_path) -> None:
    data_path, artifact_dir = grouped_artifact
    report = audit_split(
        data_path=data_path,
        artifact_dir=artifact_dir,
        output_path=tmp_path / "audit.json",
    )
    assert report["passed"] is True
    assert report["questions_crossing_splits"] == 0
    assert all(report["checks"].values())
    assert sum(report["split_sizes"].values()) == report["rows"]
    assert "private" not in json.dumps(report)
    assert json.loads((tmp_path / "audit.json").read_text(encoding="utf-8")) == report


def test_audit_detects_rehashed_leaky_assignments(grouped_artifact) -> None:
    data_path, artifact_dir = grouped_artifact
    assignment_path = artifact_dir / "split_assignments.csv"
    assignments = pd.read_csv(assignment_path)
    assignments.loc[1, "split"] = "test" if assignments.loc[0, "split"] != "test" else "train"
    assignments.to_csv(assignment_path, index=False)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["split_assignments.csv"] = hashlib.sha256(
        assignment_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_split(data_path=data_path, artifact_dir=artifact_dir)
    assert report["passed"] is False
    assert report["questions_crossing_splits"] > 0
    assert report["checks"]["question_disjoint_when_required"] is False


def test_audit_rejects_wrong_dataset(grouped_artifact, tmp_path) -> None:
    data_path, artifact_dir = grouped_artifact
    wrong = tmp_path / "wrong.csv"
    frame = pd.read_csv(data_path)
    frame.loc[0, "question1"] = "changed"
    frame.to_csv(wrong, index=False)
    with pytest.raises(ValueError, match="fingerprint"):
        audit_split(data_path=wrong, artifact_dir=artifact_dir)


def test_audit_cli_exits_nonzero_on_failed_check(grouped_artifact, capsys, monkeypatch) -> None:
    data_path, artifact_dir = grouped_artifact
    cli.main(["audit-split", "--data", str(data_path), "--artifact", str(artifact_dir)])
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True

    monkeypatch.setattr(cli, "audit_split", lambda **kwargs: {"passed": False})
    with pytest.raises(SystemExit) as exc:
        cli.main(["audit-split", "--data", str(data_path), "--artifact", str(artifact_dir)])
    assert exc.value.code == 1
