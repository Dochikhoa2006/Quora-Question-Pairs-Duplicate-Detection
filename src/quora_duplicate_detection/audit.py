"""Audit a saved split against its original labeled dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from quora_duplicate_detection.artifacts import load_artifact
from quora_duplicate_detection.data import (
    LABEL_COLUMN,
    PAIR_COLUMNS,
    dataset_fingerprint,
    load_pairs,
)

SPLIT_NAMES = ("train", "validation", "test")


def _read_assignments(path: Path, rows: int) -> pd.DataFrame:
    assignments = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(assignments.columns) != ["row_number", "split"]:
        raise ValueError("split assignments must have row_number and split columns")
    if len(assignments) != rows:
        raise ValueError("split assignments do not cover every dataset row")
    if not assignments["row_number"].str.fullmatch(r"0|[1-9][0-9]*").all():
        raise ValueError("split assignments contain invalid row numbers")
    numbers = assignments["row_number"].astype("int64")
    if sorted(numbers.tolist()) != list(range(rows)):
        raise ValueError("split assignments must contain every row exactly once")
    if not assignments["split"].isin(SPLIT_NAMES).all():
        raise ValueError("split assignments contain an unknown partition")
    assignments = assignments.assign(row_number=numbers)
    return assignments.sort_values("row_number").reset_index(drop=True)


def _overlapping_questions(frame: pd.DataFrame, splits: pd.Series) -> int:
    owners: dict[str, str] = {}
    overlapping: set[str] = set()
    for pair, split in zip(
        frame[list(PAIR_COLUMNS)].itertuples(index=False, name=None), splits, strict=True
    ):
        for question in pair:
            prior = owners.setdefault(question, split)
            if prior != split:
                overlapping.add(question)
    return len(overlapping)


def audit_split(
    *,
    data_path: str | Path,
    artifact_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check saved assignments and report only aggregate evidence."""
    artifact = load_artifact(artifact_dir)
    frame = load_pairs(data_path, labeled=True)
    fingerprint = dataset_fingerprint(frame)
    training = artifact.manifest.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("artifact training metadata must be an object")
    if training.get("dataset_fingerprint_sha256") != fingerprint:
        raise ValueError("dataset fingerprint does not match the artifact")

    root = Path(artifact_dir)
    assignments = _read_assignments(root / "split_assignments.csv", len(frame))
    splits = assignments["split"]
    counts = {name: int((splits == name).sum()) for name in SPLIT_NAMES}
    positives = {name: int(frame.loc[splits == name, LABEL_COLUMN].sum()) for name in SPLIT_NAMES}
    overlap = _overlapping_questions(frame, splits)
    config = training.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("artifact training configuration must be an object")
    strategy = config.get("split_strategy", "pair_stratified")
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict):
        raise ValueError("artifact evaluation must be an object")
    methodology = evaluation.get("methodology", {})
    if not isinstance(methodology, dict):
        raise ValueError("artifact evaluation methodology must be an object")
    checks = {
        "row_count_matches_manifest": training.get("rows") == len(frame),
        "split_sizes_match_evaluation": evaluation.get("split_sizes") == counts,
        "strategy_matches_evaluation": methodology.get("split_strategy", "pair_stratified")
        == strategy,
        "all_partitions_have_both_classes": all(
            0 < positives[name] < counts[name] for name in SPLIT_NAMES
        ),
        "question_disjoint_when_required": strategy != "question_disjoint" or overlap == 0,
    }
    report: dict[str, Any] = {
        "passed": all(checks.values()),
        "checks": checks,
        "dataset_fingerprint_sha256": fingerprint,
        "split_strategy": strategy,
        "rows": len(frame),
        "split_sizes": counts,
        "positive_rows": positives,
        "questions_crossing_splits": overlap,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report
