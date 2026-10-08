"""Audit a saved split against its original labeled dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from quora_duplicate_detection.artifacts import load_artifact
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.data import (
    LABEL_COLUMN,
    PAIR_COLUMNS,
    SPLIT_REPLAY_VERSION,
    dataset_fingerprint,
    load_pairs,
    split_labeled_pairs,
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


def _replay_split(
    frame: pd.DataFrame,
    splits: pd.Series,
    training: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    version = training.get("split_replay_version")
    if version is None:
        return {"status": "unavailable_legacy", "version": None, "mismatched_rows": None}
    if version != SPLIT_REPLAY_VERSION:
        return {"status": "unsupported_version", "version": version, "mismatched_rows": None}
    try:
        replay_config = TrainingConfig(**config)
        replayed = split_labeled_pairs(
            frame,
            validation_size=replay_config.validation_size,
            test_size=replay_config.test_size,
            random_seed=replay_config.random_seed,
            strategy=replay_config.split_strategy,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact split replay configuration is invalid") from exc
    mismatched = int((replayed.assignments["split"] != splits).sum())
    return {
        "status": "matched" if mismatched == 0 else "mismatch",
        "version": version,
        "mismatched_rows": mismatched,
    }


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
    replay = _replay_split(frame, splits, training, config)
    checks = {
        "row_count_matches_manifest": training.get("rows") == len(frame),
        "split_sizes_match_evaluation": evaluation.get("split_sizes") == counts,
        "strategy_matches_evaluation": methodology.get("split_strategy", "pair_stratified")
        == strategy,
        "all_partitions_have_both_classes": all(
            0 < positives[name] < counts[name] for name in SPLIT_NAMES
        ),
        "question_disjoint_when_required": strategy != "question_disjoint" or overlap == 0,
        "assignments_match_replay": (
            None if replay["status"] == "unavailable_legacy" else replay["status"] == "matched"
        ),
    }
    report: dict[str, Any] = {
        "passed": all(value for value in checks.values() if value is not None),
        "checks": checks,
        "replay": replay,
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
