"""Paired external evaluation of saved lexical and semantic detector artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from quora_duplicate_detection.artifacts import load_artifact
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.data import (
    LABEL_COLUMN,
    PAIR_COLUMNS,
    dataset_fingerprint,
    load_pairs,
)
from quora_duplicate_detection.metrics import evaluate_probabilities
from quora_duplicate_detection.pipeline import predict_probabilities
from quora_duplicate_detection.repeated import METRIC_NAMES


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    report = evaluate_probabilities(labels, probabilities, threshold=threshold)
    return {
        **{
            name: float(report[name])
            for name in METRIC_NAMES
            if name != "expected_calibration_error"
        },
        "expected_calibration_error": report["calibration"]["expected_calibration_error"],
    }


def _overlap_counts(external: Any, training: Any) -> dict[str, int]:
    """Count rows sharing an unordered pair or any normalized question."""
    training_pairs = {
        tuple(sorted(pair))
        for pair in training[list(PAIR_COLUMNS)].itertuples(index=False, name=None)
    }
    training_questions = set(training[PAIR_COLUMNS[0]]) | set(training[PAIR_COLUMNS[1]])
    pairs = external[list(PAIR_COLUMNS)].itertuples(index=False, name=None)
    pair_rows = 0
    question_rows = 0
    for first, second in pairs:
        pair_rows += tuple(sorted((first, second))) in training_pairs
        question_rows += first in training_questions or second in training_questions
    return {"shared_pair_rows": pair_rows, "shared_question_rows": question_rows}


def compare_artifacts(
    *,
    data_path: str | Path,
    output_path: str | Path,
    models: dict[str, str | Path],
    baseline: str,
    semantic_models: dict[str, str] | None = None,
    training_data: dict[str, str | Path] | None = None,
    bootstrap_samples: int = 200,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compare frozen artifacts on one external set with stratified paired resampling."""
    if len(models) < 2 or baseline not in models:
        raise ValueError("provide at least two models and a baseline label among them")
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", label) is None for label in models):
        raise ValueError("model labels must contain only letters, digits, underscores, or hyphens")
    overrides = semantic_models or {}
    if set(overrides) - set(models):
        raise ValueError("semantic model override refers to an unknown model label")
    source_data = training_data or {}
    if set(source_data) - set(models):
        raise ValueError("training data refers to an unknown model label")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or not 100 <= bootstrap_samples <= 2000
    ):
        raise ValueError("bootstrap_samples must be an integer between 100 and 2000")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")

    frame = load_pairs(data_path, labeled=True)
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
    if len(np.unique(labels)) != 2:
        raise ValueError("external evaluation requires both label classes")
    fingerprint = dataset_fingerprint(frame)
    probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    model_reports: dict[str, Any] = {}
    for label, directory in models.items():
        artifact = load_artifact(directory)
        training = artifact.manifest.get("training", {})
        if training.get("dataset_fingerprint_sha256") == fingerprint:
            raise ValueError(f"evaluation data matches training data for model: {label}")
        overlap = None
        if label in source_data:
            original = load_pairs(source_data[label], labeled=True)
            if dataset_fingerprint(original) != training["dataset_fingerprint_sha256"]:
                raise ValueError(f"training data fingerprint does not match model: {label}")
            overlap = _overlap_counts(frame, original)
            if overlap["shared_question_rows"]:
                raise ValueError(
                    f"external data shares {overlap['shared_pair_rows']} pair rows and "
                    f"{overlap['shared_question_rows']} question rows with model: {label}"
                )
        config = TrainingConfig(**training["config"])
        scores = predict_probabilities(
            frame,
            artifact=artifact,
            semantic_model=overrides.get(label),
            semantic_batch_size=config.semantic_batch_size,
        )
        probabilities[label] = scores
        thresholds[label] = artifact.threshold
        model_reports[label] = {
            "artifact_manifest_sha256": hashlib.sha256(
                (Path(directory) / "manifest.json").read_bytes()
            ).hexdigest(),
            "training_dataset_fingerprint_sha256": training["dataset_fingerprint_sha256"],
            "components": list(artifact.fusion.component_names),
            "threshold": artifact.threshold,
            "metrics": _metrics(labels, scores, artifact.threshold),
            "training_overlap_check": overlap,
        }

    rng = np.random.default_rng(random_seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    differences: dict[str, dict[str, list[float]]] = {
        label: {name: [] for name in METRIC_NAMES} for label in models if label != baseline
    }
    for _ in range(bootstrap_samples):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        sampled_labels = labels[indices]
        base_metrics = _metrics(
            sampled_labels, probabilities[baseline][indices], thresholds[baseline]
        )
        for label, metrics in differences.items():
            candidate = _metrics(sampled_labels, probabilities[label][indices], thresholds[label])
            for name in METRIC_NAMES:
                metrics[name].append(candidate[name] - base_metrics[name])

    comparisons = {}
    for label, metrics in differences.items():
        baseline_metrics = model_reports[baseline]["metrics"]
        candidate_metrics = model_reports[label]["metrics"]
        comparisons[label] = {
            "candidate_minus_baseline": {
                name: candidate_metrics[name] - baseline_metrics[name] for name in METRIC_NAMES
            },
            "paired_bootstrap_percentiles": {
                name: {
                    "p2_5": float(np.quantile(values, 0.025)),
                    "p97_5": float(np.quantile(values, 0.975)),
                }
                for name, values in metrics.items()
            },
        }

    report = {
        "protocol": "external_labeled_paired_comparison",
        "interpretation": (
            "Stratified paired bootstrap percentiles describe sampling variation on this "
            "external set, conditional on fixed trained artifacts and thresholds. "
            "They do not account for training variation or undisclosed data overlap."
        ),
        "dataset_fingerprint_sha256": fingerprint,
        "rows": len(frame),
        "positive_rows": int(labels.sum()),
        "baseline": baseline,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "models": model_reports,
        "comparisons": comparisons,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
