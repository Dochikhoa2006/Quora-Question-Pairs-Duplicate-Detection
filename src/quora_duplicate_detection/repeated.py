"""Repeated question-disjoint evaluation with compact uncertainty summaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.data import dataset_fingerprint, load_pairs
from quora_duplicate_detection.pipeline import train_pipeline

METRICS = (
    "log_loss",
    "brier_score",
    "roc_auc",
    "average_precision",
    "f1",
    "accuracy",
)
METRIC_NAMES = (*METRICS, "expected_calibration_error")


def _test_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    test = evaluation["test"]
    return {
        **{name: test[name] for name in METRICS},
        "expected_calibration_error": test["calibration"]["expected_calibration_error"],
    }


def _metric_summary(runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    summary = {}
    for name in METRIC_NAMES:
        values = np.asarray([run[key][name] for run in runs], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "p2_5": float(np.quantile(values, 0.025)),
            "p97_5": float(np.quantile(values, 0.975)),
        }
    return summary


def _test_overlap_summary(test_masks: list[np.ndarray]) -> dict[str, Any]:
    exposures = np.sum(np.stack(test_masks), axis=0)
    overlap = []
    for left_index, left in enumerate(test_masks):
        for right in test_masks[left_index + 1 :]:
            intersection = np.count_nonzero(left & right)
            union = np.count_nonzero(left | right)
            overlap.append(intersection / union)
    pairwise = np.asarray(overlap, dtype=np.float64)
    return {
        "rows_tested_at_least_once": int(np.count_nonzero(exposures)),
        "fraction_tested_at_least_once": float(np.count_nonzero(exposures) / len(exposures)),
        "rows_never_tested": int(np.count_nonzero(exposures == 0)),
        "test_appearances_per_row": {
            "minimum": int(exposures.min()),
            "maximum": int(exposures.max()),
            "mean": float(exposures.mean()),
        },
        "pairwise_test_jaccard": {
            "minimum": float(pairwise.min()),
            "maximum": float(pairwise.max()),
            "mean": float(pairwise.mean()),
        },
    }


def evaluate_repeated(
    *,
    data_path: str | Path,
    output_path: str | Path,
    config: TrainingConfig,
    repeats: int = 5,
    semantic_model: str | None = None,
    compare_lexical: bool = False,
) -> dict[str, Any]:
    """Fit independent grouped runs and report the empirical spread of test metrics."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 3 <= repeats <= 100:
        raise ValueError("repeats must be an integer between 3 and 100")
    if compare_lexical and semantic_model is None:
        raise ValueError("compare_lexical requires a semantic_model")
    data = load_pairs(data_path, labeled=True)
    grouped_config = replace(config, split_strategy="question_disjoint")
    runs: list[dict[str, Any]] = []
    test_masks: list[np.ndarray] = []
    for offset in range(repeats):
        seed = grouped_config.random_seed + offset
        run_config = replace(grouped_config, random_seed=seed)
        with TemporaryDirectory(prefix="qqdup-repeat-") as temporary:
            artifact_dir = Path(temporary) / "artifact"
            evaluation = train_pipeline(
                data_path=data_path,
                output_dir=artifact_dir,
                config=run_config,
                semantic_model=semantic_model,
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
            assignments = pd.read_csv(artifact_dir / "split_assignments.csv")
            test_masks.append(assignments["split"].eq("test").to_numpy(dtype=bool))
            test_metrics = _test_metrics(evaluation)
            run = {
                "seed": seed,
                "split_sizes": evaluation["split_sizes"],
                "split_assignments_sha256": manifest["files"]["split_assignments.csv"],
                "selected_threshold": evaluation["test"]["threshold"],
                "test_metrics": test_metrics,
            }
            if compare_lexical:
                lexical_dir = Path(temporary) / "lexical_artifact"
                lexical_evaluation = train_pipeline(
                    data_path=data_path,
                    output_dir=lexical_dir,
                    config=run_config,
                )
                lexical_manifest = json.loads(
                    (lexical_dir / "manifest.json").read_text(encoding="utf-8")
                )
                if (
                    lexical_manifest["files"]["split_assignments.csv"]
                    != run["split_assignments_sha256"]
                    or lexical_evaluation["split_sizes"] != run["split_sizes"]
                ):
                    raise RuntimeError("paired models did not use identical split assignments")
                lexical_metrics = _test_metrics(lexical_evaluation)
                run["lexical_test_metrics"] = lexical_metrics
                run["hybrid_minus_lexical"] = {
                    name: test_metrics[name] - lexical_metrics[name] for name in METRIC_NAMES
                }
            runs.append(run)

    report: dict[str, Any] = {
        "protocol": "repeated_question_disjoint_holdout",
        "uncertainty_interpretation": (
            "Empirical variation across overlapping seeded holdouts; percentile bounds "
            "are descriptive, not a confidence interval. Test-set overlap is summarized "
            "separately because runs do not have independent test rows."
        ),
        "dataset_fingerprint_sha256": dataset_fingerprint(data),
        "rows": len(data),
        "config": grouped_config.to_dict(),
        "repeats": repeats,
        "runs": runs,
        "test_overlap": _test_overlap_summary(test_masks),
        "summary": _metric_summary(runs, "test_metrics"),
    }
    if compare_lexical:
        report["paired_comparison"] = {
            "baseline": "lexical",
            "candidate": "hybrid",
            "difference": "hybrid_minus_lexical",
            "same_split_per_seed": True,
            "interpretation": (
                "Descriptive paired differences on the same test rows for each seed; "
                "repeated holdouts overlap and are not independent confidence intervals."
            ),
            "summary": _metric_summary(runs, "hybrid_minus_lexical"),
        }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
