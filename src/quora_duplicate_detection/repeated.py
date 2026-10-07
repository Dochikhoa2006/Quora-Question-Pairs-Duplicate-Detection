"""Repeated question-disjoint evaluation with compact uncertainty summaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

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


def evaluate_repeated(
    *,
    data_path: str | Path,
    output_path: str | Path,
    config: TrainingConfig,
    repeats: int = 5,
    semantic_model: str | None = None,
) -> dict[str, Any]:
    """Fit independent grouped runs and report the empirical spread of test metrics."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 3 <= repeats <= 100:
        raise ValueError("repeats must be an integer between 3 and 100")
    data = load_pairs(data_path, labeled=True)
    grouped_config = replace(config, split_strategy="question_disjoint")
    runs: list[dict[str, Any]] = []
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
            test = evaluation["test"]
            runs.append(
                {
                    "seed": seed,
                    "split_sizes": evaluation["split_sizes"],
                    "split_assignments_sha256": manifest["files"]["split_assignments.csv"],
                    "selected_threshold": test["threshold"],
                    "test_metrics": {
                        **{name: test[name] for name in METRICS},
                        "expected_calibration_error": test["calibration"][
                            "expected_calibration_error"
                        ],
                    },
                }
            )

    metric_names = (*METRICS, "expected_calibration_error")
    summary = {}
    for name in metric_names:
        values = np.asarray([run["test_metrics"][name] for run in runs], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "p2_5": float(np.quantile(values, 0.025)),
            "p97_5": float(np.quantile(values, 0.975)),
        }
    report: dict[str, Any] = {
        "protocol": "repeated_question_disjoint_holdout",
        "uncertainty_interpretation": (
            "Empirical variation across overlapping seeded holdouts; percentile bounds "
            "are descriptive, not a confidence interval."
        ),
        "dataset_fingerprint_sha256": dataset_fingerprint(data),
        "rows": len(data),
        "config": grouped_config.to_dict(),
        "repeats": repeats,
        "runs": runs,
        "summary": summary,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
