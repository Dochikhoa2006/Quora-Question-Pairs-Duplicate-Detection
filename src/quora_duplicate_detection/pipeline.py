"""End-to-end training, prediction, and external labeled evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from quora_duplicate_detection.artifacts import (
    LoadedArtifact,
    load_artifact,
    save_artifact,
)
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.data import (
    LABEL_COLUMN,
    SPLIT_REPLAY_VERSION,
    dataset_fingerprint,
    load_pairs,
    split_labeled_pairs,
    validate_pairs,
)
from quora_duplicate_detection.ensemble import ScoreFusion, select_threshold
from quora_duplicate_detection.lexical import LexicalModel
from quora_duplicate_detection.metrics import evaluate_probabilities
from quora_duplicate_detection.output import require_distinct_paths, write_json_atomic
from quora_duplicate_detection.semantic import SemanticScorer
from quora_duplicate_detection.slices import analyze_slices


def _semantic_scores(
    frame: pd.DataFrame,
    *,
    semantic_model: str | None,
    batch_size: int,
) -> np.ndarray | None:
    if semantic_model is None:
        return None
    return SemanticScorer(semantic_model, batch_size=batch_size).score(frame)


def train_pipeline(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    semantic_model: str | None = None,
) -> dict[str, Any]:
    data = load_pairs(data_path, labeled=True)
    splits = split_labeled_pairs(
        data,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_seed=config.random_seed,
        strategy=config.split_strategy,
    )
    lexical = LexicalModel.fit(
        splits.train,
        ngram_range=(config.char_ngram_min, config.char_ngram_max),
        max_features=config.tfidf_max_features,
        min_df=config.tfidf_min_df,
        logistic_c=config.logistic_c,
        max_iter=config.max_iter,
        random_seed=config.random_seed,
    )

    validation_lexical = lexical.predict_proba(splits.validation)
    validation_semantic = _semantic_scores(
        splits.validation,
        semantic_model=semantic_model,
        batch_size=config.semantic_batch_size,
    )
    validation_labels = splits.validation[LABEL_COLUMN].to_numpy(dtype=np.int8)
    fusion = ScoreFusion.fit(
        lexical_scores=validation_lexical,
        semantic_scores=validation_semantic,
        labels=validation_labels,
        logistic_c=config.logistic_c,
        max_iter=config.max_iter,
        random_seed=config.random_seed,
    )
    validation_probabilities = fusion.predict_proba(
        lexical_scores=validation_lexical,
        semantic_scores=validation_semantic,
    )
    threshold, threshold_score = select_threshold(
        validation_labels,
        validation_probabilities,
        metric=config.threshold_metric,  # type: ignore[arg-type]
    )

    test_lexical = lexical.predict_proba(splits.test)
    test_semantic = _semantic_scores(
        splits.test,
        semantic_model=semantic_model,
        batch_size=config.semantic_batch_size,
    )
    test_probabilities = fusion.predict_proba(
        lexical_scores=test_lexical,
        semantic_scores=test_semantic,
    )
    evaluation = {
        "methodology": {
            "validation_use": "fusion fitting and threshold selection",
            "test_use": "single final evaluation after model selection",
            "competition_test_labels_used": False,
            "submission_metric_reported": False,
            "split_strategy": config.split_strategy,
        },
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "validation": {
            "selected_threshold_metric": config.threshold_metric,
            "selected_threshold_metric_value": threshold_score,
            **evaluate_probabilities(
                validation_labels,
                validation_probabilities,
                threshold=threshold,
            ),
        },
        "test": evaluate_probabilities(
            splits.test[LABEL_COLUMN].to_numpy(dtype=np.int8),
            test_probabilities,
            threshold=threshold,
        ),
    }
    evaluation["test"]["slices"] = analyze_slices(
        splits.test, test_probabilities, threshold=threshold
    )
    training_metadata = {
        "dataset_fingerprint_sha256": dataset_fingerprint(data),
        "rows": len(data),
        "config": config.to_dict(),
        "model_kind": "hybrid" if semantic_model else "lexical",
        "split_replay_version": SPLIT_REPLAY_VERSION,
    }
    save_artifact(
        output_dir,
        lexical=lexical,
        fusion=fusion,
        threshold=threshold,
        threshold_metric=config.threshold_metric,
        semantic_model_reference=semantic_model,
        training_metadata=training_metadata,
        evaluation=evaluation,
        split_assignments=splits.assignments,
    )
    return evaluation


def predict_probabilities(
    frame: pd.DataFrame,
    *,
    artifact: LoadedArtifact,
    semantic_model: str | None,
    semantic_batch_size: int,
) -> np.ndarray:
    lexical_scores = artifact.lexical.predict_proba(frame)
    requires_semantic = "semantic_cosine" in artifact.fusion.component_names
    reference = semantic_model or artifact.semantic_model_reference
    if requires_semantic and reference is None:
        raise ValueError("hybrid artifact requires --semantic-model")
    semantic_scores = _semantic_scores(
        frame,
        semantic_model=reference if requires_semantic else None,
        batch_size=semantic_batch_size,
    )
    return artifact.fusion.predict_proba(
        lexical_scores=lexical_scores,
        semantic_scores=semantic_scores,
    )


def create_predictions(
    *,
    input_path: str | Path,
    artifact_dir: str | Path,
    output_path: str | Path,
    semantic_model: str | None = None,
    semantic_batch_size: int = 64,
    include_binary_prediction: bool = False,
) -> pd.DataFrame:
    require_distinct_paths(input_path, output_path)
    frame = load_pairs(input_path, labeled=False)
    artifact = load_artifact(artifact_dir)
    probabilities = predict_probabilities(
        frame,
        artifact=artifact,
        semantic_model=semantic_model,
        semantic_batch_size=semantic_batch_size,
    )
    output = _prediction_frame(
        frame,
        probabilities,
        artifact=artifact,
        row_offset=0,
        include_binary_prediction=include_binary_prediction,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=".qqdup-",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            output.to_csv(temporary, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output


def _prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    artifact: LoadedArtifact,
    row_offset: int,
    include_binary_prediction: bool,
) -> pd.DataFrame:
    identifier = next((name for name in ("test_id", "id") if name in frame), None)
    if identifier:
        output = frame[[identifier]].copy()
    else:
        output = pd.DataFrame(
            {"test_id": np.arange(row_offset, row_offset + len(frame), dtype=np.int64)}
        )
    output["is_duplicate"] = probabilities
    if include_binary_prediction:
        output["prediction"] = (probabilities >= artifact.threshold).astype(np.int8)

    return output


def create_predictions_chunked(
    *,
    input_path: str | Path,
    artifact_dir: str | Path,
    output_path: str | Path,
    chunk_size: int,
    semantic_model: str | None = None,
    semantic_batch_size: int = 64,
    include_binary_prediction: bool = False,
) -> int:
    """Stream an unlabeled CSV to an atomic output while loading models once."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    require_distinct_paths(input_path, output_path)
    artifact = load_artifact(artifact_dir)
    requires_semantic = "semantic_cosine" in artifact.fusion.component_names
    reference = semantic_model or artifact.semantic_model_reference
    if requires_semantic and reference is None:
        raise ValueError("hybrid artifact requires --semantic-model")
    scorer = (
        SemanticScorer(reference, batch_size=semantic_batch_size) if requires_semantic else None
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    rows = 0
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=".qqdup-",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for raw_chunk in pd.read_csv(
                input_path,
                chunksize=chunk_size,
                dtype={
                    "question1": "string",
                    "question2": "string",
                    "test_id": "string",
                    "id": "string",
                },
            ):
                chunk = validate_pairs(raw_chunk, labeled=False)
                lexical_scores = artifact.lexical.predict_proba(chunk)
                semantic_scores = scorer.score(chunk) if scorer is not None else None
                probabilities = artifact.fusion.predict_proba(
                    lexical_scores=lexical_scores,
                    semantic_scores=semantic_scores,
                )
                output = _prediction_frame(
                    chunk,
                    probabilities,
                    artifact=artifact,
                    row_offset=rows,
                    include_binary_prediction=include_binary_prediction,
                )
                output.to_csv(temporary, index=False, header=rows == 0)
                rows += len(chunk)
        if rows == 0:
            raise ValueError(f"{input_path} contains no rows")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return rows


def evaluate_labeled_file(
    *,
    data_path: str | Path,
    artifact_dir: str | Path,
    output_path: str | Path | None = None,
    semantic_model: str | None = None,
    semantic_batch_size: int = 64,
    slice_min_rows: int = 20,
) -> dict[str, Any]:
    if output_path is not None:
        require_distinct_paths(data_path, output_path)
    frame = load_pairs(data_path, labeled=True)
    artifact = load_artifact(artifact_dir)
    probabilities = predict_probabilities(
        frame,
        artifact=artifact,
        semantic_model=semantic_model,
        semantic_batch_size=semantic_batch_size,
    )
    report = evaluate_probabilities(
        frame[LABEL_COLUMN].to_numpy(dtype=np.int8),
        probabilities,
        threshold=artifact.threshold,
    )
    report["dataset_fingerprint_sha256"] = dataset_fingerprint(frame)
    report["slices"] = analyze_slices(
        frame, probabilities, threshold=artifact.threshold, min_rows=slice_min_rows
    )
    report["note"] = "Metrics are valid only because this file contains observed labels."
    if output_path is not None:
        write_json_atomic(output_path, report)
    return report
