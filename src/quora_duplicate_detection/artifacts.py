"""Versioned JSON/NPZ model artifacts with integrity checks and no pickle."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quora_duplicate_detection.ensemble import ScoreFusion
from quora_duplicate_detection.features import FEATURE_NAMES, PairFeatureExtractor
from quora_duplicate_detection.lexical import LexicalModel

SCHEMA_VERSION = 1
PAYLOAD_FILES = frozenset(
    {"model.npz", "vocabulary.json", "evaluation.json", "split_assignments.csv"}
)
ARRAY_NAMES = frozenset(
    {
        "tfidf_idf",
        "lexical_scaler_mean",
        "lexical_scaler_scale",
        "lexical_coefficients",
        "lexical_intercept",
        "fusion_coefficients",
        "fusion_intercept",
    }
)
MAX_ARRAY_ARCHIVE_BYTES = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_payloads(root: Path, files: object) -> None:
    if not isinstance(files, dict) or set(files) != PAYLOAD_FILES:
        raise ValueError("artifact file hash list does not match the schema")
    for filename in sorted(PAYLOAD_FILES):
        expected_hash = files[filename]
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ValueError(f"invalid artifact digest: {filename}")
        file_path = root / filename
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"artifact payload must be a regular file: {filename}")
        try:
            actual_hash = _sha256(file_path)
        except OSError as exc:
            raise ValueError(f"unable to read artifact payload: {filename}") from exc
        if actual_hash != expected_hash:
            raise ValueError(f"artifact integrity check failed: {filename}")


def _load_numeric_arrays(path: Path) -> dict[str, np.ndarray]:
    expected_members = {f"{name}.npy" for name in ARRAY_NAMES}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                len(members) != len(expected_members)
                or {m.filename for m in members} != expected_members
            ):
                raise ValueError("artifact array set does not match the schema")
            if sum(member.file_size for member in members) > MAX_ARRAY_ARCHIVE_BYTES:
                raise ValueError("artifact numeric arrays exceed the size limit")
        with np.load(path, allow_pickle=False) as stored:
            if len(stored.files) != len(ARRAY_NAMES) or set(stored.files) != ARRAY_NAMES:
                raise ValueError("artifact array set does not match the schema")
            arrays = {}
            for name in ARRAY_NAMES:
                value = stored[name]
                if value.dtype.kind not in "fiu":
                    raise ValueError(f"artifact array is not numeric: {name}")
                arrays[name] = np.asarray(value, dtype=np.float64)
            return arrays
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid artifact numeric archive") from exc


@dataclass(frozen=True)
class LoadedArtifact:
    lexical: LexicalModel
    fusion: ScoreFusion
    threshold: float
    threshold_metric: str
    semantic_model_reference: str | None
    manifest: dict[str, Any]


def save_artifact(
    output_dir: str | Path,
    *,
    lexical: LexicalModel,
    fusion: ScoreFusion,
    threshold: float,
    threshold_metric: str,
    semantic_model_reference: str | None,
    training_metadata: dict[str, Any],
    evaluation: dict[str, Any],
    split_assignments: pd.DataFrame,
) -> Path:
    lexical.validate()
    fusion.validate()
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"artifact output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    arrays_path = output / "model.npz"
    vocabulary_path = output / "vocabulary.json"
    evaluation_path = output / "evaluation.json"
    assignments_path = output / "split_assignments.csv"

    np.savez_compressed(
        arrays_path,
        tfidf_idf=lexical.extractor.idf,
        lexical_scaler_mean=lexical.scaler_mean,
        lexical_scaler_scale=lexical.scaler_scale,
        lexical_coefficients=lexical.coefficients,
        lexical_intercept=np.asarray([lexical.intercept], dtype=np.float64),
        fusion_coefficients=fusion.coefficients,
        fusion_intercept=np.asarray([fusion.intercept], dtype=np.float64),
    )
    _write_json(vocabulary_path, lexical.extractor.vocabulary)
    _write_json(evaluation_path, evaluation)
    split_assignments.to_csv(assignments_path, index=False)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "quora_duplicate_detector",
        "serialization": "json+npz; numpy allow_pickle=False",
        "feature_names": list(FEATURE_NAMES),
        "vectorizer": {
            "ngram_range": list(lexical.extractor.ngram_range),
            "max_features": lexical.extractor.max_features,
            "min_df": lexical.extractor.effective_min_df,
        },
        "fusion": {
            "component_names": list(fusion.component_names),
            "semantic_model_reference": semantic_model_reference,
        },
        "decision": {
            "positive_class": 1,
            "comparison": "probability >= threshold",
            "threshold": threshold,
            "selection_metric": threshold_metric,
        },
        "training": training_metadata,
        "files": {
            arrays_path.name: _sha256(arrays_path),
            vocabulary_path.name: _sha256(vocabulary_path),
            evaluation_path.name: _sha256(evaluation_path),
            assignments_path.name: _sha256(assignments_path),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return output


def load_artifact(path: str | Path) -> LoadedArtifact:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("artifact manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid artifact manifest: {manifest_path}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("artifact manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")
    if manifest.get("artifact_type") != "quora_duplicate_detector":
        raise ValueError("unexpected artifact type")
    if manifest.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("artifact feature schema does not match this package")

    _verify_payloads(root, manifest.get("files"))

    try:
        vocabulary_raw = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid artifact vocabulary") from exc
    if not isinstance(vocabulary_raw, dict):
        raise ValueError("artifact vocabulary must be a JSON object")
    vocabulary = {str(term): int(index) for term, index in vocabulary_raw.items()}
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("artifact vocabulary indices must be contiguous")

    arrays = _load_numeric_arrays(root / "model.npz")

    vectorizer = manifest.get("vectorizer", {})
    ngram_range_raw = vectorizer.get("ngram_range")
    if (
        not isinstance(ngram_range_raw, list)
        or len(ngram_range_raw) != 2
        or not all(isinstance(value, int) for value in ngram_range_raw)
    ):
        raise ValueError("invalid vectorizer ngram_range")
    extractor = PairFeatureExtractor.from_state(
        vocabulary=vocabulary,
        idf=arrays["tfidf_idf"],
        ngram_range=(ngram_range_raw[0], ngram_range_raw[1]),
        max_features=int(vectorizer["max_features"]),
        min_df=int(vectorizer["min_df"]),
    )
    lexical_intercept = arrays["lexical_intercept"]
    fusion_intercept = arrays["fusion_intercept"]
    if lexical_intercept.shape != (1,) or fusion_intercept.shape != (1,):
        raise ValueError("artifact intercept arrays have invalid shapes")
    lexical = LexicalModel(
        extractor=extractor,
        scaler_mean=arrays["lexical_scaler_mean"],
        scaler_scale=arrays["lexical_scaler_scale"],
        coefficients=arrays["lexical_coefficients"],
        intercept=float(lexical_intercept[0]),
    )
    fusion_metadata = manifest.get("fusion", {})
    component_names_raw = fusion_metadata.get("component_names")
    if not isinstance(component_names_raw, list):
        raise ValueError("artifact fusion component list is invalid")
    fusion = ScoreFusion(
        component_names=tuple(str(name) for name in component_names_raw),
        coefficients=arrays["fusion_coefficients"],
        intercept=float(fusion_intercept[0]),
    )
    lexical.validate()
    fusion.validate()

    decision = manifest.get("decision", {})
    if decision.get("comparison") != "probability >= threshold":
        raise ValueError("artifact decision direction is unsupported")
    threshold = float(decision["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("artifact threshold is invalid")
    semantic_reference = fusion_metadata.get("semantic_model_reference")
    if semantic_reference is not None and not isinstance(semantic_reference, str):
        raise ValueError("semantic model reference must be a string or null")

    return LoadedArtifact(
        lexical=lexical,
        fusion=fusion,
        threshold=threshold,
        threshold_metric=str(decision["selection_metric"]),
        semantic_model_reference=semantic_reference,
        manifest=manifest,
    )
