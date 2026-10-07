from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from quora_duplicate_detection.artifacts import load_artifact, save_artifact
from quora_duplicate_detection.ensemble import ScoreFusion
from quora_duplicate_detection.lexical import LexicalModel


def _trained_models(pair_frame: pd.DataFrame) -> tuple[LexicalModel, ScoreFusion]:
    train = pair_frame.iloc[:60]
    validation = pair_frame.iloc[60:]
    lexical = LexicalModel.fit(
        train,
        ngram_range=(3, 5),
        max_features=2_000,
        min_df=1,
        logistic_c=1.0,
        max_iter=500,
        random_seed=42,
    )
    scores = lexical.predict_proba(validation)
    fusion = ScoreFusion.fit(
        lexical_scores=scores,
        semantic_scores=None,
        labels=validation["is_duplicate"].to_numpy(),
        logistic_c=1.0,
        max_iter=500,
        random_seed=42,
    )
    return lexical, fusion


def test_json_npz_artifact_round_trip(tmp_path, pair_frame: pd.DataFrame) -> None:
    lexical, fusion = _trained_models(pair_frame)
    artifact_path = tmp_path / "artifact"
    save_artifact(
        artifact_path,
        lexical=lexical,
        fusion=fusion,
        threshold=0.5,
        threshold_metric="f1",
        semantic_model_reference=None,
        training_metadata={"seed": 42},
        evaluation={"test": {"log_loss": 0.5}},
        split_assignments=pd.DataFrame({"row_number": [0], "split": ["test"]}),
    )
    loaded = load_artifact(artifact_path)
    original = lexical.predict_proba(pair_frame)
    restored = loaded.lexical.predict_proba(pair_frame)
    np.testing.assert_allclose(original, restored, rtol=1e-12, atol=1e-12)
    assert loaded.manifest["serialization"] == "json+npz; numpy allow_pickle=False"


def test_artifact_integrity_check_detects_tampering(tmp_path, pair_frame: pd.DataFrame) -> None:
    lexical, fusion = _trained_models(pair_frame)
    artifact_path = tmp_path / "artifact"
    save_artifact(
        artifact_path,
        lexical=lexical,
        fusion=fusion,
        threshold=0.5,
        threshold_metric="f1",
        semantic_model_reference=None,
        training_metadata={},
        evaluation={},
        split_assignments=pd.DataFrame({"row_number": [0], "split": ["train"]}),
    )
    vocabulary_path = artifact_path / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary["tampered"] = len(vocabulary)
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check failed"):
        load_artifact(artifact_path)


@pytest.fixture()
def saved_artifact(tmp_path, pair_frame: pd.DataFrame):
    lexical, fusion = _trained_models(pair_frame)
    artifact_path = tmp_path / "artifact"
    save_artifact(
        artifact_path,
        lexical=lexical,
        fusion=fusion,
        threshold=0.5,
        threshold_metric="f1",
        semantic_model_reference=None,
        training_metadata={},
        evaluation={},
        split_assignments=pd.DataFrame({"row_number": [0], "split": ["train"]}),
    )
    return artifact_path


def _update_hash(artifact_path, filename: str) -> None:
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = hashlib.sha256(
        (artifact_path / filename).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("extra_name", ["../outside.txt", "/tmp/outside.txt", "extra.txt"])
def test_manifest_rejects_unexpected_paths_before_reading(saved_artifact, extra_name: str) -> None:
    manifest_path = saved_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][extra_name] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the schema"):
        load_artifact(saved_artifact)


def test_artifact_rejects_symlinked_payload(saved_artifact, tmp_path) -> None:
    vocabulary_path = saved_artifact / "vocabulary.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(vocabulary_path.read_bytes())
    vocabulary_path.unlink()
    vocabulary_path.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        load_artifact(saved_artifact)


def test_artifact_rejects_duplicate_npz_member_even_with_matching_hash(saved_artifact) -> None:
    archive_path = saved_artifact / "model.npz"
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(archive_path, "a") as archive,
    ):
        archive.writestr("tfidf_idf.npy", b"duplicate")
    _update_hash(saved_artifact, "model.npz")
    with pytest.raises(ValueError, match="array set does not match"):
        load_artifact(saved_artifact)


def test_artifact_rejects_text_array_even_with_matching_hash(saved_artifact) -> None:
    archive_path = saved_artifact / "model.npz"
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["tfidf_idf"] = np.array(["1", "2"])
    np.savez_compressed(archive_path, **arrays)
    _update_hash(saved_artifact, "model.npz")
    with pytest.raises(ValueError, match="not numeric"):
        load_artifact(saved_artifact)
