from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quora_duplicate_detection.semantic import (
    SemanticScorer,
    _model_file_hashes,
    _numeric_telemetry,
    _prepare_training_run,
    _resolve_base_model,
    verify_semantic_artifact,
)


def test_semantic_scorer_rejects_invalid_batch_size_before_loading_model() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SemanticScorer("unused-model", batch_size=0)


@pytest.mark.parametrize(
    "second_embedding",
    [np.array([[1.0, 0.0, 0.0]]), np.array([[np.nan, 0.0]])],
)
def test_semantic_scorer_rejects_invalid_embeddings(second_embedding: np.ndarray) -> None:
    scorer = SemanticScorer.__new__(SemanticScorer)
    embeddings = iter((np.array([[1.0, 0.0]]), second_embedding))
    scorer.model = SimpleNamespace(encode=lambda *args, **kwargs: next(embeddings))
    scorer.batch_size = 1
    frame = pd.DataFrame({"question1": ["first"], "question2": ["second"]})
    with pytest.raises(ValueError, match="matching finite row vectors"):
        scorer.score(frame)


def test_semantic_scorer_keeps_valid_pair_scores() -> None:
    scorer = SemanticScorer.__new__(SemanticScorer)
    embeddings = iter((np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 0.0]])))
    scorer.model = SimpleNamespace(encode=lambda *args, **kwargs: next(embeddings))
    scorer.batch_size = 2
    frame = pd.DataFrame({"question1": ["a", "b"], "question2": ["a", "c"]})
    np.testing.assert_allclose(scorer.score(frame), [1.0, 0.0])


def test_semantic_checksum_verifier_detects_model_changes(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights")
    summary = {
        "use_lora": False,
        "model_files_sha256": _model_file_hashes(model),
        "adapter_files_sha256": None,
    }
    (tmp_path / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert verify_semantic_artifact(tmp_path) == {
        "passed": True,
        "model_files": 1,
        "adapter_files": 0,
    }
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum manifest"):
        verify_semantic_artifact(tmp_path)
    weights.write_bytes(b"weights")
    (model / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum manifest"):
        verify_semantic_artifact(tmp_path)


def test_semantic_checksum_verifier_requires_complete_adapter(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    adapter = model / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    hashes = _model_file_hashes(model)
    summary = {"use_lora": True, "model_files_sha256": hashes, "adapter_files_sha256": hashes}
    (tmp_path / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter checksum manifest"):
        verify_semantic_artifact(tmp_path)


def test_hub_revision_is_resolved_and_pinned_before_model_load(monkeypatch) -> None:
    resolved = "a" * 40
    calls = []

    class FakeApi:
        def model_info(self, model, *, revision, timeout):
            calls.append((model, revision, timeout))
            return SimpleNamespace(sha=resolved)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    provenance, kwargs = _resolve_base_model("organization/model", "release-v1")
    assert calls == [("organization/model", "release-v1", 30)]
    assert provenance == {"source": "huggingface_hub", "revision": resolved, "sha256": None}
    assert kwargs == {"revision": resolved}


def test_immutable_hub_revision_does_not_require_metadata_lookup(monkeypatch) -> None:
    revision = "b" * 40
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    provenance, kwargs = _resolve_base_model("organization/model", revision)
    assert provenance["revision"] == revision
    assert kwargs == {"revision": revision}


def test_local_model_digest_tracks_contents_and_rejects_revision(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"a":1}', encoding="utf-8")
    first, kwargs = _resolve_base_model(str(model_dir), None)
    assert kwargs == {}
    assert first["source"] == "local_directory"
    assert first["revision"] is None
    assert len(first["sha256"]) == hashlib.sha256().digest_size * 2

    (model_dir / "config.json").write_text('{"a":2}', encoding="utf-8")
    second, _ = _resolve_base_model(str(model_dir), None)
    assert second["sha256"] != first["sha256"]
    with pytest.raises(ValueError, match="local model directory"):
        _resolve_base_model(str(model_dir), "main")


def test_hub_response_without_commit_sha_is_rejected(monkeypatch) -> None:
    class FakeApi:
        def model_info(self, model, *, revision, timeout):
            return SimpleNamespace(sha="main")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    with pytest.raises(ValueError, match="immutable model revision"):
        _resolve_base_model("organization/model", None)


def test_resume_requires_matching_run_and_complete_checkpoint(tmp_path) -> None:
    model_dir = tmp_path / "base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("model-v1", encoding="utf-8")
    output = tmp_path / "run"
    frame = pd.DataFrame({"question1": ["a", "b"], "question2": ["c", "d"], "is_duplicate": [0, 1]})
    options = {
        "frame": frame,
        "output": output,
        "base_model": str(model_dir),
        "base_revision": None,
        "validation_size": 0.2,
        "random_seed": 42,
        "epochs": 2.0,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "use_lora": False,
    }
    provenance, kwargs, checkpoint = _prepare_training_run(**options, resume=False)
    assert kwargs == {}
    assert checkpoint is None
    saved = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert saved["base_model_provenance"] == provenance

    with pytest.raises(ValueError, match="no complete trainer checkpoint"):
        _prepare_training_run(**options, resume=True)
    incomplete = output / "checkpoints" / "checkpoint-4"
    incomplete.mkdir(parents=True)
    complete = output / "checkpoints" / "checkpoint-3"
    complete.mkdir()
    (complete / "trainer_state.json").write_text("{}", encoding="utf-8")
    _, _, resumed = _prepare_training_run(**options, resume=True)
    assert resumed == complete

    with pytest.raises(ValueError, match="settings or dataset differ"):
        _prepare_training_run(**{**options, "epochs": 3.0}, resume=True)
    changed = frame.copy()
    changed.loc[0, "question1"] = "changed"
    with pytest.raises(ValueError, match="settings or dataset differ"):
        _prepare_training_run(**{**options, "frame": changed}, resume=True)
    (model_dir / "config.json").write_text("model-v2", encoding="utf-8")
    with pytest.raises(ValueError, match="base model contents changed"):
        _prepare_training_run(**options, resume=True)


def test_numeric_telemetry_omits_nonfinite_and_non_numeric_values() -> None:
    assert _numeric_telemetry({"loss": 0.5, "step": 4, "bad": float("nan"), "text": "x"}) == {
        "loss": 0.5,
        "step": 4.0,
    }


def test_hub_resume_uses_saved_commit_without_new_lookup(tmp_path, monkeypatch) -> None:
    from quora_duplicate_detection.data import dataset_fingerprint

    frame = pd.DataFrame({"question1": ["a", "b"], "question2": ["c", "d"], "is_duplicate": [0, 1]})
    output = tmp_path / "run"
    checkpoint = output / "checkpoints" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    revision = "d" * 40
    saved = {
        "schema_version": 1,
        "dataset_fingerprint_sha256": dataset_fingerprint(frame),
        "base_model": "organization/model",
        "requested_base_revision": "main",
        "validation_size": 0.2,
        "random_seed": 42,
        "epochs": 2.0,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "use_lora": False,
        "base_model_provenance": {
            "source": "huggingface_hub",
            "revision": revision,
            "sha256": None,
        },
    }
    (output / "run_config.json").write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    provenance, kwargs, selected = _prepare_training_run(
        frame=frame,
        output=output,
        base_model="organization/model",
        base_revision="main",
        validation_size=0.2,
        random_seed=42,
        epochs=2.0,
        batch_size=8,
        learning_rate=2e-5,
        use_lora=False,
        resume=True,
    )
    assert provenance["revision"] == revision
    assert kwargs == {"revision": revision}
    assert selected == checkpoint
