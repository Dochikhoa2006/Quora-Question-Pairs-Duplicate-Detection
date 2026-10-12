from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

from quora_duplicate_detection.semantic import _resolve_base_model


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
