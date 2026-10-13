from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quora_duplicate_detection import cli
from quora_duplicate_detection.provenance import (
    sign_semantic_artifact,
    verify_signed_semantic_artifact,
)
from quora_duplicate_detection.semantic import _model_file_hashes


@pytest.fixture()
def signed_run(tmp_path):
    artifact = tmp_path / "run"
    model = artifact / "model"
    model.mkdir(parents=True)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model / "adapter_model.safetensors").write_bytes(b"adapter weights")
    hashes = _model_file_hashes(model)
    summary = {"use_lora": True, "model_files_sha256": hashes, "adapter_files_sha256": hashes}
    (artifact / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (artifact / "run_config.json").write_text('{"seed":42}', encoding="utf-8")
    key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return artifact, private_path, public_path


def test_signed_semantic_provenance_round_trip(signed_run, capsys) -> None:
    artifact, private_path, public_path = signed_run
    cli.main(["sign-semantic", "--artifact", str(artifact), "--private-key", str(private_path)])
    assert json.loads(capsys.readouterr().out)["signed"]
    cli.main(["verify-semantic", "--artifact", str(artifact), "--public-key", str(public_path)])
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] and report["signature_verified"]
    assert report["adapter_files"] == 2
    with pytest.raises(FileExistsError):
        sign_semantic_artifact(artifact, private_path)


def test_signed_semantic_provenance_detects_tampering_and_wrong_key(signed_run) -> None:
    artifact, private_path, public_path = signed_run
    sign_semantic_artifact(artifact, private_path)
    other = Ed25519PrivateKey.generate().public_key()
    wrong_key = artifact.parent / "wrong.pem"
    wrong_key.write_bytes(
        other.public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    with pytest.raises(ValueError, match="different key"):
        verify_signed_semantic_artifact(artifact, wrong_key)
    config = artifact / "run_config.json"
    config.write_text('{"seed":43}', encoding="utf-8")
    with pytest.raises(ValueError, match="files changed"):
        verify_signed_semantic_artifact(artifact, public_path)
    config.write_text('{"seed":42}', encoding="utf-8")
    provenance = artifact / "semantic_provenance.json"
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["signature_base64"] = "AAAA"
    provenance.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="signature is invalid"):
        verify_signed_semantic_artifact(artifact, public_path)
    (artifact / "model" / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum manifest"):
        verify_signed_semantic_artifact(artifact, public_path)
    summary_path = artifact / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    new_hashes = _model_file_hashes(artifact / "model")
    summary["model_files_sha256"] = new_hashes
    summary["adapter_files_sha256"] = new_hashes
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="files changed"):
        verify_signed_semantic_artifact(artifact, public_path)


def test_private_key_cannot_be_stored_inside_artifact(signed_run) -> None:
    artifact, private_path, _ = signed_run
    internal_key = artifact / "private.pem"
    internal_key.write_bytes(private_path.read_bytes())
    with pytest.raises(ValueError, match="outside"):
        sign_semantic_artifact(artifact, internal_key)
