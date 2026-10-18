"""Detached Ed25519 provenance for completed semantic training runs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Any

from quora_duplicate_detection.output import write_json_atomic
from quora_duplicate_detection.semantic import verify_semantic_artifact

SIGNED_FILES = ("run_config.json", "training_summary.json")
PROVENANCE_FILE = "semantic_provenance.json"


def _crypto():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise RuntimeError(
            "semantic signing requires: pip install -r requirements-signing.txt"
        ) from exc
    return InvalidSignature, serialization, Ed25519PrivateKey, Ed25519PublicKey


def _signed_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _file_digests(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SIGNED_FILES}


def sign_semantic_artifact(output_dir: str | Path, private_key_path: str | Path) -> dict[str, Any]:
    """Sign the run config and checksum summary after verifying the model files."""
    root = Path(output_dir)
    key_path = Path(private_key_path)
    if key_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("private key must be outside the semantic artifact directory")
    verify_semantic_artifact(root)
    destination = root / PROVENANCE_FILE
    if destination.exists():
        raise FileExistsError(f"semantic provenance already exists: {destination}")
    _, serialization, Ed25519PrivateKey, _ = _crypto()
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key must be Ed25519")
    public_bytes = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload = {
        "schema_version": 1,
        "algorithm": "Ed25519",
        "public_key_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "files_sha256": _file_digests(root),
    }
    signed = {
        **payload,
        "signature_base64": base64.b64encode(key.sign(_signed_bytes(payload))).decode("ascii"),
    }
    write_json_atomic(destination, signed)
    return {
        "signed": True,
        "public_key_sha256": payload["public_key_sha256"],
        "provenance": str(destination),
    }


def verify_signed_semantic_artifact(
    output_dir: str | Path, public_key_path: str | Path
) -> dict[str, Any]:
    """Verify a trusted public key, detached signature, metadata, and all model files."""
    root = Path(output_dir)
    check = verify_semantic_artifact(root)
    record = json.loads((root / PROVENANCE_FILE).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("semantic provenance must be a JSON object")
    signature_text = record.get("signature_base64")
    payload = {key: value for key, value in record.items() if key != "signature_base64"}
    if (
        set(payload) != {"schema_version", "algorithm", "public_key_sha256", "files_sha256"}
        or payload["schema_version"] != 1
        or payload["algorithm"] != "Ed25519"
        or payload["files_sha256"] != _file_digests(root)
        or not isinstance(signature_text, str)
    ):
        raise ValueError("semantic provenance metadata is invalid or files changed")
    InvalidSignature, serialization, _, Ed25519PublicKey = _crypto()
    key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key must be Ed25519")
    public_bytes = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if payload["public_key_sha256"] != hashlib.sha256(public_bytes).hexdigest():
        raise ValueError("semantic provenance was signed by a different key")
    try:
        signature = base64.b64decode(signature_text, validate=True)
        key.verify(signature, _signed_bytes(payload))
    except (ValueError, binascii.Error, InvalidSignature) as exc:
        raise ValueError("semantic provenance signature is invalid") from exc
    return {**check, "signature_verified": True, "public_key_sha256": payload["public_key_sha256"]}
