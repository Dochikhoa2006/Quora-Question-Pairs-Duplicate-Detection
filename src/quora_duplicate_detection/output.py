"""Atomic writers for user-facing report files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from quora_duplicate_detection.artifacts import PAYLOAD_FILES


def require_distinct_paths(source: str | Path, destination: str | Path) -> None:
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("input and output must be different files")


def require_output_outside_artifact_files(
    output_path: str | Path, artifact_dir: str | Path
) -> None:
    destination = Path(output_path).resolve()
    root = Path(artifact_dir)
    if any(destination == (root / name).resolve() for name in (*PAYLOAD_FILES, "manifest.json")):
        raise ValueError("report output must not overwrite artifact files")


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=".qqdup-",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
