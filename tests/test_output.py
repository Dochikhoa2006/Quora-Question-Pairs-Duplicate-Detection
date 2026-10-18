from __future__ import annotations

import pytest

from quora_duplicate_detection.output import write_json_atomic


def test_json_writer_rejects_nonfinite_values_without_replacing_report(tmp_path) -> None:
    destination = tmp_path / "report.json"
    destination.write_text('{"previous": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_atomic(destination, {"metric": float("nan")})

    assert destination.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert not list(tmp_path.glob(".qqdup-*.json"))
