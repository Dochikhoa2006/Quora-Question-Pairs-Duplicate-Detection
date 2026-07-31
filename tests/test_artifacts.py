from __future__ import annotations

import json

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
