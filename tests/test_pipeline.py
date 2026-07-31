from __future__ import annotations

import json

import pandas as pd

from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.pipeline import create_predictions, train_pipeline


def test_end_to_end_lexical_training_and_prediction(
    tmp_path,
    pair_frame: pd.DataFrame,
) -> None:
    training_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    artifact_dir = tmp_path / "artifact"
    submission_csv = tmp_path / "submission.csv"
    pair_frame.to_csv(training_csv, index=False)
    pair_frame.drop(columns=["is_duplicate"]).iloc[:7].to_csv(test_csv, index=False)

    evaluation = train_pipeline(
        data_path=training_csv,
        output_dir=artifact_dir,
        config=TrainingConfig(
            validation_size=0.2,
            test_size=0.2,
            tfidf_max_features=2_000,
            tfidf_min_df=1,
        ),
    )
    predictions = create_predictions(
        input_path=test_csv,
        artifact_dir=artifact_dir,
        output_path=submission_csv,
    )

    assert evaluation["methodology"]["competition_test_labels_used"] is False
    assert set(predictions.columns) == {"id", "is_duplicate"}
    assert predictions["is_duplicate"].between(0, 1).all()
    assert not any(artifact_dir.glob("*.pkl"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision"]["comparison"] == "probability >= threshold"
