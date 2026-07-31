"""Optional Sentence Transformers scoring and CoSENT fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quora_duplicate_detection.data import LABEL_COLUMN, PAIR_COLUMNS, validate_pairs


class SemanticScorer:
    def __init__(self, model_path: str | Path, *, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "semantic scoring requires: pip install -r requirements-semantic.txt"
            ) from exc
        self.model = SentenceTransformer(str(model_path))
        self.batch_size = batch_size

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        clean = validate_pairs(frame, labeled=False)
        left = self.model.encode(
            clean[PAIR_COLUMNS[0]].tolist(),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        right = self.model.encode(
            clean[PAIR_COLUMNS[1]].tolist(),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.sum(np.asarray(left) * np.asarray(right), axis=1)


def train_semantic_model(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    base_model: str,
    validation_size: float,
    random_seed: int,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    use_lora: bool,
) -> dict[str, Any]:
    """Fine-tune one documented CoSENT objective with a held-out evaluator."""
    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.sentence_transformer import losses
        from sentence_transformers.sentence_transformer.evaluation import (
            BinaryClassificationEvaluator,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "semantic training requires: pip install -r requirements-semantic.txt"
        ) from exc

    clean = validate_pairs(frame, labeled=True)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"semantic output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_idx, validation_idx = train_test_split(
        np.arange(len(clean)),
        test_size=validation_size,
        random_state=random_seed,
        stratify=clean[LABEL_COLUMN].to_numpy(),
    )
    train_frame = clean.iloc[train_idx]
    validation_frame = clean.iloc[validation_idx]

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model = SentenceTransformer(base_model)
    if use_lora:
        from peft import LoraConfig, TaskType

        model.add_adapter(
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
            )
        )

    train_dataset = Dataset.from_dict(
        {
            "question1": train_frame[PAIR_COLUMNS[0]].tolist(),
            "question2": train_frame[PAIR_COLUMNS[1]].tolist(),
            "label": train_frame[LABEL_COLUMN].astype(float).tolist(),
        }
    )
    validation_dataset = Dataset.from_dict(
        {
            "question1": validation_frame[PAIR_COLUMNS[0]].tolist(),
            "question2": validation_frame[PAIR_COLUMNS[1]].tolist(),
            "label": validation_frame[LABEL_COLUMN].astype(float).tolist(),
        }
    )
    evaluator = BinaryClassificationEvaluator(
        validation_frame[PAIR_COLUMNS[0]].tolist(),
        validation_frame[PAIR_COLUMNS[1]].tolist(),
        validation_frame[LABEL_COLUMN].astype(int).tolist(),
        name="held_out",
        batch_size=batch_size,
        show_progress_bar=False,
        write_csv=True,
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=random_seed,
        data_seed=random_seed,
        push_to_hub=False,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss=losses.CoSENTLoss(model),
        evaluator=evaluator,
    )
    trainer.train()
    evaluation = evaluator(model, output_path=str(output))
    model.save(
        str(output / "model"),
        create_model_card=False,
        safe_serialization=True,
    )

    summary: dict[str, Any] = {
        "objective": "CoSENTLoss",
        "base_model": base_model,
        "use_lora": use_lora,
        "random_seed": random_seed,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "evaluation": {key: float(value) for key, value in evaluation.items()},
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
