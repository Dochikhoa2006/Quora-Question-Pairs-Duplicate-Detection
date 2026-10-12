"""Opt-in, offline end-to-end semantic training smoke test."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import pytest

if os.environ.get("QQDUP_RUN_SEMANTIC_INTEGRATION") != "1":
    pytest.skip(
        "set QQDUP_RUN_SEMANTIC_INTEGRATION=1 to run semantic training", allow_module_level=True
    )

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")
pytest.importorskip("datasets")

from transformers import BertConfig, BertModel, BertTokenizerFast  # noqa: E402

from quora_duplicate_detection.semantic import SemanticScorer, train_semantic_model  # noqa: E402

pytestmark = pytest.mark.semantic_integration


def _tiny_local_bert(path) -> None:
    path.mkdir()
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "how",
        "learn",
        "study",
        "python",
        "guitar",
        "where",
        "buy",
        "coffee",
        "question",
    ]
    (path / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(path)
    model = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(path, safe_serialization=True)


@pytest.mark.parametrize("use_lora", [False, True], ids=["full", "lora"])
def test_offline_semantic_train_checkpoint_resume_and_scoring(
    tmp_path, record_property, use_lora: bool
) -> None:
    base_model = tmp_path / "tiny-bert"
    output = tmp_path / "run"
    _tiny_local_bert(base_model)
    rows = []
    for index in range(20):
        if index % 2 == 0:
            rows.append((f"how learn python question {index}", f"how study python {index}", 1))
        else:
            rows.append((f"how learn guitar question {index}", f"where buy coffee {index}", 0))
    frame = pd.DataFrame(rows, columns=["question1", "question2", "is_duplicate"])
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    record_property("semantic_device", device)
    options = {
        "output_dir": output,
        "base_model": str(base_model),
        "validation_size": 0.2,
        "random_seed": 42,
        "epochs": 1.0,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "use_lora": use_lora,
    }

    started = time.perf_counter()
    summary = train_semantic_model(frame, **options)
    record_property("semantic_train_seconds", round(time.perf_counter() - started, 3))
    if torch.cuda.is_available():
        record_property("semantic_cuda_peak_bytes", torch.cuda.max_memory_allocated())
    assert summary["trainer"]["global_step"] > 0
    assert summary["base_model_provenance"]["source"] == "local_directory"
    assert summary["evaluation"]
    assert json.loads((output / "training_summary.json").read_text(encoding="utf-8")) == summary
    scorer = SemanticScorer(output / "model", batch_size=2)
    assert np.isfinite(scorer.score(frame.iloc[:2])).all()

    (output / "training_summary.json").unlink()
    resumed = train_semantic_model(frame, **options, resume=True)
    assert resumed["resumed_from_checkpoint"].startswith("checkpoint-")
    assert resumed["trainer"]["global_step"] >= summary["trainer"]["global_step"]
