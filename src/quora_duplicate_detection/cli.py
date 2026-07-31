"""Command-line interface for reproducible training and prediction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quora_duplicate_detection.artifacts import load_artifact
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.data import load_pairs
from quora_duplicate_detection.pipeline import (
    create_predictions,
    evaluate_labeled_file,
    train_pipeline,
)
from quora_duplicate_detection.semantic import train_semantic_model


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qqdup",
        description="Train and evaluate duplicate-question detectors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train and evaluate a lexical/hybrid model")
    train.add_argument("--data", required=True, type=Path, help="labeled training CSV")
    train.add_argument("--output", required=True, type=Path, help="new artifact directory")
    train.add_argument("--config", type=Path, help="training JSON; defaults are built in")
    train.add_argument("--semantic-model", help="optional local/Hugging Face semantic model")

    predict = subparsers.add_parser("predict", help="write probabilities for unlabeled pairs")
    predict.add_argument("--input", required=True, type=Path, help="unlabeled pairs CSV")
    predict.add_argument("--artifact", required=True, type=Path, help="artifact directory")
    predict.add_argument("--output", required=True, type=Path, help="output CSV")
    predict.add_argument("--semantic-model", help="override semantic model reference")
    predict.add_argument("--semantic-batch-size", type=int, default=64)
    predict.add_argument(
        "--include-binary-prediction",
        action="store_true",
        help="add a thresholded prediction column; submissions should omit this",
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate a separate labeled CSV")
    evaluate.add_argument("--data", required=True, type=Path, help="labeled pairs CSV")
    evaluate.add_argument("--artifact", required=True, type=Path)
    evaluate.add_argument("--output", type=Path, help="optional metrics JSON")
    evaluate.add_argument("--semantic-model", help="override semantic model reference")
    evaluate.add_argument("--semantic-batch-size", type=int, default=64)

    semantic = subparsers.add_parser(
        "train-semantic",
        help="optionally fine-tune Sentence Transformers with CoSENT",
    )
    semantic.add_argument("--data", required=True, type=Path, help="labeled training CSV")
    semantic.add_argument("--output", required=True, type=Path)
    semantic.add_argument(
        "--base-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    semantic.add_argument("--validation-size", type=float, default=0.15)
    semantic.add_argument("--seed", type=int, default=42)
    semantic.add_argument("--epochs", type=float, default=1.0)
    semantic.add_argument("--batch-size", type=int, default=32)
    semantic.add_argument("--learning-rate", type=float, default=2e-5)
    semantic.add_argument("--lora", action="store_true", help="train a PEFT LoRA adapter")

    inspect = subparsers.add_parser("inspect", help="verify and print an artifact manifest")
    inspect.add_argument("--artifact", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainingConfig.from_json(args.config)
        _print_json(
            train_pipeline(
                data_path=args.data,
                output_dir=args.output,
                config=config,
                semantic_model=args.semantic_model,
            )
        )
    elif args.command == "predict":
        output = create_predictions(
            input_path=args.input,
            artifact_dir=args.artifact,
            output_path=args.output,
            semantic_model=args.semantic_model,
            semantic_batch_size=args.semantic_batch_size,
            include_binary_prediction=args.include_binary_prediction,
        )
        _print_json({"output": str(args.output), "rows": len(output)})
    elif args.command == "evaluate":
        _print_json(
            evaluate_labeled_file(
                data_path=args.data,
                artifact_dir=args.artifact,
                output_path=args.output,
                semantic_model=args.semantic_model,
                semantic_batch_size=args.semantic_batch_size,
            )
        )
    elif args.command == "train-semantic":
        frame = load_pairs(args.data, labeled=True)
        _print_json(
            train_semantic_model(
                frame,
                output_dir=args.output,
                base_model=args.base_model,
                validation_size=args.validation_size,
                random_seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_lora=args.lora,
            )
        )
    elif args.command == "inspect":
        _print_json(load_artifact(args.artifact).manifest)
    else:  # pragma: no cover - argparse enforces valid commands
        parser.error(f"unsupported command: {args.command}")
