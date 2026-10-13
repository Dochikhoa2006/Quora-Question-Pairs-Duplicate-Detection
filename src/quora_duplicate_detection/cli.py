"""Command-line interface for reproducible training and prediction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quora_duplicate_detection.artifacts import load_artifact
from quora_duplicate_detection.audit import audit_split
from quora_duplicate_detection.compare import compare_artifacts
from quora_duplicate_detection.config import TrainingConfig
from quora_duplicate_detection.data import load_pairs
from quora_duplicate_detection.pipeline import (
    create_predictions,
    create_predictions_chunked,
    evaluate_labeled_file,
    train_pipeline,
)
from quora_duplicate_detection.provenance import (
    sign_semantic_artifact,
    verify_signed_semantic_artifact,
)
from quora_duplicate_detection.repeated import evaluate_repeated
from quora_duplicate_detection.semantic import train_semantic_model, verify_semantic_artifact


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _labeled_paths(items: list[str], *, option: str) -> dict[str, str]:
    parsed = {}
    for item in items:
        label, separator, path = item.partition("=")
        if not separator or not label or not path or label in parsed:
            raise ValueError(f"{option} requires unique LABEL=PATH entries")
        parsed[label] = path
    return parsed


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
    predict.add_argument("--chunk-size", type=int, help="stream input in chunks of this many rows")
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
    evaluate.add_argument(
        "--slice-min-rows",
        type=int,
        default=20,
        help="omit aggregate slices with fewer rows (default: 20)",
    )

    repeated = subparsers.add_parser(
        "evaluate-repeated",
        help="repeat question-disjoint train/validation/test evaluation",
    )
    repeated.add_argument("--data", required=True, type=Path, help="labeled training CSV")
    repeated.add_argument("--output", required=True, type=Path, help="aggregate report JSON")
    repeated.add_argument("--config", type=Path, help="training JSON; split is forced to grouped")
    repeated.add_argument("--repeats", type=int, default=5, help="number of seeded runs (3-100)")
    repeated.add_argument("--semantic-model", help="optional local/Hugging Face semantic model")
    repeated.add_argument(
        "--compare-lexical", action="store_true", help="compare hybrid and lexical on each split"
    )

    compare = subparsers.add_parser(
        "compare-artifacts", help="compare saved detectors on one external labeled CSV"
    )
    compare.add_argument("--data", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--model", required=True, action="append", help="LABEL=ARTIFACT; repeat")
    compare.add_argument(
        "--training-data",
        action="append",
        default=[],
        help="LABEL=CSV used to train a model; verify provenance and reject overlapping questions",
    )
    compare.add_argument("--baseline", required=True, help="label of the baseline model")
    compare.add_argument(
        "--semantic-model", action="append", default=[], help="LABEL=MODEL override; repeat"
    )
    compare.add_argument("--bootstrap-samples", type=int, default=200)
    compare.add_argument("--seed", type=int, default=42)

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
    semantic.add_argument("--base-revision", help="Hub branch, tag, or 40-character commit SHA")
    semantic.add_argument("--validation-size", type=float, default=0.15)
    semantic.add_argument("--seed", type=int, default=42)
    semantic.add_argument("--epochs", type=float, default=1.0)
    semantic.add_argument("--batch-size", type=int, default=32)
    semantic.add_argument("--learning-rate", type=float, default=2e-5)
    semantic.add_argument("--lora", action="store_true", help="train a PEFT LoRA adapter")
    semantic.add_argument(
        "--resume", action="store_true", help="resume a matching checkpointed run"
    )

    inspect = subparsers.add_parser("inspect", help="verify and print an artifact manifest")
    inspect.add_argument("--artifact", required=True, type=Path)
    verify_semantic = subparsers.add_parser(
        "verify-semantic", help="verify saved semantic model checksums"
    )
    verify_semantic.add_argument("--artifact", required=True, type=Path)
    verify_semantic.add_argument(
        "--public-key", type=Path, help="verify detached Ed25519 provenance with this trusted key"
    )
    sign_semantic = subparsers.add_parser("sign-semantic", help="sign a completed semantic run")
    sign_semantic.add_argument("--artifact", required=True, type=Path)
    sign_semantic.add_argument("--private-key", required=True, type=Path)
    audit = subparsers.add_parser(
        "audit-split", help="replay saved splits and evaluation against labeled data"
    )
    audit.add_argument("--data", required=True, type=Path, help="original labeled CSV")
    audit.add_argument("--artifact", required=True, type=Path)
    audit.add_argument("--output", type=Path, help="optional aggregate audit JSON")
    audit.add_argument("--semantic-model", help="semantic model for hybrid evaluation replay")
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
        options = {
            "input_path": args.input,
            "artifact_dir": args.artifact,
            "output_path": args.output,
            "semantic_model": args.semantic_model,
            "semantic_batch_size": args.semantic_batch_size,
            "include_binary_prediction": args.include_binary_prediction,
        }
        if args.chunk_size is None:
            rows = len(create_predictions(**options))
        else:
            rows = create_predictions_chunked(**options, chunk_size=args.chunk_size)
        _print_json({"output": str(args.output), "rows": rows})
    elif args.command == "evaluate":
        _print_json(
            evaluate_labeled_file(
                data_path=args.data,
                artifact_dir=args.artifact,
                output_path=args.output,
                semantic_model=args.semantic_model,
                semantic_batch_size=args.semantic_batch_size,
                slice_min_rows=args.slice_min_rows,
            )
        )
    elif args.command == "evaluate-repeated":
        _print_json(
            evaluate_repeated(
                data_path=args.data,
                output_path=args.output,
                config=TrainingConfig.from_json(args.config),
                repeats=args.repeats,
                semantic_model=args.semantic_model,
                compare_lexical=args.compare_lexical,
            )
        )
    elif args.command == "compare-artifacts":
        _print_json(
            compare_artifacts(
                data_path=args.data,
                output_path=args.output,
                models=_labeled_paths(args.model, option="--model"),
                baseline=args.baseline,
                semantic_models=_labeled_paths(args.semantic_model, option="--semantic-model"),
                training_data=_labeled_paths(args.training_data, option="--training-data"),
                bootstrap_samples=args.bootstrap_samples,
                random_seed=args.seed,
            )
        )
    elif args.command == "train-semantic":
        frame = load_pairs(args.data, labeled=True)
        _print_json(
            train_semantic_model(
                frame,
                output_dir=args.output,
                base_model=args.base_model,
                base_revision=args.base_revision,
                validation_size=args.validation_size,
                random_seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_lora=args.lora,
                resume=args.resume,
            )
        )
    elif args.command == "inspect":
        _print_json(load_artifact(args.artifact).manifest)
    elif args.command == "verify-semantic":
        _print_json(
            verify_signed_semantic_artifact(args.artifact, args.public_key)
            if args.public_key
            else verify_semantic_artifact(args.artifact)
        )
    elif args.command == "sign-semantic":
        _print_json(sign_semantic_artifact(args.artifact, args.private_key))
    elif args.command == "audit-split":
        report = audit_split(
            data_path=args.data,
            artifact_dir=args.artifact,
            output_path=args.output,
            semantic_model=args.semantic_model,
        )
        _print_json(report)
        if not report["passed"]:
            raise SystemExit(1)
    else:  # pragma: no cover - argparse enforces valid commands
        parser.error(f"unsupported command: {args.command}")
