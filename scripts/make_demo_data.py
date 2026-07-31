"""Generate a small synthetic dataset for an offline end-to-end smoke test."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

POSITIVE_PATTERNS = (
    ("How can I learn {topic} quickly?", "What is the fastest way to learn {topic}?"),
    ("What causes {topic}?", "Why does {topic} happen?"),
    ("Is {topic} useful for beginners?", "Should a beginner use {topic}?"),
    ("How much does {topic} cost?", "What is the price of {topic}?"),
)
NEGATIVE_PATTERNS = (
    ("How can I learn {topic}?", "Where can I buy {other}?"),
    ("What causes {topic}?", "How do I cook {other}?"),
    ("Is {topic} useful?", "When was {other} invented?"),
    ("How much does {topic} cost?", "Why is {other} popular?"),
)
TOPICS = (
    "Python",
    "machine learning",
    "linear algebra",
    "data science",
    "photography",
    "public speaking",
    "guitar",
    "Vietnamese",
)
OTHERS = (
    "a bicycle",
    "brown rice",
    "the telescope",
    "football",
    "a coffee grinder",
    "the piano",
    "a train ticket",
    "green tea",
)


def generate_rows(row_count: int) -> list[dict[str, object]]:
    if row_count < 20:
        raise ValueError("row_count must be at least 20")
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        duplicate = index % 2 == 0
        topic = TOPICS[index % len(TOPICS)]
        other = OTHERS[(index * 3 + 1) % len(OTHERS)]
        patterns = POSITIVE_PATTERNS if duplicate else NEGATIVE_PATTERNS
        question1, question2 = patterns[index % len(patterns)]
        rows.append(
            {
                "id": index,
                "question1": question1.format(topic=topic, other=other),
                "question2": question2.format(topic=topic, other=other),
                "is_duplicate": int(duplicate),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo_train.csv"))
    parser.add_argument("--rows", type=int, default=120)
    args = parser.parse_args()

    rows = generate_rows(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labeled pairs to {args.output}")


if __name__ == "__main__":
    main()
