# Data card

## Dataset

The intended research dataset is the Quora Question Pairs data distributed
through the official Kaggle competition. The repository does not redistribute
it. Users must obtain access from the provider and accept the applicable terms.

Source: [Quora Question Pairs on Kaggle](https://www.kaggle.com/competitions/quora-question-pairs/data)

## Intended task

Given two question strings, estimate the probability that they represent the
same intent. The training label is binary:

- `1`: labeled duplicate
- `0`: labeled non-duplicate

The project expects these fields:

| Field | Required for training | Required for prediction | Meaning |
|---|---:|---:|---|
| `question1` | Yes | Yes | First question text |
| `question2` | Yes | Yes | Second question text |
| `is_duplicate` | Yes | No | Observed binary label |
| `id` / `test_id` | No | No | Preserved output identifier when present |
| `qid1` / `qid2` | No | No | Provider question identifiers; useful for grouped analysis |

## Repository data policy

The following are intentionally ignored:

- raw CSV datasets
- generated train/test subsets
- user-provided question text
- model artifacts and checkpoints
- environment and credential files

Tests use synthetic question pairs. Generated demo data is clearly identified as
synthetic and is not committed as benchmark evidence.

## Processing

At the input boundary:

1. Required columns are checked.
2. Missing question values become empty strings.
3. Text is normalized with Unicode NFKC, case folding, and whitespace collapse.
4. Labels must be non-missing binary values and both classes must be present.
5. A normalized dataset fingerprint is recorded.
6. Labeled rows are split deterministically according to configuration.

No downloaded lemma dictionary, spelling correction, translation, or language
detection is applied.

## Known data risks

- Duplicate labels can be subjective or noisy.
- Questions may contain personal, offensive, medical, legal, political, or
  otherwise sensitive content.
- Quora users and topics are not representative of all populations or domains.
- Language and dialect performance may be uneven.
- Repeated questions can cross pair-level splits and make estimates optimistic.
- The provider's hidden test distribution and labels are not available for local
  metric calculation.

## Recommended analysis before release

- Label prevalence and missing-text counts by partition
- Normalized exact-pair duplication across partitions
- Question-identifier overlap and question-disjoint evaluation
- Question length, language, topic, and numeric-content slices
- Calibration curves and class-prevalence sensitivity
- Manual review of false positives/negatives using privacy-safe procedures
- Confidence intervals or repeated grouped folds

## Ethical use

Use predictions as decision support, not definitive judgments. Do not use the
model for sanctions, identity inference, authorship attribution, or access
decisions. Minimize retention of raw text and avoid publishing sensitive
examples in reports, issues, screenshots, or model cards.
