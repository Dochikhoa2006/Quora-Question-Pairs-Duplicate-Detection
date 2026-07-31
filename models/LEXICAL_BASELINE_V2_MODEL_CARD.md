# Model card: lexical baseline v2

## Status

This card records a reproducible aggregate evaluation candidate generated on
2026-07-31. Model weights and the learned vocabulary are not distributed in the
repository. Bind the report to the final public Git commit before creating a
tagged model release.

## Model details

| Field | Value |
|---|---|
| Software version | 2.0.0 |
| Model kind | Lexical logistic baseline with logistic probability calibration |
| Semantic component | None |
| Input | Two question strings |
| Output | Estimated duplicate probability |
| Binary rule | `probability >= 0.23` when a decision is requested |
| Threshold objective | Validation F1 |
| License | CC BY 4.0 for project code; dataset terms remain separate |

The model uses 14 symmetric features covering lengths, token sets, sequence
similarity, numeric/boundary agreement, exact normalized match, and character
n-gram TF-IDF cosine. It uses no NLTK corpus or downloaded lexical resource.

## Intended use

The baseline supports research, education, portfolio demonstration, and
comparison with optional semantic models. It is suitable for offline scoring of
English-like question pairs after domain-specific evaluation.

It is not intended for moderation, sanctions, identity/authorship inference,
plagiarism judgments, or unreviewed production decisions.

## Training data

| Field | Value |
|---|---:|
| Dataset | Quora Question Pairs labeled training data |
| Rows | 404,290 |
| Positive rows | 149,263 |
| Positive rate | 36.92% |
| Missing `question1` | 1 |
| Missing `question2` | 2 |
| Normalized fingerprint | `c0ddfee0b0a0c2f7f175973bd28026470ba2afce4e4b2d6f742e46011c604dfe` |

The dataset is not redistributed. It was normalized with Unicode NFKC, case
folding, whitespace collapse, and explicit missing-value handling.

## Training and evaluation procedure

- Seed: 42
- Split: pair-level stratified 70% train / 15% validation / 15% test
- Train rows: 283,002
- Validation rows: 60,644
- Test rows: 60,644
- TF-IDF: character word-boundary n-grams 3–5, maximum 50,000 features
- Validation use: fit the probability calibration layer and select F1 threshold
- Test use: one final local evaluation after selection
- Competition test labels: not used and not available

### Aggregate metrics

| Labeled partition | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | Avg. precision ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Validation (selection) | 0.5322 | 0.1816 | 0.7884 | 0.6251 | 0.6707 |
| Test (local held-out) | **0.5300** | **0.1807** | **0.7905** | **0.6299** | **0.6728** |

At threshold 0.23, local test accuracy was 0.6749, balanced accuracy
0.7227, precision 0.5353, and recall 0.9055. The threshold reflects an F1
objective and therefore favors recall; it is not used for probability
submissions.

Full precision and provenance are stored in
[`reports/lexical-v2-pair-stratified.json`](../reports/lexical-v2-pair-stratified.json).

## Limitations and bias

- The split is pair-stratified, not question-disjoint. Repeated questions may
  cross partitions and make results optimistic.
- The result is a local labeled holdout, not a hidden competition test score.
- Quora questions and labels are not representative of all users, languages,
  time periods, or domains.
- The low F1-selected threshold produces high recall (0.9055) and modest
  precision (0.5353); applications with different error costs need a separately
  selected threshold on representative validation data.
- Calibration may shift when duplicate prevalence differs.
- Ambiguous intent and annotation disagreement impose an irreducible label limit.
- No language, topic, fairness, adversarial, or question-disjoint slice results
  are yet published.

## Privacy, safety, and ethics

Only aggregate statistics are committed. Raw questions, row-level predictions,
vocabulary, split assignments, and model parameters remain outside the
repository. Users should minimize retention, review dataset terms, and avoid
publishing sensitive error examples.

## Reproducibility

Environment:

- Python 3.12.13
- macOS 15.7.4 arm64
- NumPy 2.4.4
- pandas 2.3.3
- SciPy 1.17.1
- scikit-learn 1.8.0

Command:

```bash
qqdup train \
  --data data/train.csv \
  --output artifacts/lexical-v2 \
  --config config/default.json
```

Generated artifact evidence:

- Manifest SHA-256: `bb9a5c7ac050ca18983e8ebf747ea23aa4e4259b1415f5fd7fc5866a55e8f201`
- Model NPZ SHA-256: `0e23637ca5e7d309b542a4c7010f35085a80a1407a2b9aa54b0f177a6024f2da`
- Serialization: JSON plus numeric NPZ loaded with `allow_pickle=False`

The artifact is not part of this source release. These hashes identify the
local evaluation candidate but do not authenticate its publisher.
