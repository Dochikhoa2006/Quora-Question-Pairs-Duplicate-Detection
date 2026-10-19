# Evaluation methodology

## Purpose

Evaluation answers two different questions:

1. How well does the current model generalize to labeled pairs withheld from
   model development?
2. What probability should be submitted for an unlabeled competition pair?

The second question cannot produce accuracy, F1, ROC-AUC, or any other
label-dependent metric until true labels exist. `sample_submission.csv` is a
format example, not ground truth.

## Default data protocol

With the default configuration and seed 42, labeled rows are stratified into:

- 70% training
- 15% validation
- 15% test

The exact row assignment is written to `split_assignments.csv`.

To evaluate generalization to unseen questions, set
`"split_strategy": "question_disjoint"` in the training JSON. This groups
rows by connected components of normalized questions: if two pairs share a
question directly or through other pairs, they stay in one partition. The
split is deterministic for a fixed seed. Component sizes can make the actual
row fractions and class balance differ from the requested values; training
fails if it cannot put both classes in every partition. The strategy is
recorded in the configuration and evaluation report.

The partitions have distinct roles:

| Partition | Permitted use |
|---|---|
| Train | Fit TF-IDF vocabulary, feature scaler, and lexical classifier |
| Validation | Fit logistic probability fusion/calibration and select the binary threshold |
| Test | One final local evaluation after all selection decisions |
| Competition test | Generate probabilities only; never report observed-label metrics |

The test partition is not used to select models, weights, hyperparameters, or
thresholds.

## Score fusion and calibration

Cosine similarity is bounded approximately by `[-1, 1]`; a lexical classifier
probability is bounded by `[0, 1]`. A hand-written weighted average assumes a
shared meaning and calibration that these scores do not have.

The replacement fits logistic stacking:

```text
p(duplicate) = sigmoid(
    intercept
    + beta_lexical * lexical_probability
    + beta_semantic * semantic_cosine
)
```

The semantic term is omitted for lexical-only runs. This learns a common
probability scale from validation labels. The untouched test set measures the
complete fitted decision system.

## Threshold selection

The default threshold objective is validation F1. Accuracy and balanced
accuracy are configurable. Candidate thresholds from 0.01 through 0.99 are
searched deterministically.

The decision rule is always:

```text
duplicate = probability >= selected_threshold
```

Thresholds are irrelevant to competition probability submissions. Changing a
threshold cannot improve probability log loss.

## Reported metrics

`evaluation.json` contains:

- **Log loss**: primary probability-quality measure; lower is better.
- **Brier score**: mean squared probability error; lower is better.
- **ROC-AUC**: ranking quality across thresholds.
- **Average precision**: precision-recall ranking summary.
- **Accuracy and balanced accuracy**: thresholded correctness.
- **Precision, recall, and F1**: positive-class threshold behavior.
- **Confusion matrix**: counts in `[[TN, FP], [FN, TP]]` order.
- **Calibration**: ten equal-width reliability bins with counts, mean predicted
  probability, observed positive rate, and expected calibration error (ECE).
  The final bin includes probability 1; empty bins have null means. ECE is
  the count-weighted mean absolute gap between the two rates. It depends on
  binning and should be read alongside Brier score and log loss.
- **Test slices**: aggregate error, Brier score, mean probability, class rate,
  and false-positive/false-negative rates for fixed shortest-question length
  and empty-question categories. Groups below 20 rows are omitted entirely.
  A missing class makes its conditional error rate null.

`qqdup evaluate` adds the same aggregate slices to an external labeled report.
Use `--slice-min-rows N` to raise the disclosure threshold. Reports contain no
question text, identifiers, or row-level predictions. The minimum row rule is
only a disclosure guard; these reports are not differentially private, and
aggregate rates may still be sensitive for some datasets.

The [Quora Question Pairs competition](https://www.kaggle.com/competitions/quora-question-pairs/overview/evaluation)
expects an `is_duplicate` probability for each `test_id`. Local accuracy is
useful only when its labels, split, and selected threshold are disclosed.

## Reproducibility record

Every artifact records:

- normalized dataset fingerprint
- row count
- random seed and complete JSON training configuration
- row-to-split assignments
- learned feature schema and component list
- threshold metric, value, and comparison direction
- validation and test reports
- SHA-256 digest for every artifact payload file

For a stronger release record, also capture the operating system, hardware,
Python version, container digest, Git commit SHA, upstream semantic model
revision, and the completed model card.

## Leakage and uncertainty

The default pair-level split is convenient and reproducible, but it does not
guarantee question-disjoint partitions. If the same question occurs in multiple
pairs, textual overlap can make the estimate optimistic. Use the opt-in grouped
split for a more demanding estimate, and report its actual partition sizes.

```bash
qqdup evaluate-repeated \
  --data data/train.csv \
  --output outputs/grouped.json \
  --repeats 5
```

This repeats the complete training, validation selection, and held-out test
evaluation under consecutive seeds, starting with the configuration seed.
It forces question-disjoint splits and writes a compact JSON report with the
dataset fingerprint, configuration, per-seed split sizes and test metrics, and
mean, sample standard deviation, and 2.5/97.5 percentiles. Temporary model
artifacts are removed after each run. The percentiles describe variation across
these overlapping holdouts; they are not a confidence interval or an independent
test set. Treat this as exploratory stability analysis and keep a separately
protected final test protocol for claims.
The report also shows how many rows were tested at least once, how often each
row appeared in a test partition, and the minimum, mean, and maximum pairwise
Jaccard overlap between test sets. These aggregate counts help reveal repeated
exposure to the same examples without publishing row identifiers.

To compare a semantic hybrid model with the lexical baseline on the same
partition for each seed, add `--semantic-model path/to/model --compare-lexical`.
The report includes both test metric sets and `hybrid_minus_lexical` differences,
with a descriptive summary of those paired differences. Lower log loss, Brier
score, and calibration error are better; higher ranking and classification
metrics are better. Both models use the same split assignments for each seed.
This is exploratory model comparison because repeated test sets overlap.

For a frozen comparison across two or more saved detectors, use a separate
labeled file:

```bash
qqdup compare-artifacts \
  --data external.csv \
  --output report.json \
  --model lexical=artifacts/lexical \
  --model cosent=artifacts/cosent \
  --training-data lexical=data/lexical_train.csv \
  --training-data cosent=data/cosent_train.csv \
  --baseline lexical
```

Each artifact supplies its already selected threshold; all models score the
same rows. The command rejects a file whose complete dataset fingerprint
matches any artifact's training data, and rejects duplicate artifact manifests
under different labels. With a matching `--training-data` file for each model,
it also rejects external rows sharing a nonempty normalized question or an exact
pair with that model's training data, including reversed pairs. The report records
`null` for models without an overlap check. Protect the external set
independently and do not use it to tune or choose a winner before making a
final claim.

The report contains artifact manifest hashes, aggregate metrics, candidate
minus baseline differences, and 2.5/97.5 percentiles from a stratified paired
bootstrap. The resampling preserves class counts and uses the same sampled rows
for every model. These percentiles describe sampling variation on the chosen
external set, conditional on fixed trained artifacts and thresholds; they do
not include training variation. No question text or row predictions are saved.

Before making scientific or production claims:

1. Construct groups so no normalized question identifier crosses partitions.
2. Repeat evaluation across several group-aware folds or seeds.
3. Report mean and dispersion; use a justified uncertainty method when a
   confidence interval is required.
4. Freeze the complete protocol before inspecting the final test results.
5. Evaluate slices such as question length, language, topic, and missing text.
6. Re-check probability calibration at the deployment class prevalence.

Do not tune against repeated Kaggle submissions and then describe the public
leaderboard result as an untouched test estimate.
