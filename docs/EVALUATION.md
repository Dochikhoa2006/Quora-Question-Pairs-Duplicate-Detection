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
pairs, textual overlap can make the estimate optimistic.

Before making scientific or production claims:

1. Construct groups so no normalized question identifier crosses partitions.
2. Repeat evaluation across several group-aware folds or seeds.
3. Report mean, dispersion, and confidence intervals.
4. Freeze the complete protocol before inspecting the final test results.
5. Evaluate slices such as question length, language, topic, and missing text.
6. Re-check probability calibration at the deployment class prevalence.

Do not tune against repeated Kaggle submissions and then describe the public
leaderboard result as an untouched test estimate.
