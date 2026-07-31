# Model card: `<release name>`

> Release blocker: replace every `<FILL BEFORE RELEASE>` field and remove this
> notice before publishing weights or metrics. This file is intentionally a
> template, not a claim about the removed legacy checkpoint.

## Model details

| Field | Value |
|---|---|
| Version | `<FILL BEFORE RELEASE>` |
| Release date | `<FILL BEFORE RELEASE>` |
| Git commit | `<FILL BEFORE RELEASE>` |
| Artifact manifest SHA-256 | `<FILL BEFORE RELEASE>` |
| Model kind | `<lexical or hybrid>` |
| Semantic base model and exact revision | `<FILL BEFORE RELEASE or not applicable>` |
| Fine-tuning method | `<CoSENT full fine-tune, CoSENT LoRA, or not applicable>` |
| Languages evaluated | `<FILL BEFORE RELEASE>` |
| License | `<FILL BEFORE RELEASE>` |
| Contact | `<FILL BEFORE RELEASE>` |

## Intended use

Describe the concrete research or portfolio use:

`<FILL BEFORE RELEASE>`

Supported input:

- Two question strings in `question1` and `question2`
- Missing values normalized to empty strings
- Output probability interpreted as estimated duplicate intent

Out-of-scope uses:

- Automated moderation, sanctions, or access decisions
- Identity, authorship, plagiarism, or mental-state inference
- Unreviewed use in languages/domains absent from evaluation
- Treating a probability or binary output as definitive human judgment

## Training data

| Field | Value |
|---|---|
| Dataset name/version | `<FILL BEFORE RELEASE>` |
| Source and terms | `<FILL BEFORE RELEASE>` |
| Dataset fingerprint from manifest | `<FILL BEFORE RELEASE>` |
| Rows before/after filtering | `<FILL BEFORE RELEASE>` |
| Label prevalence | `<FILL BEFORE RELEASE>` |
| Missing-value counts | `<FILL BEFORE RELEASE>` |
| Known sensitive content | `<FILL BEFORE RELEASE>` |

Document all filtering, deduplication, normalization, and sampling. Do not
include raw personal or sensitive examples.

## Training procedure

Copy the exact configuration from `manifest.json` and record:

- Random seed: `<FILL BEFORE RELEASE>`
- Train/validation/test protocol: `<FILL BEFORE RELEASE>`
- Lexical estimator settings: `<FILL BEFORE RELEASE>`
- Semantic objective: `<FILL BEFORE RELEASE or not applicable>`
- LoRA configuration: `<FILL BEFORE RELEASE or not applicable>`
- Hardware and duration: `<FILL BEFORE RELEASE>`
- Python/container/dependency versions: `<FILL BEFORE RELEASE>`
- Upstream model revision and cache provenance: `<FILL BEFORE RELEASE>`

## Evaluation

State whether the partitions are pair-stratified or question-disjoint. Explain
that validation fits fusion/selects the threshold and test remains untouched
until final evaluation.

| Labeled partition | Rows | Positive rate | Log loss | Brier | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Validation (model selection) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Test (final local estimate) | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

Selected threshold and objective: `<FILL BEFORE RELEASE>`

| Labeled partition | Accuracy | Balanced accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Test | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

Competition test statement:

`No observed-label metric was computed for the hidden competition test set.`

Add slice results, uncertainty intervals, baseline comparisons, and error
analysis:

`<FILL BEFORE RELEASE>`

## Limitations and bias

At minimum address:

- repeated-question leakage under pair-level splitting
- domain/language shift
- class-prevalence and probability-calibration shift
- ambiguous labels and legitimate paraphrase disagreement
- performance on short, empty, numeric, and adversarial questions
- biases inherited from Quora data and any pretrained semantic encoder

Release-specific analysis:

`<FILL BEFORE RELEASE>`

## Privacy, safety, and ethics

Describe data handling, retention, access controls, redaction, annotator/user
consent limitations, and harmful-content handling:

`<FILL BEFORE RELEASE>`

## Artifact and loading

Artifact files:

`<FILL BEFORE RELEASE>`

Integrity verification command:

```bash
qqdup inspect --artifact <artifact-directory>
```

For hybrid releases, document how the semantic model is obtained and verified.
Hashes in the manifest detect corruption but do not prove publisher identity.
Link a signed release or provenance attestation if available.

## Reproducibility checklist

- [ ] All placeholders are completed.
- [ ] Dataset access and model licenses were reviewed.
- [ ] Git commit and artifact hashes are recorded.
- [ ] A fresh clone passes tests and the documented training command.
- [ ] No hidden test labels or sample-submission placeholders were used.
- [ ] Metric claims match `evaluation.json`.
- [ ] Split leakage limitations are disclosed.
- [ ] Secrets, datasets, local checkpoints, and pickle files are not committed.
- [ ] The release was reviewed by someone other than the model author.
