# Full technical audit

## Executive summary

The original repository demonstrated a useful hybrid modeling idea, but its
evaluation and inference paths could not support public performance claims. The
most severe issue was treating `sample_submission.csv` placeholders as observed
labels. The replacement separates labeled evaluation from unlabeled
competition inference, learns score fusion, uses one consistent decision
direction, and records enough metadata to reproduce a run.

This audit describes the code found before the version 2 migration. It is not a
claim that the original author intended to misrepresent results.

## Findings and remediation

| Severity | Original finding | Consequence | Version 2 remediation |
|---|---|---|---|
| Critical | `Inference_Quora_Contest.py` read `sample_submission.csv` and treated `is_duplicate` placeholders as observed labels. | Reported “accuracy” was not accuracy against hidden test labels. | `qqdup predict` consumes only question pairs and writes probabilities. Metrics require a separate labeled file. |
| Critical | Threshold search used `score >= threshold`; inference used `score <= threshold`. | The positive class was reversed at deployment. | The manifest records `probability >= threshold`, and tests enforce it. |
| High | SBERT cosine and LightGBM probability were manually averaged. | Different score scales made the weights difficult to interpret or calibrate. | A logistic fusion/calibration layer is fitted on validation component scores. |
| High | Ensemble weights and thresholds were selected without a single, protected final test protocol. | Model-selection optimism could leak into reported performance. | Train fits the lexical model, validation fits fusion/selects threshold, and test is evaluated once afterward. |
| High | Four semantic objectives were mixed through the legacy multi-DataLoader `fit` workflow without a held-out evaluator. | The objective was difficult to justify and semantic generalization was not measured during training. | Optional semantic training uses `SentenceTransformerTrainer`, one CoSENT objective, and `BinaryClassificationEvaluator` on held-out data. |
| High | Random triplet negatives were unseeded and could accidentally be semantically related. | Training was not reproducible and labels could be noisy. | The random triplet path was removed. All supported split/model seeds are explicit. |
| High | Clean clones depended on ignored `.pkl` files and local dataset paths. | The documented workflow could not run from source alone. | An offline synthetic demo, versioned config, CLI, and explicit artifact production workflow were added. |
| High | Pickle/joblib model loading accepted executable Python object graphs. | Loading an untrusted artifact could execute code. | Artifacts are JSON plus numeric NPZ arrays loaded with `allow_pickle=False` and checked against manifest hashes. |
| Medium | The PEFT model card was an unedited generated template with unresolved sections. | The checkpoint had no defensible provenance, evaluation, or intended-use record. | The legacy checkpoint was removed. A release-blocking model card template now lists required evidence. |
| Medium | NLTK WordNet was downloaded during the image build and used by preprocessing. | Builds required an external mutable resource; missing corpora caused runtime failures. | Unicode normalization and regex tokenization require no downloaded corpora. |
| Medium | Empty strings and empty token unions could divide by zero. | Valid rows with missing questions could crash feature extraction or produce NaN. | Every ratio/overlap has explicit empty-input behavior, and tests assert finite values. |
| Medium | The original Docker image ran as root and mounted the repository read/write. | A compromised process had broader filesystem access than necessary. | The image uses UID 10001; Compose drops capabilities, prevents privilege escalation, and uses a read-only root filesystem. |
| Medium | Broad `except:` clauses hid missing/corrupt tracker state. | Operational errors silently changed behavior. | Tracker files were removed; validation errors are explicit. |
| Medium | Two process pools were opened sequentially with `apply`, not concurrently. | Complexity and process overhead without the claimed parallelism. | Batch inference is a direct, deterministic pipeline; semantic encoding handles its own batching. |
| Medium | README listed ROC-AUC and accuracy as competition evaluation metrics. | It obscured the distinction between local diagnostics and probability submission scoring. | Evaluation documentation prioritizes log loss/calibration for probabilities and clearly scopes every local metric. |
| Low | ROC/AUC and threshold implementations duplicated established metric code. | Edge cases and numerical behavior were harder to trust. | Metrics use tested scikit-learn implementations. |
| Low | No automated tests, CI, dependency review, or static security checks. | Regressions were easy to introduce unnoticed. | Pytest, Ruff, coverage, Docker smoke tests, Bandit, pip-audit, Dependabot, and dependency review are configured. |

## Replacement architecture review

### Data boundary

`data.py` validates required columns, normalizes nulls, checks binary labels, and
creates deterministic stratified partitions. A SHA-256 digest of stable pandas
row hashes identifies the exact normalized labeled input used for a run.

### Lexical boundary

`features.py` implements 14 symmetric features. The TF-IDF vocabulary is learned
on the training partition only. `lexical.py` scales the features and fits a
regularized logistic model. Runtime probability calculation uses saved numeric
parameters rather than deserializing a Python estimator.

### Fusion and decision boundary

`ensemble.py` fits a logistic layer on validation component scores. With no
semantic component this acts as a calibration layer; with semantics it learns
how to map lexical probability and cosine similarity to one probability scale.
The binary threshold is selected on validation and applied with `>=`.

### Evaluation boundary

`pipeline.py` does not inspect the held-out test partition until training,
fusion fitting, and threshold selection are complete. `evaluation.json` labels
validation metrics as model-selection metrics and test metrics as the final
local held-out estimate.

### Serialization boundary

`artifacts.py` accepts only the expected schema, exact array names, finite
parameters, contiguous vocabulary indices, expected feature names, and verified
file hashes. NPZ parsing explicitly disables pickle.

## Residual risks and honest limitations

1. The default split is pair-stratified, not question-disjoint. Repeated
   questions can appear across partitions. This is documented and should be
   changed to a grouped protocol for publication-grade generalization claims.
2. Fitting fusion and selecting a threshold on the same validation partition is
   appropriate for a final held-out estimate but does not give an unbiased
   estimate of validation performance. Only the untouched test report is used
   as the local final estimate.
3. SHA-256 checks in a writable manifest provide corruption detection, not
   cryptographic publisher identity. Signed releases are recommended.
4. Dependency pins improve repeatability but do not make CPU/GPU floating-point
   training bit-for-bit identical across platforms.
5. Optional Hugging Face model downloads depend on upstream availability and
   model licensing. Cache and record the exact model revision for a formal
   release.
6. The lexical baseline report was produced from the full labeled dataset and a
   protected local test split, but it remains a release candidate until the
   report is bound to the final public Git SHA. It is pair-stratified and must
   not be described as hidden competition performance.

## Verification evidence

The repository includes tests for:

- Unicode, null, and label validation
- deterministic and complete splitting
- finite and symmetric feature extraction
- empty-corpus TF-IDF fallback
- learned mixed-scale fusion
- positive threshold direction
- safe artifact round trips and tamper detection
- end-to-end training and prediction

CI runs these tests across supported Python versions and smoke-tests the Docker
image. The security workflow audits core dependencies and scans source code.
