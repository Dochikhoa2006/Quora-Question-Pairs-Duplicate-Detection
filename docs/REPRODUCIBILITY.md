# Reproducibility guide

## Reproducibility levels

This project distinguishes three levels:

| Level | Guarantee |
|---|---|
| Software | A clean clone can install, test, build, and run the synthetic demo |
| Experimental | The same normalized dataset, config, and seed recreate the same split and materially equivalent predictions |
| Bitwise | Artifact bytes match across runs and hardware |

Version 2 targets software and experimental reproducibility. It does not claim
bitwise equality across CPU/GPU architectures or numeric libraries.

## Clean environment

```bash
git clone https://github.com/Dochikhoa2006/Quora-Question-Pairs-Duplicate-Detection.git
cd Quora-Question-Pairs-Duplicate-Detection
python3 -m venv .venv
source .venv/bin/activate
make install-dev
make lint
make test
make train-demo
```

Record:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
uname -a
```

Use the Docker image when host-level differences matter.

## Dataset identity

The training manifest records a SHA-256 digest derived from stable pandas row
hashes over normalized `question1`, `question2`, and `is_duplicate` values. It
also records row count and exact split assignments.

The fingerprint is useful for comparing local runs; it is not a replacement for
the dataset provider's versioning or license record. Do not commit the dataset
to make a fingerprint reproducible.

## Randomness

`random_seed` controls stratified splitting and logistic estimators. Semantic
training also seeds NumPy and PyTorch and forwards the seed to trainer/data
arguments.

GPU kernels, parallel math libraries, upstream tokenizer changes, and hardware
can still produce numeric differences. A formal semantic release should record:

- GPU/accelerator model and driver
- CUDA, PyTorch, and transformer versions
- precision mode
- base model repository and immutable revision
- training duration and batch/accumulation settings
- all warnings and evaluator output

`qqdup train-semantic` resolves a Hub base model to its immutable commit before
loading it and records the revision in `training_summary.json`. For a local
base model directory it records a SHA-256 over file paths and contents. The
record identifies the base model input; it does not make GPU training bitwise
deterministic.
Semantic training also stores a dataset fingerprint and all resume-critical
settings in `run_config.json` before fitting. `--resume` checks these values and
the base model provenance, then resumes from the newest complete checkpoint.
The final summary includes aggregate training metrics and step history.
It also lists SHA-256 checksums for all files under `model/`, with separate
adapter checksums for LoRA runs. Run `qqdup verify-semantic --artifact <run>`
to detect changed, missing, or extra model files. The summary is stored beside
the model and is not signed, so these checksums establish integrity only when
the summary itself comes from a trusted source.

For a signed record, install `requirements-signing.txt` and generate an Ed25519
key pair outside the artifact directory:

```bash
umask 077
openssl genpkey -algorithm Ed25519 -out private.pem
openssl pkey -in private.pem -pubout -out public.pem
qqdup sign-semantic --artifact artifacts/semantic-cosent --private-key private.pem
qqdup verify-semantic --artifact artifacts/semantic-cosent --public-key public.pem
```

`semantic_provenance.json` signs SHA-256 digests of `run_config.json` and
`training_summary.json`. The signed summary binds the saved model and adapter
files through their checksums. Verification requires a trusted public key;
the key fingerprint inside the artifact alone does not establish ownership.
Store and distribute the public key independently, never ship the private key
with the artifact, and sign only after the training run is final. A resumed run
must be signed again after removing any stale provenance file.

## Configuration provenance

Start from `config/default.json` and commit any experiment configuration under a
descriptive path. Do not change defaults solely to reproduce a historical run;
preserve the historical file and create a new one.

The artifact embeds the effective configuration. Compare it with:

```bash
qqdup inspect --artifact artifacts/<release>
```

## Dependency provenance

Direct dependencies are pinned. CI and Dependabot continuously test updated
environments, while pip-audit checks core and semantic sets. Pins reduce drift
but do not form a complete cross-platform lock of every wheel and transitive
dependency.

For a formal release:

1. Export the resolved environment for the target platform.
2. Record hashes or an SBOM alongside the release.
3. Build inside the documented container.
4. Record the container image digest.
5. Audit the resolved environment immediately before release.

## Re-running evaluation

Use a new artifact directory for every run; training refuses to overwrite a
non-empty directory.

```bash
qqdup train \
  --data data/train.csv \
  --output artifacts/run-YYYYMMDD-seed42 \
  --config config/default.json
```

Compare:

- manifest dataset fingerprint and configuration
- `split_assignments.csv`
- model component names and feature schema
- threshold and validation-selection score
- held-out test probability metrics

Audit an artifact against the exact original labeled CSV:

```bash
qqdup audit-split \
  --data data/train.csv \
  --artifact artifacts/<release> \
  --output outputs/split_audit.json
```

The command verifies the dataset fingerprint, complete row assignments,
recorded partition sizes, class coverage, and normalized-question overlap.
It also recomputes validation and test metrics with the saved model, verifies
the selected threshold, and detects a report that disagrees with the original
data even when its file hash has been updated. For a hybrid artifact, pass
`--semantic-model` with the exact model used during training; otherwise the
evaluation replay check is marked `unavailable_semantic`.
New artifacts also record a split replay version. The audit regenerates the
assignments from the saved configuration and seed, then counts mismatched rows.
Older artifacts without that version receive `unavailable_legacy` replay status;
their other checks still run. A future change to a split algorithm must advance
the replay version and preserve a replay path for prior versions.
The grouped splitter scores candidates from component row and positive counts,
then expands only the selected candidate to row assignments. Regression tests
check that this produces the same assignments as the original row-level search
for the current replay version.
Question overlap fails a question-disjoint audit; it is reported but allowed
for a pair-stratified artifact. The JSON contains aggregate counts only. A
failed check exits with status 1; a different dataset or malformed assignment
file raises an error.

Do not compare demo metrics to real-data metrics or validation metrics to final
test metrics.
