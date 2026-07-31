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

Do not compare demo metrics to real-data metrics or validation metrics to final
test metrics.
