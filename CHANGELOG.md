# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org/) and the structure recommended by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Regular predictions preserve text identifiers, including leading zeros
- Regular prediction output now replaces an existing CSV atomically
- Training count settings now require positive integers
- Removed a duplicate artifact checksum helper and restored CI formatting
- Training configuration validates random seeds before splitting or fitting
- Training configuration rejects non-finite logistic regularization values
- Evaluation clips probabilities only for log loss, preserving extreme scores in
  ranking, Brier, and threshold metrics
- External comparison reports now replace the destination atomically
- External artifact comparison rejects the same saved model under multiple labels
- Added external labeled-set comparison for saved detectors with paired
  stratified bootstrap metric differences and aggregate provenance
- Added optional Ed25519 provenance signing for completed semantic runs and
  trusted public key verification of metadata and model checksums
- Semantic training summaries now record SHA-256 checksums for saved model and
  LoRA adapter files; `qqdup verify-semantic` checks the saved artifact
- Added an opt-in offline semantic integration smoke test covering full and LoRA
  fine-tuning, checkpoint resume, scoring, and hardware/runtime reporting
- Semantic fine-tuning can resume from a complete checkpoint after verifying
  the dataset, base model, and settings; summaries include trainer telemetry
- Repeated grouped evaluation can compare hybrid and lexical models on identical
  seeded splits and report paired held-out metric differences
- Semantic fine-tuning now pins Hub base models to immutable revisions and
  records either the Hub commit or a local directory content digest
- `qqdup predict --chunk-size` streams large CSV inputs, reuses loaded models,
  and replaces the output only after all chunks complete
- Repeated grouped evaluation now reports test-row coverage and pairwise test-set
  overlap, making repeated holdout dependence visible in aggregate form
- `audit-split` now replays saved validation and test metrics through the serialized
  lexical model and checks the selected threshold; hybrid replay accepts a supplied
  semantic model
- Updated the optional semantic stack to patched releases and pinned `fsspec`
  so dependency audits cover its transitive security fix
- Question-disjoint split search now scores component summaries and expands only
  the selected candidate, preserving split replay version 1 assignments

### Planned

- A completed model card and benchmark report from a versioned full-data run
- Signed release artifacts and software-bill-of-materials publication

## [2.0.0] - 2026-07-31

### Added

- Installable `src/` package and `qqdup` command-line interface
- Null-safe, symmetric lexical feature pipeline with character n-gram TF-IDF
- Logistic probability calibration and optional lexical/semantic score fusion
- Deterministic train/validation/test evaluation with recorded split assignments
- Optional `SentenceTransformerTrainer` CoSENT fine-tuning and LoRA
- JSON/NPZ artifacts with strict schemas, SHA-256 integrity checks, and no pickle
- Automated tests, coverage, Ruff, Bandit, dependency auditing, and CI
- Non-root Docker image and hardened Compose defaults
- Technical audit, evaluation, migration, security, and model-card documentation
- Offline synthetic dataset generator for clean-clone smoke testing

### Changed

- Competition inference now writes probabilities without pretending placeholder
  submission values are labels
- Binary decisions consistently use `probability >= threshold`
- Manual weighted score averaging was replaced by learned fusion
- Language-specific NLTK preprocessing was replaced by deterministic Unicode
  normalization and regex tokenization

### Removed

- Legacy top-level experiment scripts
- Ignored pickle/joblib runtime dependencies
- Unverifiable PEFT checkpoint and incomplete generated model card
- Obsolete ROC image based on the legacy evaluation workflow

## Legacy implementation - Before 2026-07-31

### Added

- Original SBERT, LoRA, LightGBM, ensemble-search, and competition scripts

[Unreleased]: https://github.com/Dochikhoa2006/Quora-Question-Pairs-Duplicate-Detection/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Dochikhoa2006/Quora-Question-Pairs-Duplicate-Detection/releases/tag/v2.0.0
