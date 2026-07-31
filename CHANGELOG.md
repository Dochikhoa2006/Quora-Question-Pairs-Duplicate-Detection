# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org/) and the structure recommended by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Planned

- Question-disjoint evaluation as an alternative to pair-stratified splitting
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
