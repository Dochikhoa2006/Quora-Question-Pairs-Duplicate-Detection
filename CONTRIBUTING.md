# Contributing

Thank you for improving the project. Contributions should preserve its core
contract: measured claims must come from labeled data under a disclosed
evaluation protocol.

## Development setup

```bash
git clone https://github.com/Dochikhoa2006/Quora-Question-Pairs-Duplicate-Detection.git
cd Quora-Question-Pairs-Duplicate-Detection
python3 -m venv .venv
source .venv/bin/activate
make install-dev
make lint
make test
```

Use `make train-demo` for an offline end-to-end check. The synthetic demo is not
a benchmark.

## Change workflow

1. Open an issue for material behavioral or architectural changes.
2. Create a focused branch from the latest `main`.
3. Add or update tests with the implementation.
4. Run lint, tests, dependency audits, and the relevant demo locally.
5. Update user-facing documentation and `CHANGELOG.md`.
6. Open a pull request using the repository template.

Prefer small commits with imperative messages, for example:

```text
Add question-disjoint split strategy
Validate semantic model revision metadata
```

## Engineering standards

- Keep imports of semantic dependencies lazy so the lexical baseline remains
  lightweight.
- Preserve feature symmetry unless a design decision explicitly changes it.
- Treat the held-out test partition as read-once evaluation data.
- Never calculate label-dependent metrics from competition test files.
- Never add pickle/joblib loading to the trusted artifact path.
- Validate inputs at system boundaries and include actionable error messages.
- Keep random seeds and material hyperparameters in versioned configuration.
- Do not add raw datasets, credentials, generated artifacts, or user text.

## Tests

Every bug fix should include a regression test. New features should cover normal
behavior and failure boundaries. Core coverage must remain above the configured
threshold.

Semantic training requires large model downloads and appropriate hardware, so
normal CI checks its integration boundary rather than running a full fine-tune.
Include the exact hardware, model revision, dataset fingerprint, and evaluation
report when proposing semantic training changes.

## Documentation and model claims

Performance claims must identify:

- the labeled dataset and fingerprint
- the exact split protocol
- whether results are validation, local held-out test, or leaderboard results
- the Git commit and artifact version
- uncertainty and known leakage limitations

Complete a copy of `models/MODEL_CARD_TEMPLATE.md` before releasing weights.

## Security and privacy

Follow `SECURITY.md` for vulnerability reporting. Do not attach real Quora rows,
credentials, model tokens, private URLs, or generated checkpoints to issues or
pull requests.

## Review expectations

Maintainers may request changes for correctness, reproducibility, privacy,
security, documentation, or scope. By contributing, you agree that your work is
provided under the repository license.
