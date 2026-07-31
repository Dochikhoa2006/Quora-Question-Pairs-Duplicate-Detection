# Roadmap

The roadmap is outcome-oriented. It does not promise dates or benchmark values.
Priorities can change when evidence or user needs change.

## Version 2.0 — evaluation and engineering foundation

- [x] Correct unlabeled competition inference
- [x] Deterministic train/validation/test roles
- [x] Consistent positive-class threshold semantics
- [x] Learned score calibration/fusion
- [x] Null-safe, corpus-free lexical preprocessing
- [x] Safe JSON/NPZ artifact schema
- [x] Modern optional CoSENT/LoRA training boundary
- [x] Tests, CI, Docker, dependency audits, and documentation

## Next — stronger evaluation science

- [ ] Add question-disjoint splitting using stable question IDs or components
- [ ] Add repeated grouped evaluation with uncertainty summaries
- [ ] Add calibration curves and expected calibration error
- [ ] Add privacy-safe slice/error analysis tooling
- [ ] Publish one immutable aggregate benchmark report and completed model card

Acceptance criteria: the grouped protocol has leakage tests, the benchmark is
recreated from a tagged commit, and every public metric links to release
evidence.

## Later — semantic experiment maturity

- [ ] Record immutable upstream model revisions automatically
- [ ] Add resumable semantic training and structured trainer telemetry
- [ ] Compare lexical, frozen semantic, CoSENT, and CoSENT+LoRA under one protocol
- [ ] Add a hardware-aware semantic integration test outside normal core CI
- [ ] Publish adapter checksums and signed provenance

Acceptance criteria: comparisons use identical splits and calibration rules;
cost, latency, uncertainty, and limitations are reported beside quality metrics.

## Optional productization

- [ ] Add a thin batch/API layer without duplicating package logic
- [ ] Define request-size, timeout, rate, and observability boundaries
- [ ] Add drift/calibration monitoring guidance
- [ ] Add signed containers and software bills of materials

Productization is intentionally secondary. It should proceed only for a concrete
use case with reviewed privacy, safety, and operational requirements.

## Non-goals

- Claiming accuracy for hidden competition labels
- Optimizing against repeated leaderboard submissions and calling it held-out
- Shipping raw data or user questions with the repository
- Accepting legacy pickle/joblib artifacts for convenience
- Presenting the synthetic demo as evidence of model performance
