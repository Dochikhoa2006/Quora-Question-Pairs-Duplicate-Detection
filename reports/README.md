# Evaluation reports

This directory is reserved for small, aggregate, reviewable release reports.
Raw datasets, row-level predictions, model weights, checkpoints, and local paths
must not be committed.

A benchmark report may be added only when it includes:

- release/model version and Git commit
- normalized dataset fingerprint, without raw data
- split protocol, random seed, and partition sizes
- validation selection procedure
- untouched local test probability and threshold metrics
- leakage and uncertainty limitations
- link to a completed model card and artifact checksums

Synthetic demo output is a software smoke test and must never appear here as
benchmark evidence. Real-data numbers belong in a report satisfying the evidence
contract above.

## Available aggregate report

- [`lexical-v2-pair-stratified.json`](lexical-v2-pair-stratified.json) — full
  labeled-data lexical baseline, seed 42, with an untouched local test split

The report is explicitly marked as a release candidate until its source revision
is bound to the final public Git SHA. Its metrics are not competition test
metrics, and its pair-level overlap limitation must accompany any citation.
