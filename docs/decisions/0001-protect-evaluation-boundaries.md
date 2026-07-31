# ADR 0001: Protect evaluation boundaries

- Status: Accepted
- Date: 2026-07-31

## Context

The legacy inference path treated sample-submission placeholders as observed
labels. Weight and threshold selection also lacked a clearly protected final
local test stage.

## Decision

Labeled data is divided into training, validation, and test partitions. Training
fits the lexical estimator. Validation fits probability fusion/calibration and
selects a threshold. Test is evaluated only after those choices. Competition
test input is prediction-only and produces probabilities without metrics.

## Consequences

- Local held-out metrics have a clear meaning.
- Validation metrics are explicitly model-selection metrics.
- Competition accuracy cannot be claimed without provider labels.
- A future grouped splitter can improve leakage control without changing roles.
- The model is not refit on all labeled rows after evaluation, because doing so
  would require a separate production-fit artifact and documented tradeoff.
