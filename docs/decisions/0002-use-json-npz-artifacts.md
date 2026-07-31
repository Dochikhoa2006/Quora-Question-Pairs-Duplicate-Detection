# ADR 0002: Use strict JSON/NPZ artifacts

- Status: Accepted
- Date: 2026-07-31

## Context

Pickle and joblib serialize executable Python object graphs. The legacy clean
clone also depended on ignored pickle files with undocumented provenance.

## Decision

Persist declarative metadata and vocabulary as JSON and numeric parameters as a
fixed-name NPZ archive. Load NPZ with `allow_pickle=False`. Validate schema,
feature names, component names, vocabulary indices, parameter shapes and
finiteness, decision direction, and payload hashes.

## Consequences

- Untrusted Python constructors are not invoked during artifact loading.
- Artifacts are inspectable and less coupled to estimator internals.
- Runtime probability calculation is implemented from saved numeric parameters.
- Changes to array semantics require an explicit schema version migration.
- Manifest hashes detect corruption but do not authenticate the publisher;
  signed releases remain necessary for provenance.
