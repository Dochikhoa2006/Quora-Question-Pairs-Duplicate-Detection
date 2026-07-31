# ADR 0003: Keep semantic training optional

- Status: Accepted
- Date: 2026-07-31

## Context

Transformer training brings large downloads, platform-specific PyTorch wheels,
accelerator requirements, and longer CI/runtime costs. A portable lexical model
is still valuable for tests, demonstrations, and constrained environments.

## Decision

The core install contains NumPy, pandas, SciPy, and scikit-learn. Sentence
Transformers, datasets, PEFT, Accelerate, and PyTorch are a pinned optional
dependency set. Imports occur inside semantic runtime boundaries. Hybrid
artifacts explicitly declare their semantic component and model reference.

## Consequences

- The clean-clone demo and normal CI stay lightweight.
- Semantic training requires an explicit installation step.
- Core unit coverage excludes GPU/model-download execution paths; those require
  release-specific integration evidence.
- Hybrid inference fails clearly if its semantic model is unavailable.
- Dependency audits cover both core and semantic requirement files.
