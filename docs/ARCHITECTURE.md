# Architecture

## Design goals

The system is designed around six properties:

1. **Evaluation integrity** — unlabeled competition data never enters a metric.
2. **Reproducibility** — splits, configuration, and dataset identity are recorded.
3. **Portable baseline** — lexical training needs no GPU or downloaded NLP corpus.
4. **Optional semantics** — transformer dependencies stay outside the core path.
5. **Safe artifacts** — runtime loading accepts declared numeric arrays, not
   executable Python object graphs.
6. **Honest failure** — malformed data and incompatible artifacts fail with an
   actionable error instead of silently changing behavior.

## System context

```mermaid
flowchart LR
    A[Labeled question pairs] --> B[Validated normalized data]
    B --> C[Train split]
    B --> D[Validation split]
    B --> E[Held-out test split]

    C --> F[Lexical feature extractor]
    F --> G[Lexical logistic model]
    D --> H[Optional semantic encoder]
    D --> G
    G --> I[Lexical probabilities]
    H --> J[Semantic cosine scores]
    I --> K[Logistic fusion / calibration]
    J --> K
    K --> L[Validation threshold selection]

    E --> M[Final local evaluation]
    G --> M
    H --> M
    K --> M
    L --> M

    G --> N[Versioned artifact]
    K --> N
    L --> N
    M --> N

    O[Unlabeled competition pairs] --> P[Prediction pipeline]
    N --> P
    P --> Q[Probability submission]
```

The held-out test partition is downstream of every selection decision. The
competition path terminates in probabilities and has no label-dependent metric.

## Package boundaries

| Module | Responsibility | Must not do |
|---|---|---|
| `config.py` | Validate versioned hyperparameters | Read mutable global settings |
| `data.py` | Normalize, validate, fingerprint, and split pairs | Fit model state |
| `features.py` | Learn TF-IDF state and compute 14 symmetric features | Access labels |
| `lexical.py` | Fit and score the portable logistic baseline | Serialize Python estimators |
| `semantic.py` | Optional CoSENT/LoRA training and embedding scoring | Become a core import requirement |
| `ensemble.py` | Calibrate/fuse component scores and select thresholds | Inspect held-out test labels during selection |
| `metrics.py` | Compute labeled probability and decision metrics | Infer whether labels are genuine |
| `artifacts.py` | Save/load strict JSON and NPZ schemas with integrity checks | Load pickle or arbitrary modules |
| `pipeline.py` | Orchestrate lifecycle stages in the permitted order | Treat a submission template as observations |
| `cli.py` | Expose explicit user workflows | Hide material defaults or mutate remotes |

## Training sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Data
    participant Lexical
    participant Semantic
    participant Fusion
    participant Artifact

    User->>CLI: qqdup train --data ...
    CLI->>Data: validate, normalize, fingerprint
    Data-->>CLI: train / validation / test
    CLI->>Lexical: fit(train)
    Lexical-->>CLI: lexical model
    CLI->>Lexical: score(validation)
    opt hybrid run
        CLI->>Semantic: score(validation)
    end
    CLI->>Fusion: fit(validation scores, labels)
    Fusion-->>CLI: calibrated probability model + threshold
    CLI->>Lexical: score(test)
    opt hybrid run
        CLI->>Semantic: score(test)
    end
    CLI->>Fusion: score(test once)
    CLI->>Artifact: save model, report, config, splits, hashes
    Artifact-->>User: versioned artifact directory
```

## Feature pipeline

Normalization is intentionally conservative: Unicode NFKC, case folding, and
whitespace collapse. It does not silently remove punctuation or apply an
English-only lemma model.

The features cover four views:

- **Length:** character/token ratios and normalized differences
- **Token sets:** Dice, Jaccard, containment, numeric overlap, boundary agreement
- **Order/string:** sequence similarity, token-sort similarity, exact match
- **Corpus statistics:** character n-gram TF-IDF cosine

Every feature is symmetric under swapping `question1` and `question2`. Tests
enforce symmetry and finite values for missing and empty input.

## Probability and decision semantics

The lexical model produces a probability-like score. Optional semantic cosine
is on a different scale. Logistic stacking maps either one component or both
components to one calibrated probability space.

A threshold produces a binary decision only when requested:

```text
duplicate = probability >= threshold
```

Submission output remains probabilistic.

## Artifact boundary

```text
artifact/
├── manifest.json
├── model.npz
├── vocabulary.json
├── evaluation.json
└── split_assignments.csv
```

Loading checks the schema version, artifact type, exact feature list, exact NPZ
array names, array shapes, finite values, vocabulary indices, decision
direction, and SHA-256 file digests. This is corruption detection, not publisher
authentication; signed releases remain a future enhancement.

## Deployment model

The project is a batch CLI rather than a network service. This keeps the trust
boundary small and makes competition submissions and offline evaluation easy to
reproduce. A future API should wrap the package rather than duplicate feature or
artifact logic.

Docker runs the same CLI as an unprivileged user. Compose adds read-only root
storage, a temporary `/tmp`, dropped capabilities, and no-new-privileges.

## Extension points

- Add a question-disjoint splitter behind the same `DataSplits` contract.
- Add a new semantic scorer that returns one finite score per row.
- Add a fusion component by versioning the artifact schema and negative tests.
- Add calibration diagnostics without changing competition output columns.
- Add a service layer that calls `predict_probabilities` and enforces request
  size/rate boundaries.

Any schema or decision-semantics change requires an architecture decision record
and a major or minor version assessment.
