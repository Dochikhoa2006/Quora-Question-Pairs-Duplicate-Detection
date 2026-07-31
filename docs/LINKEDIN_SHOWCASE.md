# LinkedIn and GitHub showcase guide

## Positioning

The strongest story is not “I trained another classifier.” It is:

> I audited an experimental NLP repository, found evaluation and deployment
> inconsistencies, and rebuilt it as a reproducible ML system with calibrated
> fusion, safe artifacts, modern semantic training, automated quality gates,
> and explicit limitations.

That framing demonstrates ML judgment, software engineering, security, and
technical communication without relying on inflated benchmark claims.

## Recommended GitHub About metadata

**Description**

```text
Reproducible duplicate-question detection with safe lexical features, optional CoSENT/LoRA embeddings, calibrated fusion, tests, CI, and Docker.
```

**Website**

Use the eventual LinkedIn project/post URL or a tagged GitHub release. Do not use
an unrelated personal link.

**Topics**

```text
natural-language-processing
duplicate-question-detection
sentence-transformers
cosent
lora
peft
probability-calibration
feature-engineering
scikit-learn
machine-learning
reproducible-research
mlops
pytest
github-actions
docker
model-card
```

Remove legacy topics that describe code no longer present, such as BM25,
LightGBM, multi-loss training, or the old ROC grid search.

## Repository launch checklist

- [ ] Public `main` matches the reviewed v2 tree
- [ ] CI and security badges are green
- [ ] Repository description and topics match version 2
- [ ] A tagged release exists
- [ ] README links resolve from GitHub
- [ ] No generated model card placeholders are presented as a released model
- [ ] No benchmark number appears without a labeled split and report
- [ ] Social preview image, if added, uses the project name and architecture—not
      an obsolete result graph
- [ ] Contact details are intentional and current
- [ ] LinkedIn links to the tag/release, not an unreviewed moving branch

## Evidence-safe LinkedIn post template

Replace angle-bracket fields only with verified evidence:

```text
I rebuilt my Quora Question Pairs duplicate-detection project from an
experimental notebook-style repository into a reproducible ML system.

The most important work was not changing the model—it was correcting the
evaluation contract. The hidden competition test set is now prediction-only;
fusion and threshold selection happen on validation data; and a separate held-
out test split is evaluated once.

What I implemented:
• 14 null-safe, symmetric lexical features with character n-gram TF-IDF
• optional SentenceTransformerTrainer fine-tuning with CoSENT and LoRA
• learned lexical/semantic probability fusion instead of manual score averaging
• JSON/NPZ artifacts with schema and integrity validation—no pickle loading
• deterministic configuration, split records, model/evaluation metadata
• <TEST COUNT> automated tests with <COVERAGE>% core coverage
• CI, dependency auditing, static security checks, and hardened Docker defaults

Local held-out result: <METRIC, VALUE, SPLIT PROTOCOL>
Known limitation: the default split is pair-stratified, so I disclose possible
question overlap and avoid claiming hidden-test accuracy.

Repository: https://github.com/Dochikhoa2006/Quora-Question-Pairs-Duplicate-Detection

#MachineLearning #NLP #MLOps #Python #SentenceTransformers #Docker #GitHubActions
```

If no real benchmark release exists, delete the “Local held-out result” line.
Never substitute synthetic demo metrics.

### Verified facts for the version 2 launch

- 30 automated tests
- 90.3% measured core coverage in the local release check
- 404,290 labeled pairs in the full lexical baseline run
- 60,644-pair untouched local test partition
- ROC-AUC 0.7905, log loss 0.5300, and F1 0.6728 at a
  validation-selected 0.23 threshold
- pair-level stratified protocol with repeated-question leakage disclosed
- no observed-label metric computed from the hidden competition test set

Re-check the public CI run and bind the aggregate report to the final Git SHA
before copying these facts into a post.

## Suggested carousel or image sequence

1. **Problem:** duplicate intent is semantic and lexical
2. **Audit:** three critical failures—invalid labels, reversed threshold, mixed
   score scales
3. **Architecture:** lexical and optional semantic paths into learned fusion
4. **Evaluation:** train → validation/calibration → untouched local test →
   probability-only competition inference
5. **Engineering:** CLI, safe artifact, tests, CI, Docker, security
6. **Reflection:** leakage, calibration shift, data ethics, and next steps

Use aggregate diagrams and synthetic text. Do not place raw dataset questions,
credentials, or local file paths in screenshots.

## Five-minute recruiter walkthrough

1. Open the README and explain the corrected evaluation contract.
2. Show the architecture diagram and package boundaries.
3. Open `features.py` and its symmetry/empty-input tests.
4. Show `manifest.json` structure through the README artifact example.
5. Open CI/security workflows and the technical audit.
6. End with one limitation and the question-disjoint evaluation roadmap.

This sequence makes engineering decisions visible without requiring the viewer
to read every source file.
