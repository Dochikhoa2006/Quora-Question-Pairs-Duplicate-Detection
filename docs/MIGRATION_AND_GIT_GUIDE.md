# Migration and Git push guide

This guide prepares the replacement for review. It does not authorize an
automated push, force-push, repository deletion, or LinkedIn publication.

## 1. Preserve a recovery point

Before staging the migration, inspect the current branch and create a local
backup reference:

```bash
git status --short
git branch backup/pre-v2-migration
```

Do not run `git reset --hard`, `git clean`, or a history rewrite as part of the
normal migration.

## 2. Understand removed legacy files

Version 2 removes the old top-level training/inference scripts, ROC image, and
committed PEFT checkpoint. They remain recoverable from the backup branch and
Git history. Ignored local datasets and `.pkl` files are not deleted by this
migration, but the new code never loads them.

Preview ignored files before deciding whether to remove local leftovers:

```bash
git status --ignored --short
git clean -ndX
```

`git clean -ndX` is preview-only. Do not remove anything until each path has
been reviewed and backed up if needed.

## 3. Reproduce from source

Use a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install-dev
make lint
make test
make demo
qqdup train \
  --data data/demo_train.csv \
  --output artifacts/demo \
  --config config/default.json
```

Also validate Docker:

```bash
make docker
```

The synthetic demo proves the software path works. It is not evidence of model
quality.

## 4. Complete release evidence

Before public performance claims:

- Train on the accepted dataset under its terms.
- Review `evaluation.json` and confirm the test split was untouched.
- Complete `models/MODEL_CARD_TEMPLATE.md`; save the completed card beside the
  released model or under a versioned filename.
- Record the Git SHA, dataset fingerprint, environment, base model revision, and
  artifact hashes.
- Decide whether CC BY 4.0 is the intended license for code as well as
  documentation.
- Review dataset and pretrained-model licenses independently.

## 5. Review secrets and large files

Inspect exactly what would be committed:

```bash
git status --short
git diff --check
git diff --stat
git diff
git ls-files | sort
```

Search tracked text for common credential markers:

```bash
git grep -nE '(API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE KEY|PASSWORD=)'
```

Confirm no datasets, environment files, generated artifacts, legacy pickle
objects, or local checkpoints are staged:

```bash
git status --short --ignored
```

If a real credential was ever committed, removing it in the newest commit is
not enough. Revoke it first, then use a reviewed history-rewrite procedure and
coordinate the forced update with every collaborator.

## 6. Stage deliberately

Avoid an indiscriminate `git add .`. Review and stage the intended paths:

```bash
git add \
  .dockerignore .env.example .gitattributes .github .gitignore \
  CHANGELOG.md CITATION.cff CODE_OF_CONDUCT.md CONTRIBUTING.md ROADMAP.md \
  Dockerfile LICENSE Makefile README.md SECURITY.md \
  config docker-compose.yml docs models reports pyproject.toml \
  requirements.txt requirements-dev.txt requirements-semantic.txt \
  scripts src tests
git status --short
git diff --cached --check
git diff --cached --stat
```

The deleted legacy tracked files must also appear in the staged review. If they
do not, stage their deletions explicitly with `git add -u`.

Commit only after tests and the staged diff pass review:

```bash
git commit -m "Rebuild duplicate detection pipeline with valid evaluation"
```

## 7. Verify a clean clone

The strongest reproducibility check is a fresh clone in a separate directory:

```bash
git clone --no-local /path/to/repository /tmp/qqdup-release-check
cd /tmp/qqdup-release-check
python3 -m venv .venv
source .venv/bin/activate
make install-dev
make lint
make test
make train-demo
```

Use a user-chosen temporary directory and remove it only after reviewing the
path.

## 8. Push without rewriting history

Confirm the intended remote and branch:

```bash
git remote -v
git branch --show-current
git log -1 --oneline
```

Then, only when the owner explicitly approves publication:

```bash
git push --set-upstream origin <reviewed-branch-name>
```

Open a pull request and let CI/security checks complete. Prefer merging the
reviewed pull request over pushing directly to a protected `main` branch. Do not
use `--force` for this migration.

## 9. LinkedIn release checklist

LinkedIn is a place to link to the reviewed GitHub repository, not a Git remote.
Before posting:

- Link a tagged or commit-pinned repository release.
- State that hidden competition labels were not used.
- Name the split protocol beside every metric.
- Distinguish synthetic demo results from real held-out evaluation.
- Avoid claiming production readiness.
- Mention the optional nature and upstream license of the semantic model.
- Include one architecture image or concise diagram, not the obsolete ROC plot.

A defensible summary is: “I rebuilt a duplicate-question classifier around
reproducible held-out evaluation, calibrated lexical/semantic fusion, safe
artifacts, tests, CI, Docker, and explicit limitations.”
