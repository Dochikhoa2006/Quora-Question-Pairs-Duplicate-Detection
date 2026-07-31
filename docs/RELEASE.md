# Release guide

## Versioning

The package uses semantic versioning:

- **Patch** — compatible fixes, documentation, or test improvements
- **Minor** — backward-compatible features or optional components
- **Major** — artifact schema, CLI, feature semantics, or decision-contract
  changes that require migration

Keep `pyproject.toml`, `src/quora_duplicate_detection/__init__.py`,
`CITATION.cff`, and `CHANGELOG.md` aligned.

## Release gates

Before tagging:

- [ ] Pull request reviewed and all CI/security jobs pass
- [ ] `make lint`, `make test`, `make security`, and `make docker` pass
- [ ] A clean-clone synthetic demo passes
- [ ] Package wheel builds and installs in a fresh environment
- [ ] Changelog has a dated version section
- [ ] Documentation matches CLI help and artifact schema
- [ ] Core and optional semantic dependencies were audited
- [ ] No dataset, secrets, local paths, generated model, or pickle files are tracked
- [ ] Real metric claims match an immutable `evaluation.json`
- [ ] Released weights have a completed model card
- [ ] License and upstream dataset/model terms were reviewed

## Package verification

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python -m venv /tmp/qqdup-wheel-check
/tmp/qqdup-wheel-check/bin/python -m pip install dist/*.whl
/tmp/qqdup-wheel-check/bin/qqdup --help
```

Use a reviewed temporary path appropriate to the operating system.

## Git release

After explicit owner approval:

```bash
git switch main
git pull --ff-only
git status --short
git tag -s v2.0.0 -m "Quora duplicate detection v2.0.0"
git push origin main
git push origin v2.0.0
```

Do not use a force-push. A signed tag requires configured signing keys; if none
exist, establish and document signing before the release rather than silently
creating an unsigned tag.

## GitHub release contents

Include:

- concise user-facing changes from the changelog
- supported Python versions
- source archive and built wheel
- SHA-256 checksums
- evaluation report and completed model card when publishing a model
- software bill of materials or resolved dependency record
- migration notes and known limitations

Do not attach raw datasets, private examples, local caches, or legacy pickle
artifacts.

## Repository settings

Recommended GitHub settings:

- protect `main` and require pull requests
- require CI, security, and dependency-review checks
- block force-pushes and branch deletion
- enable Dependabot alerts and security updates
- enable secret scanning and push protection when available
- enable private vulnerability reporting
- use squash or rebase merge consistently
- create a `v2.0.0` release only after the public tree matches the reviewed local
  replacement
