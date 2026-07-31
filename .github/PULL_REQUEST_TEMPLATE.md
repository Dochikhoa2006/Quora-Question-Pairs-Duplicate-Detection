## Summary

Describe the problem and the smallest useful solution.

## Change type

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Documentation
- [ ] CI, security, or dependency update
- [ ] Model or evaluation change

## Validation

- [ ] `make lint`
- [ ] `make test`
- [ ] Relevant end-to-end or semantic checks
- [ ] Documentation and changelog updated

List commands and material results:

```text
<paste concise validation summary>
```

## ML evaluation checklist

- [ ] No competition placeholder was treated as a label
- [ ] Test data was not used for model selection
- [ ] Split protocol, seed, and dataset fingerprint are recorded
- [ ] Threshold direction remains `probability >= threshold`
- [ ] Metric claims identify their labeled partition
- [ ] Model card updated if weights or claims changed
- [ ] Not applicable

## Security and privacy

- [ ] No secrets, private data, generated artifacts, or pickle files added
- [ ] New dependencies are pinned and audited
- [ ] Artifact/input boundary changes include negative tests

## Breaking changes and migration

`<none, or describe user action required>`
