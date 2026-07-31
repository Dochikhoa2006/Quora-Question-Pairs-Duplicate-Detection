# Architecture decision records

Architecture decision records preserve why consequential choices were made.

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-protect-evaluation-boundaries.md) | Accepted | Protect train, validation, test, and competition roles |
| [0002](0002-use-json-npz-artifacts.md) | Accepted | Use strict JSON/NPZ artifacts instead of pickle |
| [0003](0003-keep-semantic-training-optional.md) | Accepted | Keep semantic dependencies behind an optional boundary |

Create a new numbered record when a change alters the evaluation contract,
artifact schema, dependency boundary, or public interface. Do not rewrite an
accepted decision to hide history; supersede it with a new record.
