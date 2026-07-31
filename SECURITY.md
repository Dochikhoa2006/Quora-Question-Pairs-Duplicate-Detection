# Security policy

## Supported version

Security fixes are applied to the current `main` branch. This research project is
not an online service and has no production availability commitment.

## Reporting

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting feature when it is enabled, or contact the repository
owner privately.

Include the affected version, reproduction steps, impact, and any suggested
mitigation. Do not include real Quora data or credentials.

## Security properties and limits

- Model artifacts use JSON and numeric NPZ arrays loaded with
  `numpy.load(..., allow_pickle=False)`. Legacy pickle/joblib artifacts are not
  accepted.
- Artifact files are checked against SHA-256 digests in the manifest. This
  detects accidental corruption but is not a digital signature and does not
  establish publisher authenticity.
- Dataset, output, checkpoint, environment, and credential files are ignored.
- The Docker image runs as an unprivileged user. The Compose service drops Linux
  capabilities, prevents privilege escalation, and uses a read-only root
  filesystem.
- CI runs dependency auditing, Bandit static analysis, dependency review, and
  CodeQL analysis.

Only load models and datasets from sources you trust. Hugging Face model loading
may retrieve executable custom code if a caller deliberately chooses a model
requiring it; this project never enables `trust_remote_code`.
