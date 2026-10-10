# Credential checks

Audited on 2026-10-10 with Gitleaks 8.30.1. Both the complete tracked-file snapshot
and all reachable Git history were scanned. No credentials were found. Two default
scanner matches were reviewed as false positives: tokenizer function arguments in
Prime RL and Windows key-handling enum names in NetHack. `.gitleaks.toml` permits
only those exact code patterns, while retaining all default detection rules.

Local `SECRETS`, `.env*`, Codex auth files, private key files, session logs, and
machine settings are ignored. The historical BALROG `SECRETS` file is local and
was not in either published commit. New run logs stay under `.local/`. The selected
tracked trial archives contain game observations only.

## Repeat the checks

With Gitleaks installed, run from the repository root:

```sh
gitleaks git --redact=100 --no-banner --log-opts="--all" .
```

For a tracked-file check, export the commit into a temporary directory and scan
it using the root `.gitleaks.toml`; avoid scanning local credential files into a
public report. CI performs both checks with a pinned, checksum-verified binary.
All diagnostics redact detected values. Ignored files can still be forcibly
staged, so review `git diff --cached --name-only` before committing.

These are detection checks, not a guarantee that arbitrary secrets can never be
missed. If a real credential is discovered in a published commit, revoke it at its
provider first, then remove it from current files and affected Git history. Merely
moving a secret into `old/` or adding an ignore rule does not remove it from history.
No history rewrite or credential revocation was needed for this audit.
