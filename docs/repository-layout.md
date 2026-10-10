# Repository organization

The repository originally tracked only README.md and two submodule pointers.
Most Oriole experiments, custom BALROG agents, datasets, and dependency edits were
untracked. The organized repository tracks source files and small derived data
under the parent repository, so clones contain the actual implementation.

## Dependencies

`nle/`, `ultra/`, and `BALROG/` are source snapshots rather than Git submodules.
Their existing local changes are included. Previously nested source projects
inside NLE and BALROG are included too, apart from redundant duplicate checkouts
(`nle/ultra/` and `BALROG/BALROG/`). The duplicate directories remain on disk.
Upstream commit IDs are recorded in `vendor-provenance.json`. The broken absolute AutoAscend glyph mapping link was replaced with an import
from NLE so the source snapshot is portable. Upstream licenses
remain in their respective directories; consult them before distributing code or
wiki-derived material. Existing Git metadata is preserved locally in
`.local/git-backups/`; it is not included in a clone. The parent Git index and
README from before organization are also backed up there.

The new gameplay environment installs published NLE 1.2.0. Source snapshots are
kept for the existing research work and are not implicitly used by `scripts/setup.sh`.
The game uses the standard NetHackScore task with the full keyboard and interactive
prompts enabled; it supports old Gym and new Gymnasium reset/step result formats.

## Historical experiments

Root Python experiments moved together into `research/` so sibling imports still
resolve. Run them from the repository root to preserve relative paths to `data/`
and `ultra/`. Shell scripts moved into `scripts/legacy/` and references to root
Python scripts were updated. These scripts remain historical experiments, not a
fully verified research suite. Installing their old CUDA stacks is unnecessary for
local gameplay. The old API can still be run with
`python research/nethack_qa_server.py` in a suitable research environment; it needs
its separate vLLM service and server requirements.

## What belongs in Git

Source code, configuration, documentation, notebooks, and small derived graph
files are tracked. Local credentials (including BALROG/SECRETS), virtual
environments, compiled output, caches, new gameplay logs, model weights, raw wiki data,
trajectory archives, training datasets, and generated run directories are ignored.
No local artifact was deleted as part of organization.

`local-artifacts.json` inventories the existing files larger than 5 MB, which stay
on this machine and are explicitly excluded. This is an inventory, not a download
manifest: a fresh clone does not recover those historical raw datasets or trained
checkpoints. The maintained gameplay path only needs the included final wiki graph
and the packages installed by `scripts/setup.sh`.

## Codex

`AGENTS.md` documents the commands and rules for playing through local shell tools.
`scripts/play-codex.sh` opens an interactive Codex CLI session using the existing
login. It does not extract authentication tokens or replace the old API agent with
an unofficial endpoint. No API key is required by the gameplay code.

Official setup references:
- [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)
- [Codex authentication](https://learn.chatgpt.com/docs/auth)

A server process owns the live game, accessed over a local Unix socket restricted
to the current user. There are no network listeners or inference calls in the
local game bridge. Stopping it loses the live episode. Resetting starts over and
replaces the latest episode transcript. Tests use their own transcript directories.

## Recorded reference experiment

The four paired-seed games from the published musing are archived under
`data/trials/2026-10-10/MODEL/`. Each has `episode.jsonl` and `milestones.json`.
These are selected game observations, not Codex event logs or account data.
`python scripts/archive-trials.py` rebuilds the archive from the original local
logs while they remain available. Fresh clones can replay the archive without
those local logs. See [the project guide](README.md) for the documentation map.
