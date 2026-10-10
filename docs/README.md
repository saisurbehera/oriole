# Oriole project guide

Oriole now has a maintained local NetHack runner alongside older knowledge-graph
and agent research. Start with the [root README](../README.md) to install and play.

- [Gameplay](gameplay.md): commands, prompts, persistent sessions, and wiki lookup.
- [Model trials](model-trials.md): measured results, setup, limits, and published musing.
- [Credential checks](credential-checks.md): history scan, exclusions, and CI guard.
- [Historical archive](../old/README.md): older projects and their execution paths.
- [Repository layout](repository-layout.md): maintained code, historical experiments,
  vendored dependencies, and files deliberately kept local.
- [Vendor provenance](vendor-provenance.json): upstream snapshot commit IDs.
- [Local artifact inventory](local-artifacts.json): historical large files that a
  fresh clone does not contain.
- [Codex instructions](../AGENTS.md): how Codex should play and report progress.

## Return to this experiment

The reference experiment is four concurrent five-minute runs on seed 43, medium
reasoning effort, using Codex CLI 0.162.1 and NLE 1.2.0. GPT-6.1 Sol reached floor 7;
Astra reached 6; Sol and Luna reached 2. All four were alive at the time limit.
None ascended. One short seed does not establish a general model ranking.

The [comparison report](../data/trials/comparison-2026-10-10.json) records the
setup, outcomes, and milestones. Archived screens and keys live in
`data/trials/2026-10-10/MODEL/`. After setup, replay them without a model call:

```sh
./play replay data/trials/2026-10-10/gpt-6.1-sol
./play replay data/trials/2026-10-10/gpt-6-astra --start 300 --frames 20
```

Read or share [Codex in the dungeon](https://www.saisur.me/musings/codex-in-the-dungeon).
The article and browser player are maintained in the separate
[website repository](https://github.com/saisurbehera/saisur.me).

## Continue the research

Use `./play bench` with multiple seeds and longer budgets, then inspect traces,
floor milestones, health, and deaths before comparing models. New trials save to
`.local/trials/`; they are ignored until selected for a documented archive. Model
availability and subscription limits can change, so retain the requested model,
CLI version, reasoning effort, seed, budgets, and errors with each experiment.
The original five-minute outcomes are observations, not a promise of ascension.
