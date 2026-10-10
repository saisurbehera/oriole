# Oriole

Oriole helps AI agents play NetHack using a local game environment and structured
knowledge extracted from the NetHack wiki. The maintained gameplay interface lets
Codex read the dungeon and take actions through local commands. It does not require
an API key, vLLM, or BALROG to play.

The project is named after the Oriole, a songbird associated here with learning
and navigation.

## Set up

On Linux, use Python 3.11 (recommended) and run:

```sh
./scripts/setup.sh
```

This installs the small gameplay dependency set in `.local/play-venv/`, separate
from existing research environments. It uses the published NLE 1.2.0 package;
vendored NLE source remains available for research and modifications. If pip needs
to build NLE from source, install CMake and the compiler/build dependencies listed
in `nle/README.md`. The local command server uses Unix sockets (Linux/macOS).

## Play yourself

```sh
./play reset --seed 42
./play human
```

Enter named actions such as `north`, `search`, or `open east`, or literal NetHack
keys such as `i`. Use `:quit` to leave human mode while keeping the game running.
Human mode attaches to the existing game and starts one if necessary.

## Let Codex play

Codex CLI is required for the launcher. Sign in with your ChatGPT account using
`codex login` if you have not already done so, then run:

```sh
./scripts/play-codex.sh
```

The launcher opens an interactive Codex session with a gameplay prompt. Codex reads
`AGENTS.md` and uses ordinary shell commands to play; model usage is governed by
your Codex account and plan. You can also open this repository in Codex and ask:

> Play NetHack through ./play. Follow AGENTS.md, consult local knowledge when useful,
> and keep playing until the episode ends or I stop you.

This sets up an attempt to play a full game, including ascension as the eventual
objective. It does not guarantee a win. A completed episode may mean death or a
step limit; the final game screen is the evidence for an actual ascension.

Useful commands:

```sh
./play observe                    # screen and current stats; no time passes
./play act north                  # move one square
./play act open east              # open a door to the east
./play act inventory              # inspect items
./play act escape                 # dismiss a menu
./play act '>'                    # descend on a staircase
./play knowledge 'soldier ant'    # sourced facts from the local wiki graph
./play --json observe             # machine readable output
./play actions                    # names and corresponding keys
./play status                     # persistent game process status
./play stop                       # close the process; live game is lost
```

The game persists across commands in a background process. State is not saved
across shutdown/reboot. `.local/episode.jsonl` records the latest episode, including
screens and actions; resetting replaces that transcript. The game log is
`.local/game-server.log`. Use short action sequences and inspect prompts before
sending more input.

## Repository layout

| Path | Contents |
| --- | --- |
| `oriole/` | Maintained game session, CLI, and local knowledge lookup |
| `play` | Launcher that selects the gameplay Python environment |
| `scripts/` | Setup and Codex launchers |
| `research/` | Historical wiki extraction, QA, training, analysis, and benchmarks |
| `scripts/legacy/` | Historical research installers and pipeline launchers |
| `data/` | Wiki-derived knowledge graphs and benchmark splits |
| `tests/` | Real NLE gameplay checks |
| `nle/` | Vendored NetHack Learning Environment source |
| `ultra/` | Vendored neural graph reasoning source, with local edits |
| `BALROG/` | Vendored game agent/evaluation source, with custom agents |
| `docs/` | Repository organization, dependency provenance, and artifact inventory |

Run research scripts from the repository root, for example:
`python research/process_full_wiki_to_a_plus.py`. Their dependencies are separate
from the lightweight gameplay setup. See [repository notes](docs/repository-layout.md).

## Verify

```sh
.local/play-venv/bin/python -m unittest discover -s tests -v
```

The tests check real game turns, non-mutating observation, invalid-action handling,
episode limits/reset, and sourced knowledge retrieval.

## Isolated trials and viewing

`./play trial --model MODEL` starts an isolated game using the subscription-only
runner. `./play bench --models MODEL_A MODEL_B` compares models using matching
seeds, character, reasoning effort, and wall-clock/key budgets. `--jobs` defaults
to 1 and supports up to 4 independent simultaneous games. The supervisor stops
Codex at its deadline, copies the trace and milestones, and closes the trial's
game process. `result.json` distinguishes inference errors, early model exits,
time limits, key limits, and ended episodes. It does not infer an ascension.

`./play --session NAME watch` observes a live game without taking actions.
`./play watch --run DIRECTORY` reads the latest saved trial frame, and
`./play replay DIRECTORY` replays its saved screens. Add `--once` to watch or
`--frames 1 --delay 0` to replay for a single frame. Named games live under
`.local/sessions/NAME/`. `ORIOLE_SESSION` selects the named game inside the trial's
Codex subprocess. The original default game remains in `.local/`.
