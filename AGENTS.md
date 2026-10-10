# Oriole

This repository includes a local NetHack game interface for Codex. When asked to
play, use local shell commands; no OpenAI API key, BALROG evaluator, vLLM server,
or model training is needed. Do not spawn agents for gameplay.

## Playing

- `./play status`: check whether the persistent game process exists.
- `./play observe`: get the screen, cursor, message, HP, depth, turn, and done flag.
  This does not advance time. If no episode exists, use `./play reset --seed 42`.
- `./play act north`: take an action. Directions use NetHack vi keys.
- `./play act open east`: send a command followed by a direction.
- `./play act inventory`, `./play act escape`, `./play act enter`, `./play act space`:
  inspect inventory, dismiss a prompt, submit input, or advance a --More-- page.
- `./play act '>'`: descend when standing on stairs. Quote shell punctuation.
- `./play actions`: list names. Literal one-character NetHack keys also work.
- `./play knowledge 'soldier ant'`: retrieve sourced facts from the local graph.
- `./play --json observe`: machine readable form (global --json goes first).
- `./play stop`: close the game process. This loses the live episode.

Observe before acting. Read menus and questions carefully; keys can mean different
things inside prompts. Use one action or a short sequence, then inspect the screen.
Long repeats can miss monsters, traps, or low HP. Watch hunger and HP; avoid combat
you cannot win. Explore, collect useful supplies, find stairs, and progress toward
ascension. Graph facts are extracted wiki material, including variants, and may be
incomplete or wrong; consult source fields and the actual game state.

The game stays alive across shell commands. Do not reset a live episode unless
asked to start over. Death and time limits end an episode; `done` alone does not
prove a win. Only report completion of NetHack when the final screen establishes
ascension. No promise is made that Codex can win a full game. Do not modify rules,
rewards, or telemetry to manufacture success.

The transcript is `.local/episode.jsonl`; copy it before reset if it is needed.
The latest game log is `.local/game-server.log`. Game state is held in memory;
it survives command exits, but not process shutdown or a machine reboot.

## Development

The maintained local interface is `oriole/`. Historical experiments are in
`old/research/`; run them from `old/` (`cd old; python research/example.py`) so
their existing data paths work. Historical shell scripts are in `old/scripts/legacy/`.
Do not run those installers as part of local gameplay.

Run `.local/play-venv/bin/python -m unittest discover -s tests -v` for the gameplay checks.
Vendor source snapshots are tracked in `old/nle/`, `old/ultra/`, and `old/BALROG/`, including
local edits. Provenance is in `docs/vendor-provenance.json`. Source files and small
derived datasets belong in Git; environments, credentials, checkpoints, raw data,
caches, and episode logs do not. See `docs/repository-layout.md` for exclusions.

## Model comparison sessions

The trial runner selects an isolated game with `ORIOLE_SESSION`. Use ordinary
`./play` commands; do not change this variable, reset the episode, or access other
sessions. Follow the runner's wall-clock and key budgets. Observation includes the
current stage and exploration visit counts. Track your route in your reasoning,
prefer unexplored paths, and avoid repeated searches around the same room.

`./play trial --model MODEL` and `./play bench --models MODEL_A MODEL_B` use Codex
with forced ChatGPT authentication. They require a subscription login and preserve
traces under `.local/trials/`. `./play results` summarizes them. `watch` and `replay`
are read-only views; the trial closes its game process after saving its final state.

Never commit credentials, auth files, or local Codex session logs. Run the
Gitleaks history and tracked-file checks described in `docs/credential-checks.md`
before publishing changes. Keep new experiments out of `old/`, which is archival.
