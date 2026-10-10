# Oriole — NetHack with Codex

Let OpenAI models play a real local NetHack game through shell commands using your
**Codex/ChatGPT subscription login**. Watch the dungeon live, replay every key, and
compare how far different models get. Model inference is remote; the game is local.

## Start

Install Codex CLI and Python 3.11, then:

```sh
./scripts/setup.sh
codex login
./scripts/play-codex.sh --model gpt-6-luna
```

The sign-in uses your ChatGPT plan; no API key or vLLM server is needed. Trials
require ChatGPT authentication and never fall back to API-key billing. Availability
and usage limits depend on your account. See [official authentication docs](https://learn.chatgpt.com/docs/auth).

## Run and compare

```sh
# One bounded run in its own game
./play trial --model gpt-6-luna --seconds 600 --seed 43

# Same seed, character, reasoning effort and time allowance for each model
./play bench --models gpt-6-luna gpt-6-sol gpt-6.1-sol gpt-6-astra \
  --seeds 43 --seconds 300 --effort medium

./play results
./play results --live                 # progress and watch commands
```

Comparisons run sequentially by default. Add `--jobs 4` for separate concurrent
games, or `--seeds 43 44 45` for more trials. Unsupported models and inference errors
are recorded as failures rather than silently substituted. Ctrl+C stops the runners
and preserves their traces. Runs consume your normal Codex allowance.

## Watch the stages

Each trial prints its session name and saved run directory:

```sh
./play --session SESSION_NAME watch        # live dungeon, floor, HP, score, turn
./play watch --run .local/trials/RUN_NAME   # live/saved trial state
./play replay .local/trials/RUN_NAME        # every dungeon screen and key
```

Runs save `result.json`, `milestones.json`, `episode.jsonl`, and Codex output under
`.local/trials/`. Milestones include reaching floors, taking damage, killing a
monster, and gaining experience levels. Trial game processes close at the end;
saved screens and replays remain available. An episode ending is not a win:
ascension must be confirmed from the final screen.

For manual play: `./play human`. For interactive Codex, open another terminal and
use `./play watch` to observe the default game.

## Results and code

Read the published musing: [Codex in the dungeon](https://www.saisur.me/musings/codex-in-the-dungeon).
The [project guide](docs/README.md) maps the code, recorded experiment, and next steps.

See [trial results](docs/model-trials.md) for measured runs and their limits.
The maintained interface is in `oriole/`; historical experiments are in
`research/`. [Gameplay details](docs/gameplay.md) and [repository layout](docs/repository-layout.md)
explain the rest. Source, small graph data, and the four archived game replays are tracked.
New run logs and environments stay local. Replay the published experiment after cloning:

```sh
./play replay data/trials/2026-10-10/gpt-6.1-sol
```

```sh
.local/play-venv/bin/python -m unittest discover -s tests -v
```

## Repository map

| Path | Purpose |
| --- | --- |
| `oriole/`, `play` | Maintained game bridge, Codex trial runner, and terminal replays |
| `scripts/` | Setup, interactive Codex launcher, and recorded-trial export |
| `tests/` | Checks for game state, isolated trials, authentication, and archives |
| `docs/`, `data/trials/` | Experiment notes, measurements, and replayable evidence |
| `research/`, `scripts/legacy/` | Historical knowledge-graph, training, and API experiments |
| `nle/`, `ultra/`, `BALROG/` | Vendored research dependencies with original licenses |
| `.local/` | Ignored environments, live sessions, and new trial outputs |
