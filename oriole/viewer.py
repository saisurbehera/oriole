"""Read-only terminal views for live games and saved action-by-action replays."""

import json
import sys
import time
from pathlib import Path


def read_events(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines()
    events = []
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return events


def frame(state, title="NetHack"):
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    stats = state["stats"]
    print(
        f"{title} | {state.get('stage', 'exploring')} | floor {stats['depth']} | "
        f"turn {stats['turn']} | HP {stats['hp']}/{stats['max_hp']} | "
        f"score {stats['score']} | keys {state['steps']}"
    )
    print(state["screen"])
    if "key" in state:
        print(f"Last key: {state['key']}")
    print(flush=True)


def watch_game(args, observe):
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        state = (
            json.loads((args.run / "state.json").read_text()) if args.run else observe()
        )
        frame(state, args.run.name if args.run else args.session)
        if (
            args.once
            or state["done"]
            or (args.run and (args.run / "result.json").exists())
        ):
            return
        time.sleep(args.interval)


def replay_game(args):
    if (
        args.delay < 0
        or args.start < 0
        or (args.frames is not None and args.frames < 1)
    ):
        raise ValueError("delay and start must be nonnegative; frames must be positive")
    path = args.run / "episode.jsonl" if args.run.is_dir() else args.run
    rows = [row for row in read_events(path) if row["steps"] >= args.start]
    if not rows:
        raise ValueError(f"No replay frames found in {path}")
    for i, row in enumerate(rows):
        if args.frames is not None and i >= args.frames:
            break
        frame(row, args.run.name)
        time.sleep(args.delay)
