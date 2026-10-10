"""Persistent NLE game, compatible with old Gym and new Gymnasium NLE."""

import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def session_path(name="default"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", name):
        raise ValueError(
            "Session names must be 1–32 letters, numbers, underscores or hyphens"
        )
    return ROOT / ".local" if name == "default" else ROOT / ".local/sessions" / name


LOCAL = session_path(os.environ.get("ORIOLE_SESSION", "default"))
KEYS = {
    "north": "k",
    "south": "j",
    "east": "l",
    "west": "h",
    "northeast": "u",
    "northwest": "y",
    "southeast": "n",
    "southwest": "b",
    "wait": ".",
    "search": "s",
    "inventory": "i",
    "pickup": ",",
    "down": ">",
    "up": "<",
    "open": "o",
    "close": "c",
    "eat": "e",
    "quaff": "q",
    "read": "r",
    "wear": "W",
    "wield": "w",
    "drop": "d",
    "enter": "\r",
    "space": " ",
    "escape": "\x1b",
    "kick": "\x04",
}


class Game:
    def __init__(self):
        self.env = None
        self.obs = None
        self.done = False
        self.steps = 0
        self.reward = 0.0
        self.visits = Counter()
        self.end_reason = None

    def reset(self, seed=42, character="val-hum-fem-law", max_steps=100000):
        from nle.env.tasks import NetHackScore
        from nle import nethack

        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(seed, int) or not 0 <= seed <= 4294967295:
            raise ValueError("seed must be between 0 and 4294967295")
        candidate = NetHackScore(
            character=character,
            max_episode_steps=max_steps,
            actions=nethack.ACTIONS,
            allow_all_modes=True,
            allow_all_yn_questions=True,
            penalty_step=0.0,
            observation_keys=("tty_chars", "tty_cursor", "blstats", "message"),
        )
        # The installed older NLE has seed()/reset(); vendored NLE uses reset(seed=).
        try:
            if hasattr(candidate, "seed"):
                candidate.seed(seed, seed)
                result = candidate.reset()
            else:
                result = candidate.reset(seed=seed)
        except Exception:
            candidate.close()
            raise
        if self.env is not None:
            self.env.close()
        self.env = candidate
        self.obs = result[0] if isinstance(result, tuple) else result
        self.done = False
        self.steps = 0
        self.reward = 0.0
        self.visits.clear()
        self.end_reason = None
        self._visit()
        LOCAL.mkdir(parents=True, exist_ok=True)
        (LOCAL / "episode.jsonl").write_text("")
        self._record({"event": "reset", "seed": seed, "character": character})
        return self.observe()

    def observe(self):
        if self.obs is None:
            raise ValueError("No game yet. Run: ./play reset")
        stats = self.obs["blstats"]
        screen = "\n".join(
            bytes(row).decode("ascii", errors="replace").rstrip()
            for row in self.obs["tty_chars"]
        )
        position = self._position()
        stage = "exploring"
        if self.done:
            stage = "episode_ended"
        elif "--More--" in screen or "(end)" in screen:
            stage = "menu"
        elif int(stats[10]) <= max(1, int(stats[11]) // 3):
            stage = "low_health"
        elif int(stats[21]) >= 2:
            stage = "hungry"
        return {
            "screen": screen,
            "cursor": self.obs["tty_cursor"].tolist(),
            "message": bytes(self.obs["message"])
            .split(b"\0", 1)[0]
            .decode("utf-8", errors="replace"),
            "stats": dict(
                zip(
                    (
                        "x",
                        "y",
                        "score",
                        "hp",
                        "max_hp",
                        "depth",
                        "gold",
                        "energy",
                        "max_energy",
                        "armor_class",
                        "level",
                        "experience",
                        "turn",
                        "hunger",
                        "dungeon_number",
                        "dungeon_level",
                    ),
                    (
                        int(stats[i])
                        for i in (
                            0,
                            1,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            18,
                            19,
                            20,
                            21,
                            23,
                            24,
                        )
                    ),
                )
            ),
            "steps": self.steps,
            "reward": self.reward,
            "done": self.done,
            "stage": stage,
            "end_reason": self.end_reason,
            "exploration": {
                "unique_positions": len(self.visits),
                "keys_at_current_position": self.visits[position],
            },
            "completion": (
                "episode ended; inspect the final screen for death, quitting, or ascension"
                if self.done
                else "in progress"
            ),
        }

    def act(self, keys, repeat=1):
        if self.env is None:
            raise ValueError("No game yet. Run: ./play reset")
        if self.done:
            raise ValueError("Episode ended. Run ./play reset for a new game.")
        if not isinstance(keys, list) or not keys or len(keys) > 32:
            raise ValueError("Supply between 1 and 32 keys.")
        if not isinstance(repeat, int) or not 1 <= repeat <= 20:
            raise ValueError("repeat must be between 1 and 20")
        action_ids = {int(action): i for i, action in enumerate(self.env.actions)}
        actions = []
        for key in keys:
            char = KEYS.get(key, key)
            if len(char) != 1 or ord(char) not in action_ids:
                raise ValueError(f"Unsupported key: {key!r}. Run ./play actions.")
            actions.append((key, action_ids[ord(char)]))
        for _ in range(repeat):
            for key, action in actions:
                result = self.env.step(action)
                if len(result) == 5:
                    self.obs, reward, terminated, truncated, info = result
                    self.done = bool(terminated or truncated)
                    if truncated:
                        self.end_reason = "step_limit"
                else:
                    self.obs, reward, done, info = result
                    self.done = bool(done)
                if self.done and self.end_reason is None:
                    self.end_reason = "episode_ended"
                self.reward += float(reward)
                self.steps += 1
                self._visit()
                self._record({"event": "action", "key": key})
                if self.done:
                    break
            if self.done:
                break
        return self.observe()

    def _record(self, event):
        with (LOCAL / "episode.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        **event,
                        **self.observe(),
                    }
                )
                + "\n"
            )

    def _position(self):
        stats = self.obs["blstats"]
        return tuple(int(stats[i]) for i in (23, 24, 0, 1))

    def _visit(self):
        self.visits[self._position()] += 1

    def close(self):
        if self.env is not None:
            self.env.close()


def knowledge(query, limit=12):
    """Return sourced existing triples without LLM calls or inferred facts."""
    path = ROOT / "data/nethack_kg_final_a_plus.tsv"
    if not query.strip():
        raise ValueError("Supply an entity or search phrase")
    query = query.casefold().strip()
    with path.open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream, delimiter="\t")
            if query in row["head"].casefold() or query in row["tail"].casefold()
        ]
    rows.sort(
        key=lambda row: (row["head"].casefold() == query, int(row.get("priority", 0))),
        reverse=True,
    )
    return {"query": query, "matches": len(rows), "facts": rows[:limit]}
