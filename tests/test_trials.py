"""Checks for isolated sessions, trace metrics, and subscription-only runs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from oriole.game import ROOT, session_path
from oriole.trials import (
    codex_command,
    milestones,
    run_trial,
    subscription_environment,
    summarize,
)
from oriole.viewer import read_events


def row(step, depth=1, hp=16, score=0, message="", done=False):
    return {
        "event": "reset" if step == 0 else "action",
        "key": "search",
        "steps": step,
        "stats": {
            "x": step,
            "y": 3,
            "depth": depth,
            "hp": hp,
            "max_hp": 16,
            "score": score,
            "level": 1,
            "turn": step + 1,
        },
        "message": message,
        "done": done,
    }


class TrialTests(unittest.TestCase):
    def test_metrics_and_milestones_use_observed_progress(self):
        rows = [
            row(0),
            row(1, hp=13),
            row(2, depth=2, score=9, message="You kill the jackal!"),
        ]
        result = summarize(rows)
        self.assertEqual(result["deepest_depth"], 2)
        self.assertEqual(result["search_keys"], 2)
        self.assertEqual(result["min_hp"], 13)
        names = [item["milestone"] for item in milestones(rows)]
        self.assertIn("reached_floor_2", names)
        self.assertIn("first_damage", names)
        self.assertIn("first_monster_killed", names)
        self.assertNotIn("ascension", names)

    def test_session_paths_cannot_escape_local_directory(self):
        self.assertEqual(session_path("default"), ROOT / ".local")
        self.assertNotEqual(session_path("first"), session_path("second"))
        for name in ("../outside", "", "x" * 33):
            with self.assertRaises(ValueError):
                session_path(name)

    def test_subscription_authentication_removes_api_key_and_rejects_api_login(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-value"}),
            patch("oriole.trials.shutil.which", return_value="codex"),
            patch("oriole.trials.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess(
                [], 0, "Logged in using ChatGPT", ""
            )
            self.assertNotIn("OPENAI_API_KEY", subscription_environment())
            run.return_value = subprocess.CompletedProcess(
                [], 0, "Logged in using an API key", ""
            )
            with self.assertRaises(ValueError):
                subscription_environment()
        command = codex_command("gpt-6-luna", "medium", Path("/tmp/test-trial"))
        self.assertIn('forced_login_method="chatgpt"', command)

    def test_incomplete_live_trace_line_is_ignored_until_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.jsonl"
            path.write_text(json.dumps(row(0)) + '\n{"event":')
            self.assertEqual(len(read_events(path)), 1)

    def test_separate_processes_observe_isolated_games_and_seeded_starts(self):
        names = ["test-one", "test-two"]

        def call(name, *args):
            result = subprocess.run(
                [sys.executable, "-m", "oriole", "--session", name, "--json", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode:
                self.fail(result.stderr)
            return json.loads(result.stdout)

        try:
            first = call(names[0], "reset", "--seed", "43")
            second = call(names[1], "reset", "--seed", "43")
            self.assertEqual(first["screen"], second["screen"])
            advanced = call(names[0], "act", "search")
            self.assertGreater(advanced["stats"]["turn"], first["stats"]["turn"])
            self.assertEqual(second, call(names[1], "observe"))
            call(names[0], "observe")
            self.assertEqual(advanced, call(names[0], "observe"))
        finally:
            for name in names:
                subprocess.run(
                    [sys.executable, "-m", "oriole", "--session", name, "stop"],
                    cwd=ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )

    def test_supervisor_enforces_deadline_preserves_trace_and_closes_game(self):
        # Stand in for inference while exercising the real game process and timeout.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            executable.write_text("#!/bin/sh\n./play act search >/dev/null\nsleep 30\n")
            executable.chmod(0o755)
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]}
            output = io.StringIO()
            with redirect_stdout(output):
                result = run_trial(
                    "test-model", 43, 1, 100, "medium", env, runs_dir=root, as_json=True
                )
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(events[-1]["event"], "trial.finished")
            self.assertEqual(result["stop_reason"], "time_limit")
            self.assertEqual(result["actions"], 1)
            self.assertLess(result["elapsed_seconds"], 10)
            self.assertTrue((Path(result["folder"]) / "episode.jsonl").exists())
            self.assertFalse((session_path(result["session"]) / "game.sock").exists())
