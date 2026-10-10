"""Ensure a fresh clone retains the evidence behind the published comparison."""
import hashlib
import json
import unittest
from pathlib import Path

from oriole.trials import milestones, summarize
from oriole.viewer import read_events

ROOT = Path(__file__).resolve().parent.parent


class ArchiveTests(unittest.TestCase):
    def test_archived_games_reproduce_reported_results_and_milestones(self):
        report = json.loads((ROOT / "data/trials/comparison-2026-10-10.json").read_text())
        for result in report["results"]:
            with self.subTest(model=result["model_requested"]):
                folder = ROOT / "data/trials/2026-10-10" / result["model_requested"]
                rows = read_events(folder / "episode.jsonl")
                self.assertEqual([r["steps"] for r in rows], list(range(result["actions"] + 1)))
                self.assertEqual(hashlib.sha256(rows[0]["screen"].encode()).hexdigest(), result["initial_screen_sha256"])
                self.assertEqual(rows[-1]["stats"], result["final_stats"])
                actual = summarize(rows)
                for field in ("actions", "deepest_depth", "highest_score", "unique_positions", "search_keys", "min_hp"):
                    self.assertEqual(actual[field], result[field])
                self.assertEqual(milestones(rows), result["milestones"])
                self.assertEqual(json.loads((folder / "milestones.json").read_text()), result["milestones"])
