"""Checks using real NLE: persistence, turn advancement, validation and termination."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from oriole.game import Game, knowledge


class GameTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.local_patch = patch("oriole.game.LOCAL", Path(self.directory.name))
        self.local_patch.start()
        self.game = Game()

    def tearDown(self):
        self.game.close()
        self.local_patch.stop()
        self.directory.cleanup()

    def test_observing_does_not_advance_turn_and_search_does(self):
        initial = self.game.reset(seed=42)
        self.assertIn("Agent", initial["screen"])
        self.assertGreater(initial["stats"]["hp"], 0)
        self.assertEqual(initial, self.game.observe())
        after = self.game.act(["search"])
        self.assertGreater(after["stats"]["turn"], initial["stats"]["turn"])
        self.assertFalse(after["done"])

    def test_invalid_sequence_is_rejected_before_any_action(self):
        initial = self.game.reset(seed=42)
        with self.assertRaises(ValueError):
            self.game.act(["north", "not-a-key"])
        self.assertEqual(initial, self.game.observe())
        with self.assertRaises(ValueError):
            self.game.act(["north"], repeat=21)
        with self.assertRaises(ValueError):
            self.game.reset(max_steps=0)
        self.assertEqual(initial, self.game.observe())

    def test_episode_limit_and_reset(self):
        self.game.reset(seed=42, max_steps=2)
        result = self.game.act(["search"], repeat=5)
        self.assertTrue(result["done"])
        self.assertEqual(result["steps"], 2)
        with self.assertRaises(ValueError):
            self.game.act(["north"])
        self.assertFalse(self.game.reset(seed=43)["done"])

    def test_known_facts_have_sources_and_limit(self):
        result = knowledge("Medusa", limit=3)
        self.assertEqual(len(result["facts"]), 3)
        self.assertEqual(result["facts"][0]["head"], "Medusa")
        self.assertTrue(all(fact["source"] for fact in result["facts"]))
        self.assertEqual(knowledge("an entity that does not exist")["matches"], 0)


if __name__ == "__main__":
    unittest.main()
