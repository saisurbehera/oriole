#!/usr/bin/env python3
"""Archive game observations from the recorded comparison, without Codex logs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data/trials/comparison-2026-10-10.json"
DESTINATION = ROOT / "data/trials/2026-10-10"
EVENT_FIELDS = {
    "recorded_at", "event", "seed", "character", "screen", "cursor", "message",
    "stats", "steps", "reward", "done", "stage", "end_reason", "exploration",
    "completion", "key",
}


def main():
    report = json.loads(REPORT.read_text())
    # Validate every source before writing the archive.
    prepared = []
    for result in report["results"]:
        rows = [json.loads(line) for line in
                (ROOT / result["run_directory"] / "episode.jsonl").read_text().splitlines()]
        assert len(rows) == result["actions"] + 1, "Incomplete recorded trace"
        assert [row["steps"] for row in rows] == list(range(len(rows)))
        assert hashlib.sha256(rows[0]["screen"].encode()).hexdigest() == result["initial_screen_sha256"]
        assert rows[-1]["stats"] == result["final_stats"]
        clean = [{key: value for key, value in row.items() if key in EVENT_FIELDS} for row in rows]
        prepared.append((result, clean))
    for result, rows in prepared:
        folder = DESTINATION / result["model_requested"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "episode.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (folder / "milestones.json").write_text(json.dumps(result["milestones"], indent=2) + "\n")
    print(f"Archived {len(prepared)} games to {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
