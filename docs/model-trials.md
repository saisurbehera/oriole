# Codex NetHack trials

Run on 2026-10-10 using Codex CLI 0.162.1 and NLE 1.2.0, authenticated with
ChatGPT. Every model received seed 43, a lawful human Valkyrie, medium reasoning,
a 300-second wall-clock budget, a 1,500-key cap, and the same instructions.
The four games ran concurrently in isolated sessions. Their initial dungeon
screens had identical SHA-256 hashes. Shutdown adds about one second to elapsed time.

| Model | Deepest floor | Score | Game turn | Keys | Final HP | End |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| gpt-6-luna | 2 | 67 | 126 | 159 | 14/16 | time limit, alive |
| gpt-6-sol | 2 | 79 | 175 | 198 | 16/16 | time limit, alive |
| gpt-6.1-sol | 7 | 607 | 394 | 470 | 29/39 | time limit, alive |
| gpt-6-astra | 6 | 438 | 496 | 592 | 27/27 | time limit, alive |

GPT-6.1 Sol reached the deepest floor in this trial. Astra sent more keys,
while Sol and Luna progressed more slowly. All four were
alive when the supervisor stopped them. None ascended. This single seed and short
time allowance are insufficient for a broad performance ranking; multiple seeds,
longer budgets, and repeated runs would make the comparison more useful. Concurrent
runs measure real-time progress, including inference latency, not equal numbers of
decisions or equal token consumption.

[Machine-readable measurements and milestones](../data/trials/comparison-2026-10-10.json)
are tracked in Git. The four game traces and milestones are archived under
`data/trials/2026-10-10/`, so a clone retains the screens and keys. Original Codex
events, live snapshots, and optional model summaries remain under `.local/trials/`
on the machine where the runs happened.
No API keys or authentication tokens are saved in the comparison report.

## Watch or replay

During a comparison, `./play results --live` prints live metrics and watch commands.
For these completed runs, replay a saved directory:

```sh
./play replay data/trials/2026-10-10/gpt-6.1-sol
```

`milestones.json` lists key numbers for each reached floor, first damage, first
monster killed, and experience level increases. To jump to one, use
`./play replay RUN_DIRECTORY --start KEY_NUMBER`. `watch --run .local/trials/RUN_DIRECTORY`
shows the final snapshot of a locally retained trial; the Git archive contains
replay frames and milestones only. Trial servers close after their
traces are saved; replays do not resume the live game.

## Original Luna experiment

The earlier ten-minute Luna run on seed 43 reached floor 1, score 9, and turn 273
with 305 keys and 16/16 HP. It used harness version 1 and older instructions;
reasoning effort was not recorded. Its raw trace remains in
`.local/trials/20261010T124825Z-gpt-6-luna/`. It is retained as an exploratory
baseline and is not included in the paired comparison table above. The improved
instructions explicitly ask models to avoid cycling and repeated searches; the
new observation also reports exploration visit counts.

## Published reference

[Codex in the dungeon](https://www.saisur.me/musings/codex-in-the-dungeon) presents
these measurements and recordings. Website source lives in the separate
[saisur.me repository](https://github.com/saisurbehera/saisur.me), introduced in
commit `acf5bdf`. Its `scripts/export-oriole.mjs` reads this comparison report and
the original local episode logs to build browser replays. The Git archive here
retains the same screens and keys for future reference.
