# Historical archive

This directory preserves earlier research and dependency snapshots. The maintained
Codex game runner is in `../oriole/`; it installs published NLE 1.2.0 and does not
import these archived projects.

| Path | Contents |
| --- | --- |
| `research/` | Knowledge graph extraction, retrieval, training, and older API experiments |
| `scripts/legacy/` | Historical installers and experiment launchers |
| `data/` | Historical derived graphs and benchmarks |
| `notebooks/`, `sample_data/` | Exploration notebooks and example graphs |
| `nle/`, `ultra/`, `BALROG/` | Source snapshots and local research edits; original licenses retained |
| `requirements_*.txt` | Older CUDA and server dependencies |

Run historical scripts from this directory if needed:

```sh
cd old
python research/example.py
```

These experiments are not the maintained test suite. Their old environments,
CUDA dependencies, raw datasets, and checkpoints may be needed to reproduce them.
Large artifacts remain ignored and are not included in clones. The root-owned
`data/nethack_ultra/` directory remains at its ignored local path after its tracked
files were copied to the archive. See
[repository layout](../docs/repository-layout.md) and
[artifact inventory](../docs/local-artifacts.json).

Use [the root README](../README.md) for current gameplay and recorded model trials.
