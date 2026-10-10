#!/bin/sh
set -eu
ORIOLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ORIOLE_ROOT"
mkdir -p .local
if [ ! -x .local/play-venv/bin/python ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.11 .local/play-venv
    else
        python3 -m venv .local/play-venv
    fi
fi
if command -v uv >/dev/null 2>&1; then
    uv pip install --python .local/play-venv/bin/python -e .
else
    .local/play-venv/bin/python -m pip install -e .
fi
./play --help
printf '\nReady. Run ./play human, or ask Codex to play using AGENTS.md.\n'
