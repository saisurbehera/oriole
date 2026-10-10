#!/bin/sh
set -eu
ORIOLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ORIOLE_ROOT"
command -v codex >/dev/null 2>&1 || { echo 'Install Codex CLI and run codex login first.' >&2; exit 1; }
# Interactive Codex uses ChatGPT subscription authentication and local shell tools.
unset OPENAI_API_KEY
exec codex -c 'forced_login_method="chatgpt"' 'Play the current NetHack episode through ./play following AGENTS.md. If there is no episode, start one. Consult ./play knowledge when useful. Aim to survive and explore toward ascension. Use short action sequences and inspect the screen after each. Keep playing until the episode ends or I stop you. Report actual progress and whether the final screen proves ascension. Do not change the code to alter game results.' "$@"
