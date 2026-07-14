#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_all_experiments.sh <suite.json> [options]}"
shift

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "error: activate the navdp conda environment before running this script" >&2
  exit 127
fi

"$PYTHON_BIN" -m experiments run-suite --config "$CONFIG" "$@"
