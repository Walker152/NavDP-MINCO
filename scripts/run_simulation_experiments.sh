#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_simulation_experiments.sh [options]

By default runs the local mock suite and prepares/validates an eight-run Isaac
dynamic dry-run plan without starting Isaac or NavDP processes.

Options:
  --output PATH             Workflow output root
  --resume                  Reuse hash-compatible completed stages
  --retry-failed            Retry previously failed stages
  --skip-video              Disable video capture for authorized real runs
  --allow-real-simulation   Explicitly authorize the dynamic Isaac pilot
  --full-suite              With authorization, also run the full Isaac suite
  -h, --help                Show this help
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${NAVDP_PYTHON:-}" ]]; then
  PYTHON_BIN="$NAVDP_PYTHON"
elif [[ -x "/home/alioth/miniforge3/envs/navdp/bin/python" ]]; then
  PYTHON_BIN="/home/alioth/miniforge3/envs/navdp/bin/python"
else
  echo "NAVDP_PYTHON is not executable and the navdp Conda environment was not found" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "NAVDP_PYTHON is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "${ISAACLAB_PYTHON:-}" && -x "/home/alioth/miniforge3/envs/isaaclab/bin/python" ]]; then
  ISAACLAB_PYTHON="/home/alioth/miniforge3/envs/isaaclab/bin/python"
fi
export NAVDP_PYTHON="$PYTHON_BIN"
export ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-}"
export PYTHONPATH="$REPO_ROOT/minco_processor/build:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m experiments run-simulation-workflow "$@"
