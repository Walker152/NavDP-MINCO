#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_static_experiments.sh [options]

Run calibration, native checks, static legacy/SuperPlanner-2D-SFC benchmarks, boundary tests,
paper figures, tables, validation, and artifact indexing.

Options:
  --output PATH     Workflow output root (default: timestamped results directory)
  --resume          Reuse hash-compatible completed stages
  --retry-failed    Retry previously failed stages
  --skip-video      Disable video work where applicable
  --rolling-showcase-config PATH
                    Override the deterministic rolling showcase configuration
  --skip-rolling-showcase
                    Skip the paper showcase stage explicitly
  -h, --help        Show this help
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
elif [[ -x "${CONDA_ENVS_PATH:-}/navdp/bin/python" ]]; then
  # AutoDL's verified runtime exports CONDA_ENVS_PATH.  Prefer this path over
  # any workstation environment so the static entry is deployable unchanged.
  PYTHON_BIN="${CONDA_ENVS_PATH}/navdp/bin/python"
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
export NAVDP_PYTHON="$PYTHON_BIN"
export PYTHONPATH="$REPO_ROOT/minco_processor/build:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m experiments run-static-workflow "$@"
