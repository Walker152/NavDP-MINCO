#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_simulation_experiments.sh [options]

By default runs the local mock suite and prepares/validates a twelve-run Isaac
dynamic dry-run plan: native NavDP, MINCO without a corridor, and MINCO with
the recorded SuperPlanner 2D SFC. It does not start Isaac or NavDP processes.
Real Isaac execution is accepted only on the configured AutoDL server.

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

# Real Isaac runs are deliberately server-only.  A local laptop may have a
# visible GPU and even an IsaacLab environment, but it does not have the
# memory/cache layout validated by the AutoDL repair workflow.  Static work,
# local mock runs and Isaac dry-runs remain available everywhere.
REAL_SIMULATION=0
for arg in "$@"; do
  [[ "$arg" == "--allow-real-simulation" ]] && REAL_SIMULATION=1
done
if ((REAL_SIMULATION)); then
  RUNTIME_ENV_FILE="${NAVDP_RUNTIME_ENV_FILE:-/root/.config/navdp/autodl-runtime.env}"
  if [[ ! -f "$RUNTIME_ENV_FILE" || -L "$RUNTIME_ENV_FILE" ]]; then
    echo "real simulation is restricted to the AutoDL server runtime; run scripts/setup_autodl.sh and scripts/autodl_self_check_repair.sh on AutoDL first" >&2
    exit 2
  fi
  # This file is written atomically by autodl_self_check_repair.sh and fixes
  # the Vulkan ICD, cache root, Conda environments, and IsaacLab checkout.
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_FILE"
  if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != YES ]]; then
    echo "real simulation requires explicit NVIDIA Omniverse EULA acceptance; rerun scripts/autodl_self_check_repair.sh --accept-isaac-eula" >&2
    exit 2
  fi
  if [[ "${AUTODL_WORK_DIR:-}" != "/root/autodl-tmp/navdp" ||
        "${ISAACLAB_DIR:-}" != "/root/autodl-tmp/navdp/IsaacLab" ||
        "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" != "/root/NavDP" ||
        ! -x "$ISAACLAB_DIR/isaaclab.sh" ]]; then
    echo "real simulation is restricted to the AutoDL server runtime (/root/NavDP + /root/autodl-tmp/navdp); local execution is disabled" >&2
    exit 2
  fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${NAVDP_PYTHON:-}" ]]; then
  PYTHON_BIN="$NAVDP_PYTHON"
elif [[ -x "${CONDA_ENVS_PATH:-}/navdp/bin/python" ]]; then
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
if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
  if [[ -x "${CONDA_ENVS_PATH:-}/isaaclab/bin/python" ]]; then
    ISAACLAB_PYTHON="${CONDA_ENVS_PATH}/isaaclab/bin/python"
  elif [[ -x "/home/alioth/miniforge3/envs/isaaclab/bin/python" ]]; then
    ISAACLAB_PYTHON="/home/alioth/miniforge3/envs/isaaclab/bin/python"
  fi
fi
if [[ -z "${ISAACLAB_DIR:-}" ]]; then
  if [[ -n "${AUTODL_WORK_DIR:-}" && -x "$AUTODL_WORK_DIR/IsaacLab/isaaclab.sh" ]]; then
    ISAACLAB_DIR="$AUTODL_WORK_DIR/IsaacLab"
  elif [[ -x "$REPO_ROOT/../IsaacLab/isaaclab.sh" ]]; then
    ISAACLAB_DIR="$REPO_ROOT/../IsaacLab"
  fi
fi
export NAVDP_PYTHON="$PYTHON_BIN"
export ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-}"
export ISAACLAB_DIR="${ISAACLAB_DIR:-}"
export PYTHONPATH="$REPO_ROOT/minco_processor/build:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m experiments run-simulation-workflow "$@"
