#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/root/NavDP}"
ARCHIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_NAME="_minco_processor.cpython-310-x86_64-linux-gnu.so"
TARGET_DIR="$REPO_ROOT/minco_processor/build"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/root/autodl-tmp/navdp/conda/envs/isaaclab/bin/python}"

mkdir -p "$TARGET_DIR"
install -m 755 "$ARCHIVE_DIR/$MODULE_NAME" "$TARGET_DIR/$MODULE_NAME"

if [[ ! -x "$ISAACLAB_PYTHON" ]]; then
  echo "IsaacLab Python not found or not executable: $ISAACLAB_PYTHON" >&2
  echo "Set ISAACLAB_PYTHON to the correct interpreter path and retry." >&2
  exit 1
fi

cd "$REPO_ROOT"
"$ISAACLAB_PYTHON" - <<'PY'
import minco_processor
import _minco_processor

doc = minco_processor.MincoProcessor.configure.__doc__ or ""
required = (
    "optimization_safe_dist",
    "validation_safe_dist",
    "start_validation_exemption_radius",
)
missing = [name for name in required if name not in doc]
if missing:
    raise RuntimeError(f"wrong MINCO extension loaded; missing arguments: {missing}")

processor = minco_processor.MincoProcessor()
processor.configure(
    max_vel=1.0,
    max_acc=1.0,
    optimization_safe_dist=0.45,
    validation_safe_dist=0.35,
    start_validation_exemption_radius=0.35,
    sample_dt=0.05,
    max_iterations=64,
    max_yaw_rate=0.5,
    penalty_weight_pos=10000.0,
    penalty_weight_vel=1000.0,
    penalty_weight_acc=1000.0,
    penalty_weight_attractor=20.0,
    time_weight=0.1,
    time_barrier_weight=10.0,
)
print("MINCO extension:", _minco_processor.__file__)
print("MINCO decoupled configure: OK")
PY
