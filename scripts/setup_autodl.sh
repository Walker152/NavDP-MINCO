#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NAVDP_ENV_NAME="${NAVDP_ENV_NAME:-navdp}"
ISAACLAB_ENV_NAME="${ISAACLAB_ENV_NAME:-isaaclab}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${AUTODL_ROOT:-$HOME}/IsaacLab}"
AUTODL_EXPORT_DIR="${AUTODL_EXPORT_DIR:-$REPO_ROOT/requirements/autodl}"
AUTODL_MIN_FREE_GB="${AUTODL_MIN_FREE_GB:-35}"
ISAACSIM_VERIFY_TIMEOUT="${ISAACSIM_VERIFY_TIMEOUT:-180}"
CHECK_ONLY=0
SKIP_VERIFY=0
CURRENT_STAGE="startup"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_autodl.sh [OPTIONS]

Create the NavDP and IsaacLab Conda environments documented by this repository.

Options:
  --check-only   Validate the host and repository without installing anything
  --skip-verify  Install dependencies but skip runtime smoke tests
  -h, --help     Show this help message

Environment variables:
  NAVDP_ENV_NAME          NavDP environment name (default: navdp)
  ISAACLAB_ENV_NAME       IsaacLab environment name (default: isaaclab)
  ISAACLAB_DIR            IsaacLab checkout path (default: AUTODL_ROOT/IsaacLab or HOME/IsaacLab)
  PIP_INDEX_URL           Optional primary pip mirror
  AUTODL_MIN_FREE_GB      Required free disk space in GiB (default: 35)
  ISAACSIM_VERIFY_TIMEOUT Headless verification timeout in seconds (default: 180)
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local status=$?
  printf 'error: stage "%s" failed (exit %s)\n' "$CURRENT_STAGE" "$status" >&2
  exit "$status"
}
trap on_error ERR

log() {
  printf '[autodl-setup] %s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_env_name() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid Conda environment name: $1"
}

conda_env_exists() {
  local listing
  listing="$(conda env list --json)"
  grep -Eq "/${1}\"([][,}])" <<<"$listing"
}

create_or_reuse_env() {
  local name="$1"
  local definition="$2"
  if conda_env_exists "$name"; then
    log "Reusing Conda environment: $name"
  else
    CURRENT_STAGE="create Conda environment $name"
    conda env create -n "$name" -f "$definition"
  fi
}

pip_install() {
  local env_name="$1"
  shift
  local args=(run -n "$env_name" python -m pip install)
  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    args+=(--index-url "$PIP_INDEX_URL")
  fi
  conda "${args[@]}" "$@"
}

preflight() {
  CURRENT_STAGE="preflight"
  [[ "$(uname -s)" == "Linux" ]] || die "this setup supports Linux hosts only"
  require_command conda
  require_command git
  require_command nvidia-smi
  require_command timeout
  validate_env_name "$NAVDP_ENV_NAME"
  validate_env_name "$ISAACLAB_ENV_NAME"
  [[ "$NAVDP_ENV_NAME" != "$ISAACLAB_ENV_NAME" ]] || die "NavDP and IsaacLab environment names must differ"
  [[ "$AUTODL_MIN_FREE_GB" =~ ^[0-9]+$ ]] || die "AUTODL_MIN_FREE_GB must be a non-negative integer"
  [[ "$ISAACSIM_VERIFY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "ISAACSIM_VERIFY_TIMEOUT must be a positive integer"

  local required
  for required in \
    "$REPO_ROOT/baselines/navdp/requirements.txt" \
    "$REPO_ROOT/requirements.txt" \
    "$REPO_ROOT/configs/environments/navdp-autodl.yml" \
    "$REPO_ROOT/configs/environments/isaaclab-autodl.yml"; do
    [[ -f "$required" ]] || die "required repository file not found: $required"
  done

  local free_kb required_kb
  free_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
  required_kb=$((AUTODL_MIN_FREE_GB * 1024 * 1024))
  (( free_kb >= required_kb )) || die "insufficient disk space: need at least ${AUTODL_MIN_FREE_GB} GiB free"

  log "Repository: $REPO_ROOT"
  log "Conda: $(command -v conda)"
  log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n 1)"
  log "IsaacLab directory: $ISAACLAB_DIR"
  log "Preflight checks passed"
}

install_isaaclab_checkout() {
  CURRENT_STAGE="prepare IsaacLab v1.2.0 checkout"
  if [[ -e "$ISAACLAB_DIR" ]]; then
    [[ -d "$ISAACLAB_DIR/.git" ]] || die "ISAACLAB_DIR exists but is not a Git checkout: $ISAACLAB_DIR"
    [[ -x "$ISAACLAB_DIR/isaaclab.sh" ]] || die "existing IsaacLab checkout has no executable isaaclab.sh: $ISAACLAB_DIR"
    if [[ -n "$(git -C "$ISAACLAB_DIR" status --porcelain)" ]]; then
      die "IsaacLab checkout has local changes; clean or move it before selecting v1.2.0"
    fi
    log "Reusing IsaacLab checkout: $ISAACLAB_DIR"
    git -C "$ISAACLAB_DIR" fetch --tags --quiet
  else
    mkdir -p "$(dirname "$ISAACLAB_DIR")"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
  fi
  git -C "$ISAACLAB_DIR" checkout --detach v1.2.0

  CURRENT_STAGE="install IsaacLab v1.2.0"
  local install_log
  install_log="$(mktemp)"
  if ! conda run -n "$ISAACLAB_ENV_NAME" bash "$ISAACLAB_DIR/isaaclab.sh" -i >"$install_log" 2>&1; then
    if ! grep -Eqi 'rsl[-_ ]?rl.*(unavailable|not found|no matching distribution)' "$install_log"; then
      sed -n '1,240p' "$install_log" >&2
      rm -f "$install_log"
      die "IsaacLab installer failed"
    fi
    log "Ignoring the README-documented unavailable rsl-rl dependency"
  fi
  sed -n '1,240p' "$install_log"
  rm -f "$install_log"
}

install_benchmark_requirements() {
  CURRENT_STAGE="install NavDP benchmark requirements"
  log "Installing NavDP benchmark requirements (keeping IsaacLab v1.2.0)"
  local compatible_requirements
  compatible_requirements="$(mktemp)"
  # The repository freeze contains isaaclab==2.0.2, but README and this setup
  # intentionally use the v1.2.0 source checkout. Keep every other root pin.
  awk 'tolower($0) !~ /^isaaclab([<=>!~ ]|$)/' \
    "$REPO_ROOT/requirements.txt" >"$compatible_requirements"
  if ! pip_install "$ISAACLAB_ENV_NAME" -r "$compatible_requirements"; then
    rm -f "$compatible_requirements"
    die "benchmark requirements installation failed"
  fi
  rm -f "$compatible_requirements"
}

verify_installation() {
  CURRENT_STAGE="verify NavDP environment"
  conda run -n "$NAVDP_ENV_NAME" python -c 'import diffusers, flask, torch; print("NavDP imports OK; torch", torch.__version__)'

  CURRENT_STAGE="verify IsaacLab Python packages"
  conda run -n "$ISAACLAB_ENV_NAME" python -c 'import isaacsim; print("Isaac Sim import OK")'

  CURRENT_STAGE="verify Isaac Sim headless startup"
  timeout "${ISAACSIM_VERIFY_TIMEOUT}s" conda run -n "$ISAACLAB_ENV_NAME" \
    "$ISAACLAB_DIR/isaaclab.sh" -p \
    "$ISAACLAB_DIR/source/standalone/tutorials/00_sim/create_empty.py" --headless
}

export_snapshots() {
  CURRENT_STAGE="export dependency snapshots"
  mkdir -p "$AUTODL_EXPORT_DIR"
  conda run -n "$NAVDP_ENV_NAME" python -m pip freeze >"$AUTODL_EXPORT_DIR/navdp-freeze.txt"
  conda run -n "$ISAACLAB_ENV_NAME" python -m pip freeze >"$AUTODL_EXPORT_DIR/isaaclab-freeze.txt"
  log "Dependency snapshots: $AUTODL_EXPORT_DIR"
}

while (($#)); do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

preflight
if ((CHECK_ONLY)); then
  exit 0
fi

create_or_reuse_env "$NAVDP_ENV_NAME" "$REPO_ROOT/configs/environments/navdp-autodl.yml"
create_or_reuse_env "$ISAACLAB_ENV_NAME" "$REPO_ROOT/configs/environments/isaaclab-autodl.yml"

CURRENT_STAGE="upgrade NavDP packaging tools"
pip_install "$NAVDP_ENV_NAME" --upgrade pip setuptools wheel
CURRENT_STAGE="install NavDP requirements"
pip_install "$NAVDP_ENV_NAME" -r "$REPO_ROOT/baselines/navdp/requirements.txt"

CURRENT_STAGE="upgrade IsaacLab packaging tools"
pip_install "$ISAACLAB_ENV_NAME" --upgrade pip setuptools wheel
CURRENT_STAGE="install Isaac Sim 4.2.0.2"
pip_install "$ISAACLAB_ENV_NAME" \
  isaacsim==4.2.0.2 \
  isaacsim-extscache-physics==4.2.0.2 \
  isaacsim-extscache-kit==4.2.0.2 \
  isaacsim-extscache-kit-sdk==4.2.0.2 \
  --extra-index-url https://pypi.nvidia.com

install_isaaclab_checkout

install_benchmark_requirements

if ((SKIP_VERIFY)); then
  log "Runtime verification skipped"
else
  verify_installation
fi
export_snapshots

CURRENT_STAGE="complete"
log "Setup complete"
printf '\nRun the NavDP server:\n'
printf '  conda run -n %q python %q --port 8888 --checkpoint /path/to/navdp_checkpoint.ckpt\n' \
  "$NAVDP_ENV_NAME" "$REPO_ROOT/baselines/navdp/navdp_server.py"
printf '\nRun the IsaacLab smoke test again:\n'
printf '  conda run -n %q %q -p %q --headless\n' \
  "$ISAACLAB_ENV_NAME" "$ISAACLAB_DIR/isaaclab.sh" \
  "$ISAACLAB_DIR/source/standalone/tutorials/00_sim/create_empty.py"
