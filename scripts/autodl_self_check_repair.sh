#!/usr/bin/env bash
set -Euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTED_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
readonly AUTODL_STRICT_PROFILE=autodl-strict-history
readonly AUTODL_EXPECTED_REPO_ROOT=/root/NavDP
readonly AUTODL_EXPECTED_WORK_DIR=/root/autodl-tmp/navdp
readonly AUTODL_EXPECTED_ISAACLAB_DIR=/root/autodl-tmp/navdp/IsaacLab
readonly AUTODL_EXPECTED_CONDA_BIN=/root/miniconda3/bin/conda
readonly AUTODL_EXPECTED_CONDA_ENVS_PATH=/root/autodl-tmp/navdp/conda/envs
readonly AUTODL_EXPECTED_RUNTIME_ENV_FILE=/root/.config/navdp/autodl-runtime.env
readonly AUTODL_EXPECTED_BASHRC=/root/.bashrc

usage() {
  cat <<'EOF'
Usage: bash scripts/autodl_self_check_repair.sh [OPTIONS]

Diagnose and safely repair the NavDP AutoDL runtime without starting a real
experiment.

Production profile:
  autodl-strict-history   Requires the repository at /root/NavDP and uses the
                          fixed 2026-07-17 AutoDL installation paths. It never
                          falls back to local IsaacLab or Conda installations.

Options:
  --check-only             Diagnose without writing configuration or killing processes
  --kill-stale             Compatibility flag; stale cleanup is enabled by default
  --skip-smoke             Skip the Isaac headless GPU smoke test
  --skip-dry-run           Skip experiment command generation and validation
  --config PATH            Suite config (default: configs/experiments/full_suite.json)
  --report-dir PATH        Base report directory
  --smoke-timeout SECONDS  Isaac smoke timeout (default: 180)
  -h, --help               Show this help

Exit status:
  0  All required current checks passed
  1  One or more required current checks failed
  2  Invalid command-line arguments
  130  Interrupted by the user
EOF
}

cli_error() {
  printf 'error: %s\n' "$*" >&2
  usage >&2
  exit 2
}

if [[ -n "${AUTODL_REPAIR_REPO_ROOT:-}" &&
      "${AUTODL_REPAIR_TESTING:-0}" != 1 ]]; then
  cli_error "AUTODL_REPAIR_REPO_ROOT is only allowed with AUTODL_REPAIR_TESTING=1"
fi

REPO_ROOT="${AUTODL_REPAIR_REPO_ROOT:-$DETECTED_REPO_ROOT}"
if [[ -d "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
fi

CHECK_ONLY=0
SKIP_SMOKE=0
SKIP_DRY_RUN=0
SMOKE_TIMEOUT=180
CONFIG="$REPO_ROOT/configs/experiments/full_suite.json"
REPORT_BASE="$REPO_ROOT/results/autodl_self_check"

while (($#)); do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --kill-stale)
      # Retained for compatibility. Cleanup is already the default.
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    --skip-dry-run)
      SKIP_DRY_RUN=1
      shift
      ;;
    --config)
      (($# >= 2)) || cli_error "--config requires a path"
      [[ "$2" != --* ]] || cli_error "--config requires a path"
      CONFIG="$2"
      shift 2
      ;;
    --report-dir)
      (($# >= 2)) || cli_error "--report-dir requires a path"
      [[ "$2" != --* ]] || cli_error "--report-dir requires a path"
      REPORT_BASE="$2"
      shift 2
      ;;
    --smoke-timeout)
      (($# >= 2)) || cli_error "--smoke-timeout requires a positive integer"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] ||
        cli_error "--smoke-timeout requires a positive integer"
      SMOKE_TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      cli_error "unknown option: $1"
      ;;
    *)
      cli_error "unexpected argument: $1"
      ;;
  esac
done

NAVDP_ENV_NAME="${NAVDP_ENV_NAME:-navdp}"
ISAACLAB_ENV_NAME="${ISAACLAB_ENV_NAME:-isaaclab}"
REQUESTED_AUTODL_WORK_DIR="${AUTODL_WORK_DIR:-}"
REQUESTED_ISAACLAB_DIR="${ISAACLAB_DIR:-}"
REQUESTED_CONDA_BIN="${CONDA_BIN:-}"
REQUESTED_CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-}"
REQUESTED_RUNTIME_ENV_FILE="${NAVDP_RUNTIME_ENV_FILE:-}"

if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 ]]; then
  RUNTIME_PROFILE=testing
  AUTODL_WORK_DIR="${AUTODL_WORK_DIR:-$HOME/.navdp-autodl}"
  ISAACLAB_DIR="${ISAACLAB_DIR:-$AUTODL_WORK_DIR/IsaacLab}"
  CONDA_BIN="${CONDA_BIN:-}"
  CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$AUTODL_WORK_DIR/conda/envs}"
  NAVDP_RUNTIME_ENV_FILE="${NAVDP_RUNTIME_ENV_FILE:-$HOME/.config/navdp/autodl-runtime.env}"
  NAVDP_BASHRC_FILE="$HOME/.bashrc"
  NAVDP_VULKAN_ICD_DIRS="${NAVDP_VULKAN_ICD_DIRS:-/etc/vulkan/icd.d:/usr/share/vulkan/icd.d}"
else
  RUNTIME_PROFILE="$AUTODL_STRICT_PROFILE"
  AUTODL_WORK_DIR="$AUTODL_EXPECTED_WORK_DIR"
  ISAACLAB_DIR="$AUTODL_EXPECTED_ISAACLAB_DIR"
  CONDA_BIN="$AUTODL_EXPECTED_CONDA_BIN"
  CONDA_ENVS_PATH="$AUTODL_EXPECTED_CONDA_ENVS_PATH"
  NAVDP_RUNTIME_ENV_FILE="$AUTODL_EXPECTED_RUNTIME_ENV_FILE"
  NAVDP_BASHRC_FILE="$AUTODL_EXPECTED_BASHRC"
  NAVDP_VULKAN_ICD_DIRS=/etc/vulkan/icd.d:/usr/share/vulkan/icd.d
fi
NAVDP_RESULTS_ROOT="${NAVDP_RESULTS_ROOT:-}"
NAVDP_STALE_MIN_AGE_SECONDS="${NAVDP_STALE_MIN_AGE_SECONDS:-60}"

if [[ -d "$ISAACLAB_DIR" ]]; then
  ISAACLAB_DIR="$(cd "$ISAACLAB_DIR" && pwd -P)"
fi
if [[ -d "$AUTODL_WORK_DIR" ]]; then
  AUTODL_WORK_DIR="$(cd "$AUTODL_WORK_DIR" && pwd -P)"
fi

NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
VULKANINFO_BIN="${VULKANINFO_BIN:-vulkaninfo}"
PS_BIN="${PS_BIN:-ps}"
if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 && -z "${KILL_BIN:-}" ]]; then
  cli_error "KILL_BIN must be explicitly set when AUTODL_REPAIR_TESTING=1"
fi
if [[ -z "${KILL_BIN:-}" ]]; then
  KILL_BIN="$(type -P kill || true)"
elif [[ "$KILL_BIN" != */* ]]; then
  resolved_kill_bin="$(type -P "$KILL_BIN" || true)"
  [[ -z "$resolved_kill_bin" ]] || KILL_BIN="$resolved_kill_bin"
fi
if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 ]]; then
  kill_backend_real="$(readlink -f -- "${KILL_BIN:-}" 2>/dev/null || true)"
  if [[ "$(basename "${KILL_BIN:-kill}")" == kill ||
        "$kill_backend_real" == /bin/kill ||
        "$kill_backend_real" == /usr/bin/kill ]]; then
    cli_error "KILL_BIN must not resolve to the real system kill in test mode"
  fi
fi
SLEEP_BIN="${SLEEP_BIN:-sleep}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"

PASS_COUNT=0
REPAIR_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
SELECTED_ICD=""
REPORT_DIR=""
SMOKE_PID=""
SMOKE_GROUPED=0
SMOKE_PGID=""
SMOKE_SID=""
SMOKE_CLEANUP_REPORTED=0
DRY_RUN_PLAN=""
SUITE_OUTPUT_ROOT=""
SUITE_ID=""
RUNTIME_ENV_CHANGED=0
BASHRC_CHANGED=0

signal_backend() {
  if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 ]]; then
    if [[ -z "${FAKE_KILL_LOG:-}" ]]; then
      return 1
    fi
    printf '%s\n' "$*" >>"$FAKE_KILL_LOG" || return 1
    if [[ "${1:-}" == -0 ]]; then
      [[ "${FAKE_KILL_LIVENESS_FAIL:-0}" != 1 ]] || return 2
      [[ "${FAKE_KILL_SURVIVES:-0}" == 1 ]]
      return
    fi
    if [[ "${1:-}" == -TERM &&
          "${FAKE_KILL_TERM_ALREADY_EXITED:-0}" == 1 ]]; then
      return 1
    fi
    [[ "${FAKE_KILL_SIGNAL_FAIL:-0}" != 1 ]]
    return
  fi
  "$KILL_BIN" "$@"
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf '[PASS] %s\n' "$*"
}

repaired() {
  REPAIR_COUNT=$((REPAIR_COUNT + 1))
  printf '[REPAIRED] %s\n' "$*"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  printf '[WARN] %s\n' "$*"
}

fail_check() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf '[FAIL] %s\n' "$*" >&2
}

create_report_directory() {
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%S.%NZ)"
  if ! mkdir -p "$REPORT_BASE"; then
    printf 'error: cannot create report base directory: %s\n' "$REPORT_BASE" >&2
    return 1
  fi
  if ! REPORT_DIR="$(mktemp -d "$REPORT_BASE/${stamp}.XXXXXX")"; then
    printf 'error: cannot atomically create report directory under: %s\n' \
      "$REPORT_BASE" >&2
    return 1
  fi

  local report
  for report in \
    summary.txt \
    environment.txt \
    process-scan.txt \
    vulkan-before.txt \
    torch-cuda.txt \
    runtime-contract.txt \
    isaac-smoke.log \
    experiment-dry-run.txt \
    historical-diagnostics.txt; do
    if ! : >"$REPORT_DIR/$report"; then
      printf 'error: cannot initialize report file: %s\n' "$report" >&2
      return 1
    fi
  done
  if ! mkdir "$REPORT_DIR/vulkan-probes"; then
    printf 'error: cannot initialize Vulkan probe report directory\n' >&2
    return 1
  fi
}

write_final_summary() {
  [[ -n "$REPORT_DIR" ]] || return 0
  if ! {
    printf 'NavDP AutoDL self-check repair\n'
    printf 'UTC completed: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Profile: %s\n' "$RUNTIME_PROFILE"
    printf 'Repository: %s\n' "$REPO_ROOT"
    printf 'Config: %s\n' "$CONFIG"
    printf 'Mode: %s\n' "$([[ "$CHECK_ONLY" == 1 ]] && printf check-only || printf repair)"
    printf 'Passed checks: %d\n' "$PASS_COUNT"
    printf 'Repairs: %d\n' "$REPAIR_COUNT"
    printf 'Warnings: %d\n' "$WARN_COUNT"
    printf 'Failures: %d\n' "$FAIL_COUNT"
    printf 'Selected NVIDIA ICD: %s\n' "${SELECTED_ICD:-none}"
    printf 'Report: %s\n' "$REPORT_DIR"
    if ((FAIL_COUNT == 0)); then
      printf 'Final result: PASS\n'
    else
      printf 'Final result: FAIL\n'
    fi
  } >"$REPORT_DIR/summary.txt"; then
    printf 'error: unable to write final summary: %s\n' \
      "$REPORT_DIR/summary.txt" >&2
    return 1
  fi

  printf '\nChecks: %d passed, %d repaired, %d warnings, %d failures\n' \
    "$PASS_COUNT" "$REPAIR_COUNT" "$WARN_COUNT" "$FAIL_COUNT"
  printf 'Selected ICD: %s\n' "${SELECTED_ICD:-none}"
  printf 'Report: %s\n' "$REPORT_DIR"
  if ((FAIL_COUNT == 0)); then
    printf '[PASS] AutoDL self-check completed\n'
  else
    printf '[FAIL] AutoDL self-check completed with required failures\n' >&2
  fi
  return 0
}

read_proc_state() {
  local pid="$1" stat_line remainder
  [[ -r "/proc/$pid/stat" ]] || return 1
  IFS= read -r stat_line <"/proc/$pid/stat" || return 1
  remainder="${stat_line##*) }"
  [[ "$remainder" != "$stat_line" ]] || return 1
  printf '%s\n' "${remainder%% *}"
}

smoke_group_is_alive() {
  [[ -n "${SMOKE_PGID:-}" ]] || return 1
  signal_backend -0 -- "-$SMOKE_PGID" 2>/dev/null
}

smoke_leader_is_alive() {
  [[ -n "${SMOKE_PID:-}" ]] || return 1
  signal_backend -0 "$SMOKE_PID" 2>/dev/null
}

reap_smoke_leader_if_exited() {
  [[ -n "${SMOKE_PID:-}" ]] || return 0
  local state=""
  if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 ]]; then
    if smoke_leader_is_alive; then
      return 0
    fi
  elif state="$(read_proc_state "$SMOKE_PID" 2>/dev/null)"; then
    case "$state" in
      Z|X)
        ;;
      *)
        return 0
        ;;
    esac
  fi
  wait "$SMOKE_PID" 2>/dev/null || true
  SMOKE_PID=""
}

clear_smoke_identity() {
  SMOKE_PID=""
  SMOKE_PGID=""
  SMOKE_SID=""
  SMOKE_GROUPED=0
}

stop_smoke_group() {
  [[ -n "${SMOKE_PID:-}" || -n "${SMOKE_PGID:-}" ]] || return 0
  local cleanup_failed=0 attempt

  reap_smoke_leader_if_exited
  if smoke_group_is_alive; then
    if ! signal_backend -TERM -- "-$SMOKE_PGID" 2>/dev/null; then
      cleanup_failed=1
    fi
    for attempt in 1 2 3 4 5; do
      reap_smoke_leader_if_exited
      smoke_group_is_alive || break
      "$SLEEP_BIN" 0.2
    done
    reap_smoke_leader_if_exited
    if smoke_group_is_alive; then
      if ! signal_backend -KILL -- "-$SMOKE_PGID" 2>/dev/null; then
        cleanup_failed=1
      fi
      for attempt in 1 2 3 4 5; do
        reap_smoke_leader_if_exited
        smoke_group_is_alive || break
        "$SLEEP_BIN" 0.1
      done
    fi
    reap_smoke_leader_if_exited
    if smoke_group_is_alive; then
      cleanup_failed=1
    fi
  fi

  reap_smoke_leader_if_exited
  if [[ -n "${SMOKE_PID:-}" ]]; then
    if smoke_leader_is_alive; then
      cleanup_failed=1
    else
      wait "$SMOKE_PID" 2>/dev/null || true
      SMOKE_PID=""
    fi
  fi
  if ((cleanup_failed == 0)); then
    clear_smoke_identity
  fi
  ((cleanup_failed == 0))
}

record_smoke_cleanup_failure() {
  if ((SMOKE_CLEANUP_REPORTED == 0)); then
    SMOKE_CLEANUP_REPORTED=1
    [[ -z "$REPORT_DIR" ]] ||
      printf 'SMOKE_CLEANUP_FAILED pid=%s pgid=%s sid=%s\n' \
        "${SMOKE_PID:-none}" "${SMOKE_PGID:-none}" "${SMOKE_SID:-none}" \
        >>"$REPORT_DIR/isaac-smoke.log" 2>/dev/null || true
    fail_check "Failed to clean up Isaac smoke process group"
  fi
}

on_interrupt() {
  trap - EXIT HUP INT TERM
  if ! stop_smoke_group; then
    record_smoke_cleanup_failure
  fi
  warn "Interrupted by user"
  write_final_summary || true
  exit 130
}

on_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  if ! stop_smoke_group; then
    record_smoke_cleanup_failure
    write_final_summary || true
    if ((status == 0)); then
      status=1
    fi
  fi
  exit "$status"
}

trap on_interrupt HUP INT TERM
trap on_exit EXIT

resolve_conda() {
  if [[ -n "$CONDA_BIN" ]]; then
    command -v "$CONDA_BIN" >/dev/null 2>&1
    return
  fi
  if [[ -x /root/miniconda3/bin/conda ]]; then
    CONDA_BIN=/root/miniconda3/bin/conda
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  else
    return 1
  fi
}

require_tool() {
  local label="$1"
  local command_name="$2"
  if command -v "$command_name" >/dev/null 2>&1; then
    return 0
  fi
  printf 'MISSING_COMMAND %s (%s)\n' "$label" "$command_name" \
    >>"$REPORT_DIR/environment.txt"
  fail_check "Required command not found: $label"
  return 1
}

preflight() {
  local failed=0 tool
  {
    printf 'Profile: %s\n' "$RUNTIME_PROFILE"
    printf 'Expected repository: %s\n' "$AUTODL_EXPECTED_REPO_ROOT"
    printf 'Actual repository: %s\n' "$REPO_ROOT"
    printf 'Repository: %s\n' "$REPO_ROOT"
    printf 'AutoDL work directory: %s\n' "$AUTODL_WORK_DIR"
    printf 'IsaacLab directory: %s\n' "$ISAACLAB_DIR"
    printf 'Conda: %s\n' "$CONDA_BIN"
    printf 'Conda envs path: %s\n' "$CONDA_ENVS_PATH"
    printf 'Runtime environment: %s\n' "$NAVDP_RUNTIME_ENV_FILE"
    printf 'Bashrc: %s\n' "$NAVDP_BASHRC_FILE"
    printf 'Config: %s\n' "$CONFIG"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    printf 'NVIDIA_VISIBLE_DEVICES=%s\n' "${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    printf 'NVIDIA_DRIVER_CAPABILITIES=%s\n' "${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
  } >>"$REPORT_DIR/environment.txt"

  if [[ "$RUNTIME_PROFILE" == "$AUTODL_STRICT_PROFILE" ]]; then
    if [[ "$REPO_ROOT" != "$AUTODL_EXPECTED_REPO_ROOT" ]]; then
      fail_check \
        "Strict AutoDL repository mismatch: expected $AUTODL_EXPECTED_REPO_ROOT, actual $REPO_ROOT"
      failed=1
    fi
    if [[ -n "$REQUESTED_AUTODL_WORK_DIR" &&
          "$REQUESTED_AUTODL_WORK_DIR" != "$AUTODL_EXPECTED_WORK_DIR" ]]; then
      fail_check "Strict AutoDL path override rejected: AUTODL_WORK_DIR"
      failed=1
    fi
    if [[ -n "$REQUESTED_ISAACLAB_DIR" &&
          "$REQUESTED_ISAACLAB_DIR" != "$AUTODL_EXPECTED_ISAACLAB_DIR" ]]; then
      fail_check "Strict AutoDL path override rejected: ISAACLAB_DIR"
      failed=1
    fi
    if [[ -n "$REQUESTED_CONDA_BIN" &&
          "$REQUESTED_CONDA_BIN" != "$AUTODL_EXPECTED_CONDA_BIN" ]]; then
      fail_check "Strict AutoDL path override rejected: CONDA_BIN"
      failed=1
    fi
    if [[ -n "$REQUESTED_CONDA_ENVS_PATH" &&
          "$REQUESTED_CONDA_ENVS_PATH" != "$AUTODL_EXPECTED_CONDA_ENVS_PATH" ]]; then
      fail_check "Strict AutoDL path override rejected: CONDA_ENVS_PATH"
      failed=1
    fi
    if [[ -n "$REQUESTED_RUNTIME_ENV_FILE" &&
          "$REQUESTED_RUNTIME_ENV_FILE" != "$AUTODL_EXPECTED_RUNTIME_ENV_FILE" ]]; then
      fail_check "Strict AutoDL path override rejected: NAVDP_RUNTIME_ENV_FILE"
      failed=1
    fi
  fi

  if [[ "$(uname -s 2>/dev/null || true)" != Linux ]]; then
    fail_check "This self-check supports Linux hosts only"
    failed=1
  fi
  [[ -d "$REPO_ROOT" ]] || {
    fail_check "Repository root does not exist: $REPO_ROOT"
    failed=1
  }
  [[ -d "$REPORT_DIR" && -w "$REPORT_DIR" ]] || {
    fail_check "Report directory is not writable: $REPORT_DIR"
    failed=1
  }

  if [[ ! "$NAVDP_STALE_MIN_AGE_SECONDS" =~ ^[0-9]+$ ]]; then
    fail_check "NAVDP_STALE_MIN_AGE_SECONDS must be a non-negative integer"
    failed=1
  fi

  for tool in \
    bash awk grep sed find pgrep sha256sum cmp cp mktemp mv chmod readlink \
    python3 flock setsid stat; do
    require_tool "$tool" "$tool" || failed=1
  done
  require_tool ps "$PS_BIN" || failed=1
  require_tool nvidia-smi "$NVIDIA_SMI_BIN" || failed=1
  require_tool vulkaninfo "$VULKANINFO_BIN" || failed=1
  require_tool timeout "$TIMEOUT_BIN" || failed=1
  if [[ "${AUTODL_REPAIR_TESTING:-0}" != 1 ]]; then
    require_tool kill "$KILL_BIN" || failed=1
  fi
  require_tool sleep "$SLEEP_BIN" || failed=1
  if ! resolve_conda; then
    fail_check "Conda was not found; run: bash scripts/setup_autodl.sh"
    failed=1
  else
    printf 'Conda: %s\n' "$CONDA_BIN" >>"$REPORT_DIR/environment.txt"
  fi

  local required_file
  for required_file in \
    "$CONFIG" \
    "$REPO_ROOT/run_scripts/eval_pointgoal_wheeled.py" \
    "$REPO_ROOT/experiments/simulators/isaac_navdp_backend.py" \
    "$REPO_ROOT/baselines/navdp/navdp_server.py" \
    "$ISAACLAB_DIR/isaaclab.sh" \
    "$ISAACLAB_DIR/source/standalone/tutorials/00_sim/create_empty.py"; do
    if [[ ! -f "$required_file" ]]; then
      printf 'MISSING_FILE %s\n' "$required_file" >>"$REPORT_DIR/environment.txt"
      fail_check "Required installation component missing: $required_file; run: bash scripts/setup_autodl.sh"
      failed=1
    fi
  done

  if ((failed == 0)); then
    pass "Preflight checks passed"
    return 0
  fi
  return 1
}

declare -A SNAPSHOT_PPID=()
declare -A SNAPSHOT_PGID=()
declare -A SNAPSHOT_ARGS=()
declare -A SNAPSHOT_ETIMES=()
declare -A SNAPSHOT_STARTTIME=()
declare -A ANCESTOR_PID=()
declare -A STALE_PID=()
declare -a SNAPSHOT_PIDS=()
declare -a STALE_PIDS=()

read_proc_starttime() {
  local pid="$1" stat_line rest
  local -a fields=()
  [[ -r "/proc/$pid/stat" ]] || return 1
  if ! IFS= read -r stat_line <"/proc/$pid/stat"; then
    return 1
  fi
  [[ "$stat_line" == *") "* ]] || return 1
  rest="${stat_line##*) }"
  read -r -a fields <<<"$rest"
  ((${#fields[@]} > 19)) || return 1
  [[ "${fields[19]}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${fields[19]}"
}

capture_stale_starttime() {
  local pid="$1" starttime
  if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 ]]; then
    SNAPSHOT_STARTTIME[$pid]="TEST"
    return 0
  fi
  if starttime="$(read_proc_starttime "$pid")"; then
    SNAPSHOT_STARTTIME[$pid]="$starttime"
    return 0
  fi
  SNAPSHOT_STARTTIME[$pid]=""
  return 1
}

collect_ancestor_pids() {
  ANCESTOR_PID=()
  ANCESTOR_PID[1]=1
  ANCESTOR_PID[$$]=1
  ANCESTOR_PID[$PPID]=1
  local current="$$" parent guard=0
  while ((guard < 128)); do
    parent="${SNAPSHOT_PPID[$current]:-}"
    [[ "$parent" =~ ^[0-9]+$ && "$parent" != 0 ]] || break
    ANCESTOR_PID[$parent]=1
    [[ "$parent" == 1 || "$parent" == "$current" ]] && break
    current="$parent"
    guard=$((guard + 1))
  done
}

python_script_is_target() {
  local script="$1"
  [[ "$script" == "$REPO_ROOT/run_scripts/eval_pointgoal_wheeled.py" ||
     "$script" == "$REPO_ROOT/baselines/navdp/navdp_server.py" ||
     "$script" == "$ISAACLAB_DIR"/*.py ||
     "$script" == "$ISAACLAB_DIR"/*/*.py ]]
}

python_argv_matches_target() {
  local -a argv=("$@")
  local index=1 token
  while ((index < ${#argv[@]})); do
    token="${argv[$index]}"
    case "$token" in
      --)
        index=$((index + 1))
        break
        ;;
      -c|-c*|-m|-m*)
        return 1
        ;;
      -W|-X|--check-hash-based-pycs)
        index=$((index + 2))
        ;;
      -*)
        index=$((index + 1))
        ;;
      *)
        break
        ;;
    esac
  done
  ((index < ${#argv[@]})) || return 1
  python_script_is_target "${argv[$index]}"
}

shell_argv_matches_target() {
  local -a argv=("$@")
  local index=1 token script
  while ((index < ${#argv[@]})); do
    token="${argv[$index]}"
    case "$token" in
      --)
        index=$((index + 1))
        break
        ;;
      -c|-c*|-O|-s|-r|-v|-x)
        return 1
        ;;
      -*)
        index=$((index + 1))
        ;;
      *)
        break
        ;;
    esac
  done
  ((index < ${#argv[@]})) || return 1
  script="${argv[$index]}"
  [[ "$script" == "$ISAACLAB_DIR/isaaclab.sh" ||
     "$script" == "$ISAACLAB_DIR"/*.sh ||
     "$script" == "$ISAACLAB_DIR"/*/*.sh ]]
}

execution_argv_matches_target() {
  local -a argv=("$@")
  ((${#argv[@]} > 0)) || return 1
  local executable="${argv[0]}" executable_base="${argv[0]##*/}"
  case "$executable_base" in
    python|python[0-9]*)
      python_argv_matches_target "${argv[@]}"
      ;;
    bash|sh)
      shell_argv_matches_target "${argv[@]}"
      ;;
    kit|omni.kit|isaac-sim|isaac-sim.sh)
      [[ "$executable" == "$ISAACLAB_DIR"/* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

conda_argv_matches_target() {
  local -a argv=("$@")
  local index=1 token saw_run=0
  while ((index < ${#argv[@]})); do
    token="${argv[$index]}"
    if ((saw_run == 0)); then
      if [[ "$token" == run ]]; then
        saw_run=1
      fi
      index=$((index + 1))
      continue
    fi
    case "$token" in
      --)
        index=$((index + 1))
        break
        ;;
      -n|--name|-p|--prefix|--cwd)
        index=$((index + 2))
        ;;
      --no-capture-output|--live-stream|--debug-wrapper-scripts|--dev)
        index=$((index + 1))
        ;;
      -*)
        return 1
        ;;
      *)
        break
        ;;
    esac
  done
  ((saw_run == 1 && index < ${#argv[@]})) || return 1
  execution_argv_matches_target "${argv[@]:index}"
}

is_stale_target_command() {
  local args="$1"
  local -a argv=()
  read -r -a argv <<<"$args"
  ((${#argv[@]} > 0)) || return 1
  case "${argv[0]##*/}" in
    conda)
      conda_argv_matches_target "${argv[@]}"
      ;;
    *)
      execution_argv_matches_target "${argv[@]}"
      ;;
  esac
}

is_sensitive_process_option() {
  local option="${1,,}"
  option="${option%%=*}"
  [[ "$option" == -* ]] || return 1
  [[ "$option" =~ (^|[-_])(token|password|passwd|secret|key|api[-_]?key|access[-_]?key|private[-_]?key|authorization|credential|cookie|bearer)([-_]|$) ]]
}

redact_process_command() {
  local args="$1" word name lower_word redact_next=0
  local -a argv=() redacted=()
  read -r -a argv <<<"$args"
  for word in "${argv[@]}"; do
    if ((redact_next)); then
      redacted+=("<redacted>")
      redact_next=0
      continue
    fi
    lower_word="${word,,}"
    if [[ "$word" == *\?* &&
          "$lower_word" =~ (\?|&)(token|password|passwd|secret|key|api[-_]?key|access[-_]?key|authorization|credential)= ]]; then
      redacted+=("${word%%\?*}?<redacted-query>")
      continue
    fi
    if [[ "$word" == *=* ]]; then
      name="${word%%=*}"
      if is_sensitive_process_option "$name" ||
         [[ "$name" =~ (^|[-_])(TOKEN|PASSWORD|PASSWD|SECRET|KEY|API_KEY|ACCESS_KEY|PRIVATE_KEY)$ ]]; then
        redacted+=("$name=<redacted>")
        continue
      fi
    fi
    redacted+=("$word")
    if is_sensitive_process_option "$word"; then
      redact_next=1
    fi
  done
  if ((redact_next)); then
    redacted+=("<missing-redacted-value>")
  fi
  printf '%s' "${redacted[*]}"
}

order_stale_pids_leaf_first() {
  local pid current parent depth max_depth=0 target_depth
  local -A depths=()
  local -a ordered=()
  for pid in "${STALE_PIDS[@]}"; do
    current="$pid"
    depth=0
    while :; do
      parent="${SNAPSHOT_PPID[$current]:-}"
      [[ "$parent" =~ ^[0-9]+$ &&
         -n "${STALE_PID[$parent]:-}" ]] || break
      depth=$((depth + 1))
      current="$parent"
      ((depth <= ${#STALE_PIDS[@]})) || return 1
    done
    depths[$pid]="$depth"
    ((depth <= max_depth)) || max_depth="$depth"
  done
  for ((target_depth = max_depth; target_depth >= 0; target_depth--)); do
    for pid in "${STALE_PIDS[@]}"; do
      if ((depths[$pid] == target_depth)); then
        ordered+=("$pid")
      fi
    done
  done
  STALE_PIDS=("${ordered[@]}")
}

scan_stale_processes() {
  SNAPSHOT_PPID=()
  SNAPSHOT_PGID=()
  SNAPSHOT_ARGS=()
  SNAPSHOT_ETIMES=()
  SNAPSHOT_STARTTIME=()
  SNAPSHOT_PIDS=()
  STALE_PID=()
  STALE_PIDS=()
  local snapshot pid ppid pgid sid etimes args redacted_args
  if ! snapshot="$("$PS_BIN" -eo pid=,ppid=,pgid=,sid=,etimes=,args= 2>&1)"; then
    printf 'PROCESS_SCAN_ERROR\n' >>"$REPORT_DIR/process-scan.txt"
    fail_check "Unable to read process snapshot"
    return 1
  fi
  while read -r pid ppid pgid sid etimes args; do
    [[ "$pid" =~ ^[0-9]+$ && "$ppid" =~ ^[0-9]+$ &&
       "$pgid" =~ ^[0-9]+$ ]] || continue
    SNAPSHOT_PIDS+=("$pid")
    SNAPSHOT_PPID[$pid]="$ppid"
    SNAPSHOT_PGID[$pid]="$pgid"
    SNAPSHOT_ETIMES[$pid]="$etimes"
    SNAPSHOT_ARGS[$pid]="$args"
  done <<<"$snapshot"

  collect_ancestor_pids
  # Only a matching target process adopted by init is a stale root. Matching
  # descendants are added by following the PPID graph from those roots.
  for pid in "${SNAPSHOT_PIDS[@]}"; do
    [[ -z "${ANCESTOR_PID[$pid]:-}" ]] || continue
    if [[ "${SNAPSHOT_PPID[$pid]}" == 1 ]] &&
       is_stale_target_command "${SNAPSHOT_ARGS[$pid]}"; then
      if [[ ! "${SNAPSHOT_ETIMES[$pid]}" =~ ^[0-9]+$ ]]; then
        printf 'ORPHAN_AGE_UNKNOWN_EXCLUDED pid=%s age=%q threshold=%s\n' \
          "$pid" "${SNAPSHOT_ETIMES[$pid]}" "$NAVDP_STALE_MIN_AGE_SECONDS" \
          >>"$REPORT_DIR/process-scan.txt"
        continue
      fi
      if ((SNAPSHOT_ETIMES[$pid] < NAVDP_STALE_MIN_AGE_SECONDS)); then
        printf 'YOUNG_ORPHAN_EXCLUDED pid=%s age=%s threshold=%s\n' \
          "$pid" "${SNAPSHOT_ETIMES[$pid]}" "$NAVDP_STALE_MIN_AGE_SECONDS" \
          >>"$REPORT_DIR/process-scan.txt"
        continue
      fi
      STALE_PID[$pid]=1
      STALE_PIDS+=("$pid")
      capture_stale_starttime "$pid" || true
      redacted_args="$(redact_process_command "${SNAPSHOT_ARGS[$pid]}")"
      printf 'STALE_ROOT pid=%s ppid=%s pgid=%s command=%q\n' \
        "$pid" "${SNAPSHOT_PPID[$pid]}" "${SNAPSHOT_PGID[$pid]}" \
        "$redacted_args" >>"$REPORT_DIR/process-scan.txt"
    fi
  done
  local changed=1
  while ((changed)); do
    changed=0
    for pid in "${SNAPSHOT_PIDS[@]}"; do
      [[ -z "${STALE_PID[$pid]:-}" ]] || continue
      [[ -n "${STALE_PID[${SNAPSHOT_PPID[$pid]}]:-}" ]] || continue
      if is_stale_target_command "${SNAPSHOT_ARGS[$pid]}"; then
        STALE_PID[$pid]=1
        STALE_PIDS+=("$pid")
        capture_stale_starttime "$pid" || true
        changed=1
        redacted_args="$(redact_process_command "${SNAPSHOT_ARGS[$pid]}")"
        printf 'STALE_DESCENDANT pid=%s ppid=%s pgid=%s command=%q\n' \
          "$pid" "${SNAPSHOT_PPID[$pid]}" "${SNAPSHOT_PGID[$pid]}" \
          "$redacted_args" >>"$REPORT_DIR/process-scan.txt"
      fi
    done
  done
  if ! order_stale_pids_leaf_first; then
    fail_check "Unable to establish a safe leaf-first stale process order"
    return 1
  fi
  return 0
}

current_process_identity() {
  local target_pid="$1" snapshot pid ppid pgid sid etimes args
  if ! snapshot="$("$PS_BIN" -eo pid=,ppid=,pgid=,sid=,etimes=,args= 2>/dev/null)"; then
    return 3
  fi
  while read -r pid ppid pgid sid etimes args; do
    [[ "$pid" == "$target_pid" ]] || continue
    printf '%s\t%s\t%s\n' "$ppid" "$pgid" "$args"
    return 0
  done <<<"$snapshot"
  return 1
}

stale_identity_is_current() {
  local pid="$1" identity expected status
  if identity="$(current_process_identity "$pid")"; then
    :
  else
    status=$?
    return "$status"
  fi
  expected="${SNAPSHOT_PPID[$pid]}"$'\t'"${SNAPSHOT_PGID[$pid]}"$'\t'"${SNAPSHOT_ARGS[$pid]}"
  [[ "$identity" == "$expected" ]] || return 2
}

pidfd_stale_action() {
  local pid="$1" action="$2" expected="${SNAPSHOT_STARTTIME[$pid]:-}"
  if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 ]]; then
    case "$action" in
      0) signal_backend -0 "$pid" ;;
      TERM) signal_backend -TERM "$pid" ;;
      KILL) signal_backend -KILL "$pid" ;;
      *) return 2 ;;
    esac
    return
  fi
  [[ "$expected" =~ ^[0-9]+$ ]] || return 3
  python3 - "$pid" "$expected" "$action" <<'PY'
import os
import signal
import sys

pid = int(sys.argv[1])
expected = sys.argv[2]
action = sys.argv[3]


def starttime(target):
    with open(f"/proc/{target}/stat", encoding="utf-8") as handle:
        line = handle.read()
    close = line.rfind(")")
    if close < 0:
        raise RuntimeError("malformed proc stat")
    fields = line[close + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("short proc stat")
    return fields[19]


if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
    raise SystemExit(20)

try:
    fd = os.pidfd_open(pid, 0)
except ProcessLookupError:
    raise SystemExit(1)
except PermissionError:
    raise SystemExit(4)
except OSError:
    raise SystemExit(20)

try:
    if starttime(pid) != expected:
        raise SystemExit(3)
    signum = {"0": 0, "TERM": signal.SIGTERM, "KILL": signal.SIGKILL}[action]
    signal.pidfd_send_signal(fd, signum, None, 0)
except ProcessLookupError:
    raise SystemExit(1)
except PermissionError:
    raise SystemExit(4)
except OSError:
    raise SystemExit(4)
finally:
    os.close(fd)
PY
  local status=$?
  return "$status"
}

signal_stale_processes() {
  local count="${#STALE_PIDS[@]}"
  if ((count == 0)); then
    pass "No strictly matched stale NavDP/Isaac processes"
    return 0
  fi
  if [[ "$CHECK_ONLY" == 1 ]]; then
    warn "Check-only: found $count stale NavDP/Isaac process(es); no signals sent"
    return 0
  fi

  local pid attempt alive identity_status liveness_status signal_status failed=0
  for pid in "${STALE_PIDS[@]}"; do
    if stale_identity_is_current "$pid"; then
      printf 'SIGNAL TERM pid=%s\n' "$pid" >>"$REPORT_DIR/process-scan.txt"
      if pidfd_stale_action "$pid" TERM 2>>"$REPORT_DIR/process-scan.txt"; then
        :
      else
        signal_status=$?
        if ((signal_status == 1)); then
          printf 'ALREADY_EXITED_BEFORE_TERM pid=%s\n' \
            "$pid" >>"$REPORT_DIR/process-scan.txt"
        else
          printf 'SIGNAL_FAILED TERM pid=%s status=%s\n' \
            "$pid" "$signal_status" >>"$REPORT_DIR/process-scan.txt"
          failed=1
        fi
      fi
    else
      identity_status=$?
      if ((identity_status == 1)); then
        printf 'ALREADY_EXITED pid=%s\n' "$pid" >>"$REPORT_DIR/process-scan.txt"
      else
        printf 'IDENTITY_CHANGED_BEFORE_TERM pid=%s\n' "$pid" >>"$REPORT_DIR/process-scan.txt"
        failed=1
      fi
    fi
  done

  for attempt in 1 2 3 4 5; do
    alive=0
    for pid in "${STALE_PIDS[@]}"; do
      if pidfd_stale_action "$pid" 0 2>/dev/null; then
        alive=1
      else
        liveness_status=$?
        if ((liveness_status != 1)); then
          printf 'LIVENESS_UNKNOWN pid=%s status=%s\n' \
            "$pid" "$liveness_status" >>"$REPORT_DIR/process-scan.txt"
          failed=1
        fi
      fi
    done
    ((alive == 0)) && break
    "$SLEEP_BIN" 1
  done
  for pid in "${STALE_PIDS[@]}"; do
    if pidfd_stale_action "$pid" 0 2>/dev/null; then
      if stale_identity_is_current "$pid"; then
        printf 'SIGNAL KILL pid=%s\n' "$pid" >>"$REPORT_DIR/process-scan.txt"
        if ! pidfd_stale_action "$pid" KILL 2>>"$REPORT_DIR/process-scan.txt"; then
          printf 'SIGNAL_FAILED KILL pid=%s\n' "$pid" >>"$REPORT_DIR/process-scan.txt"
          failed=1
        fi
      else
        identity_status=$?
        printf 'IDENTITY_CHANGED_BEFORE_KILL pid=%s status=%s\n' \
          "$pid" "$identity_status" >>"$REPORT_DIR/process-scan.txt"
        failed=1
      fi
    else
      liveness_status=$?
      if ((liveness_status != 1)); then
        printf 'LIVENESS_UNKNOWN_BEFORE_KILL pid=%s status=%s\n' \
          "$pid" "$liveness_status" >>"$REPORT_DIR/process-scan.txt"
        failed=1
      fi
    fi
  done
  for attempt in 1 2 3 4 5; do
    alive=0
    for pid in "${STALE_PIDS[@]}"; do
      if pidfd_stale_action "$pid" 0 2>/dev/null; then
        alive=1
      else
        liveness_status=$?
        if ((liveness_status != 1)); then
          printf 'LIVENESS_UNKNOWN_AFTER_KILL pid=%s status=%s\n' \
            "$pid" "$liveness_status" >>"$REPORT_DIR/process-scan.txt"
          failed=1
        fi
      fi
    done
    ((alive == 0)) && break
    "$SLEEP_BIN" 0.2
  done
  for pid in "${STALE_PIDS[@]}"; do
    if pidfd_stale_action "$pid" 0 2>/dev/null; then
      printf 'STILL_ALIVE pid=%s\n' "$pid" >>"$REPORT_DIR/process-scan.txt"
      failed=1
    else
      liveness_status=$?
      if ((liveness_status != 1)); then
        printf 'FINAL_LIVENESS_UNKNOWN pid=%s status=%s\n' \
          "$pid" "$liveness_status" >>"$REPORT_DIR/process-scan.txt"
        failed=1
      fi
    fi
  done
  if ((failed)); then
    if grep -Fq 'LIVENESS_UNKNOWN' "$REPORT_DIR/process-scan.txt"; then
      fail_check "Unable to verify stale process liveness safely"
    fi
    fail_check "Failed to terminate stale NavDP/Isaac process(es)"
    return 1
  fi
  repaired "Terminated $count strictly matched stale NavDP/Isaac process(es)"
}

record_gpu_environment() {
  {
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    printf 'NVIDIA_VISIBLE_DEVICES=%s\n' "${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    printf 'NVIDIA_DRIVER_CAPABILITIES=%s\n' "${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
    printf '\nGPU inventory:\n'
  } >>"$REPORT_DIR/environment.txt"
  if "$NVIDIA_SMI_BIN" \
      --query-gpu=name,driver_version,memory.total,utilization.gpu \
      --format=csv,noheader >>"$REPORT_DIR/environment.txt" 2>&1 &&
     "$NVIDIA_SMI_BIN" >>"$REPORT_DIR/environment.txt" 2>&1; then
    pass "NVIDIA GPU inventory recorded"
  else
    fail_check "nvidia-smi failed; verify container NVIDIA device access"
  fi

  if "$TIMEOUT_BIN" 30 "$VULKANINFO_BIN" --summary \
      >"$REPORT_DIR/vulkan-before.txt" 2>&1; then
    if awk '
      BEGIN { count = 0 }
      tolower($0) ~ /devicename/ && tolower($0) ~ /nvidia/ { count++ }
      END { exit !(count > 1) }
    ' "$REPORT_DIR/vulkan-before.txt"; then
      warn "Original Vulkan environment enumerates duplicate NVIDIA devices"
    else
      pass "Original Vulkan summary recorded"
    fi
  else
    warn "Original vulkaninfo failed; individual NVIDIA ICD probes will continue"
  fi
}

declare -a NVIDIA_ICD_CANDIDATES=()

discover_nvidia_icds() {
  NVIDIA_ICD_CANDIDATES=()
  local dir candidate
  local -a dirs=()
  IFS=: read -r -a dirs <<<"$NAVDP_VULKAN_ICD_DIRS"
  for dir in "${dirs[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r candidate; do
      [[ -n "$candidate" ]] && NVIDIA_ICD_CANDIDATES+=("$candidate")
    done < <(find "$dir" -maxdepth 1 -type f -iname '*nvidia*.json' -print | sort)
  done
}

probe_nvidia_icd() {
  local candidate="$1"
  local index="$2"
  local output="$REPORT_DIR/vulkan-probes/$(printf '%03d' "$index")-$(basename "$candidate").txt"
  local status=0 nvidia_count llvmpipe_count
  {
    printf 'candidate=%s\n' "$candidate"
    printf 'VK_ICD_FILENAMES=%s\n' "$candidate"
    printf 'VK_DRIVER_FILES=%s\n\n' "$candidate"
  } >"$output"
  if VK_ICD_FILENAMES="$candidate" VK_DRIVER_FILES="$candidate" \
      "$TIMEOUT_BIN" 30 "$VULKANINFO_BIN" --summary >>"$output" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '\nexit_status=%s\n' "$status" >>"$output"
  ((status == 0)) || return 1

  nvidia_count="$(awk '
    BEGIN { count = 0 }
    tolower($0) ~ /devicename/ && tolower($0) ~ /nvidia/ { count++ }
    END { print count }
  ' "$output")"
  llvmpipe_count="$(awk '
    BEGIN { count = 0 }
    tolower($0) ~ /devicename/ && tolower($0) ~ /llvmpipe/ { count++ }
    END { print count }
  ' "$output")"
  [[ "$nvidia_count" == 1 && "$llvmpipe_count" == 0 ]]
}

write_runtime_environment() {
  local env_dir temp backup_stamp
  RUNTIME_ENV_CHANGED=0
  env_dir="$(dirname "$NAVDP_RUNTIME_ENV_FILE")"
  [[ ! -L "$NAVDP_RUNTIME_ENV_FILE" ]] || return 2
  if [[ -e "$NAVDP_RUNTIME_ENV_FILE" && ! -f "$NAVDP_RUNTIME_ENV_FILE" ]]; then
    return 2
  fi
  if ! mkdir -p "$env_dir"; then
    return 1
  fi
  if ! temp="$(mktemp "$env_dir/.autodl-runtime.env.XXXXXX")"; then
    return 1
  fi
  if ! {
    printf '# Managed by NavDP scripts/autodl_self_check_repair.sh\n'
    printf 'export VK_ICD_FILENAMES=%q\n' "$SELECTED_ICD"
    printf 'export VK_DRIVER_FILES=%q\n' "$SELECTED_ICD"
    printf 'export CONDA_ENVS_PATH=%q\n' "$CONDA_ENVS_PATH"
    printf 'export CONDA_BIN=%q\n' "$CONDA_BIN"
    printf 'export AUTODL_WORK_DIR=%q\n' "$AUTODL_WORK_DIR"
    printf 'export ISAACLAB_DIR=%q\n' "$ISAACLAB_DIR"
    printf 'export NAVDP_REPO_ROOT=%q\n' "$REPO_ROOT"
  } >"$temp"; then
    command rm -f "$temp"
    return 1
  fi
  if ! chmod 600 "$temp"; then
    command rm -f "$temp"
    return 1
  fi
  if [[ -f "$NAVDP_RUNTIME_ENV_FILE" ]] &&
     cmp -s "$temp" "$NAVDP_RUNTIME_ENV_FILE"; then
    command rm -f "$temp"
    return 0
  fi
  if [[ -e "$NAVDP_RUNTIME_ENV_FILE" ]]; then
    backup_stamp="$(date -u +%Y%m%dT%H%M%S).$$"
    if ! cp -p "$NAVDP_RUNTIME_ENV_FILE" \
        "$NAVDP_RUNTIME_ENV_FILE.before-navdp-autodl.$backup_stamp"; then
      command rm -f "$temp"
      return 1
    fi
  fi
  if ! mv "$temp" "$NAVDP_RUNTIME_ENV_FILE"; then
    command rm -f "$temp"
    return 1
  fi
  RUNTIME_ENV_CHANGED=1
}

validate_bashrc_runtime_markers() {
  [[ "${NAVDP_SKIP_BASHRC_UPDATE:-0}" == 1 ]] && return 0
  local bashrc="$NAVDP_BASHRC_FILE"
  [[ ! -L "$bashrc" ]] || return 4
  [[ -e "$bashrc" ]] || return 0
  [[ -f "$bashrc" ]] || return 1
  local begin_count end_count
  begin_count="$(grep -Fxc '# >>> NavDP AutoDL runtime >>>' "$bashrc" || true)"
  end_count="$(grep -Fxc '# <<< NavDP AutoDL runtime <<<' "$bashrc" || true)"
  if [[ "$begin_count" == 0 && "$end_count" == 0 ]]; then
    return 0
  fi
  [[ "$begin_count" == 1 && "$end_count" == 1 ]] || return 1
  awk '
    $0 == "# >>> NavDP AutoDL runtime >>>" {
      if (seen_begin || seen_end) exit 1
      seen_begin = 1
      next
    }
    $0 == "# <<< NavDP AutoDL runtime <<<" {
      if (!seen_begin || seen_end) exit 1
      seen_end = 1
    }
    END {
      if (!seen_begin || !seen_end) exit 1
    }
  ' "$bashrc"
}

update_bashrc() {
  BASHRC_CHANGED=0
  [[ "${NAVDP_SKIP_BASHRC_UPDATE:-0}" == 1 ]] && return 0
  local bashrc="$NAVDP_BASHRC_FILE"
  local lock_file="${NAVDP_BASHRC_FILE}.navdp.lock"
  local temp="" quoted backup_stamp canonical_backup
  local original_meta original_hash current_meta current_hash
  local lock_fd status=0
  if ! mkdir -p "$(dirname "$bashrc")"; then
    return 1
  fi
  [[ ! -L "$lock_file" ]] || return 4
  if ! exec {lock_fd}>"$lock_file"; then
    return 1
  fi
  if ! flock -x "$lock_fd"; then
    exec {lock_fd}>&-
    return 1
  fi

  while true; do
    if [[ -L "$bashrc" ]]; then
      status=4
      break
    fi
    if [[ ! -e "$bashrc" ]] && ! : >"$bashrc"; then
      status=1
      break
    fi
    if validate_bashrc_runtime_markers; then
      :
    else
      status=$?
      ((status == 0)) && status=2
      break
    fi
    if ! original_meta="$(stat -c '%d:%i' -- "$bashrc")" ||
       ! original_hash="$(sha256sum "$bashrc" | awk '{print $1}')"; then
      status=1
      break
    fi
    if ! temp="$(mktemp "$(dirname "$bashrc")/.bashrc.navdp.XXXXXX")"; then
      status=1
      break
    fi
    if ! awk '
      $0 == "# >>> NavDP AutoDL runtime >>>" { managed = 1; next }
      $0 == "# <<< NavDP AutoDL runtime <<<" { managed = 0; next }
      !managed { print }
    ' "$bashrc" >"$temp"; then
      status=1
      break
    fi
    printf -v quoted '%q' "$NAVDP_RUNTIME_ENV_FILE"
    if ! {
      printf '# >>> NavDP AutoDL runtime >>>\n'
      printf 'source %s\n' "$quoted"
      printf '# <<< NavDP AutoDL runtime <<<\n'
    } >>"$temp"; then
      status=1
      break
    fi
    if cmp -s "$temp" "$bashrc"; then
      status=0
      break
    fi

    if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 &&
          -n "${AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK:-}" ]]; then
      if ! "$AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK"; then
        status=1
        break
      fi
    fi
    if [[ -L "$bashrc" ]] ||
       ! current_meta="$(stat -c '%d:%i' -- "$bashrc")" ||
       ! current_hash="$(sha256sum "$bashrc" | awk '{print $1}')"; then
      status=4
      break
    fi
    if [[ "$current_meta" != "$original_meta" ||
          "$current_hash" != "$original_hash" ]]; then
      status=3
      break
    fi

    backup_stamp="$(date -u +%Y%m%dT%H%M%S).$$"
    canonical_backup="$bashrc.before-navdp-autodl"
    if [[ ! -e "$canonical_backup" ]] &&
       ! cp -p "$bashrc" "$canonical_backup"; then
      status=1
      break
    fi
    if ! cp -p "$bashrc" "$bashrc.before-navdp-autodl.$backup_stamp" ||
       ! chmod --reference="$bashrc" "$temp"; then
      status=1
      break
    fi

    if [[ -L "$bashrc" ]] ||
       ! current_meta="$(stat -c '%d:%i' -- "$bashrc")" ||
       ! current_hash="$(sha256sum "$bashrc" | awk '{print $1}')"; then
      status=4
      break
    fi
    if [[ "$current_meta" != "$original_meta" ||
          "$current_hash" != "$original_hash" ]]; then
      status=3
      break
    fi
    if ! mv "$temp" "$bashrc"; then
      status=1
      break
    fi
    temp=""
    BASHRC_CHANGED=1
    status=0
    break
  done

  [[ -z "$temp" ]] || command rm -f "$temp"
  flock -u "$lock_fd" || status=1
  exec {lock_fd}>&-
  return "$status"
}

bashrc_runtime_block_is_current() {
  [[ "${NAVDP_SKIP_BASHRC_UPDATE:-0}" == 1 ]] && return 0
  local bashrc="$NAVDP_BASHRC_FILE" quoted
  [[ ! -L "$bashrc" ]] || return 1
  [[ -f "$bashrc" ]] || return 1
  printf -v quoted '%q' "$NAVDP_RUNTIME_ENV_FILE"
  [[ "$(grep -Fxc '# >>> NavDP AutoDL runtime >>>' "$bashrc" || true)" == 1 &&
     "$(grep -Fxc '# <<< NavDP AutoDL runtime <<<' "$bashrc" || true)" == 1 &&
     "$(grep -Fxc "source $quoted" "$bashrc" || true)" == 1 ]]
}

rollback_runtime_environment() {
  local existed_before="$1" rollback_copy="$2" expected_current_hash="$3"
  local env_dir temp current_hash
  env_dir="$(dirname "$NAVDP_RUNTIME_ENV_FILE")"
  [[ -f "$NAVDP_RUNTIME_ENV_FILE" &&
     ! -L "$NAVDP_RUNTIME_ENV_FILE" ]] || return 2
  if ! current_hash="$(sha256sum "$NAVDP_RUNTIME_ENV_FILE" | awk '{print $1}')"; then
    return 2
  fi
  [[ "$current_hash" == "$expected_current_hash" ]] || return 2
  if [[ "$existed_before" == 0 ]]; then
    command rm -f "$NAVDP_RUNTIME_ENV_FILE"
    return
  fi
  [[ -f "$rollback_copy" && ! -L "$NAVDP_RUNTIME_ENV_FILE" ]] || return 1
  if ! temp="$(mktemp "$env_dir/.autodl-runtime.rollback.XXXXXX")"; then
    return 1
  fi
  if ! cp -p "$rollback_copy" "$temp" ||
     ! mv "$temp" "$NAVDP_RUNTIME_ENV_FILE"; then
    command rm -f "$temp"
    return 1
  fi
}

select_nvidia_icd() {
  discover_nvidia_icds
  local candidate index=0
  for candidate in "${NVIDIA_ICD_CANDIDATES[@]}"; do
    index=$((index + 1))
    if probe_nvidia_icd "$candidate" "$index"; then
      SELECTED_ICD="$candidate"
      break
    fi
  done
  if [[ -z "$SELECTED_ICD" ]]; then
    fail_check "No valid single-GPU NVIDIA Vulkan ICD was found"
    return 1
  fi

  export VK_ICD_FILENAMES="$SELECTED_ICD"
  export VK_DRIVER_FILES="$SELECTED_ICD"
  NAVDP_REPO_ROOT="$REPO_ROOT"
  export CONDA_BIN CONDA_ENVS_PATH AUTODL_WORK_DIR ISAACLAB_DIR NAVDP_REPO_ROOT
  if [[ "$CHECK_ONLY" == 1 ]]; then
    pass "Selected NVIDIA Vulkan ICD for this check: $SELECTED_ICD"
    return 0
  fi

  local old_env_hash="" new_env_hash written_env_hash="" bashrc_was_current=0
  local update_status validation_status write_status rollback_status
  local runtime_existed_before=0 runtime_rollback_copy=""
  if [[ -f "$NAVDP_RUNTIME_ENV_FILE" ]]; then
    runtime_existed_before=1
    if ! old_env_hash="$(sha256sum "$NAVDP_RUNTIME_ENV_FILE" | awk '{print $1}')"; then
      fail_check "Unable to read existing NavDP runtime environment"
      return 1
    fi
    if ! runtime_rollback_copy="$(mktemp "$REPORT_DIR/runtime-env.rollback.XXXXXX")" ||
       ! cp -p "$NAVDP_RUNTIME_ENV_FILE" "$runtime_rollback_copy"; then
      fail_check "Unable to prepare NavDP runtime environment rollback"
      return 1
    fi
  elif [[ -L "$NAVDP_RUNTIME_ENV_FILE" || -e "$NAVDP_RUNTIME_ENV_FILE" ]]; then
    fail_check "Refusing unsafe NavDP runtime environment target: $NAVDP_RUNTIME_ENV_FILE"
    return 1
  fi
  if bashrc_runtime_block_is_current; then
    bashrc_was_current=1
  fi
  if validate_bashrc_runtime_markers; then
    validation_status=0
  else
    validation_status=$?
  fi
  if ((validation_status != 0)); then
    if ((validation_status == 4)); then
      fail_check "Refusing symlinked NavDP runtime bashrc: $NAVDP_BASHRC_FILE"
    else
      fail_check "Malformed NavDP runtime block in $NAVDP_BASHRC_FILE; refusing to modify it"
    fi
    [[ -z "$runtime_rollback_copy" ]] || command rm -f "$runtime_rollback_copy"
    return 1
  fi
  if write_runtime_environment; then
    write_status=0
  else
    write_status=$?
  fi
  if ((write_status != 0)); then
    fail_check "Unable to write NavDP runtime environment: $NAVDP_RUNTIME_ENV_FILE"
    [[ -z "$runtime_rollback_copy" ]] || command rm -f "$runtime_rollback_copy"
    return 1
  fi
  if ! written_env_hash="$(sha256sum "$NAVDP_RUNTIME_ENV_FILE" | awk '{print $1}')"; then
    fail_check "Unable to verify the newly written NavDP runtime environment"
    [[ -z "$runtime_rollback_copy" ]] || command rm -f "$runtime_rollback_copy"
    return 1
  fi
  if update_bashrc; then
    update_status=0
  else
    update_status=$?
  fi
  if ((update_status != 0)); then
    if ((RUNTIME_ENV_CHANGED)); then
      if rollback_runtime_environment \
          "$runtime_existed_before" "$runtime_rollback_copy" "$written_env_hash"; then
        :
      else
        rollback_status=$?
        if ((rollback_status == 2)); then
          fail_check "NavDP runtime environment changed concurrently; refusing to roll it back"
        else
          fail_check "Unable to roll back NavDP runtime environment after bashrc failure"
        fi
      fi
    fi
    if ((update_status == 2)); then
      fail_check "Malformed NavDP runtime block in $NAVDP_BASHRC_FILE; refusing to modify it"
    elif ((update_status == 3)); then
      fail_check "NavDP runtime bashrc changed concurrently; refusing to overwrite it"
    elif ((update_status == 4)); then
      fail_check "Refusing symlinked NavDP runtime bashrc: $NAVDP_BASHRC_FILE"
    else
      fail_check "Unable to update NavDP runtime block in $NAVDP_BASHRC_FILE"
    fi
    [[ -z "$runtime_rollback_copy" ]] || command rm -f "$runtime_rollback_copy"
    return 1
  fi
  [[ -z "$runtime_rollback_copy" ]] || command rm -f "$runtime_rollback_copy"
  if ! new_env_hash="$(sha256sum "$NAVDP_RUNTIME_ENV_FILE" | awk '{print $1}')"; then
    fail_check "Unable to verify NavDP runtime environment write"
    return 1
  fi
  if [[ "$old_env_hash" == "$new_env_hash" && -n "$old_env_hash" &&
        "$bashrc_was_current" == 1 ]]; then
    pass "Selected NVIDIA Vulkan ICD already persisted: $SELECTED_ICD"
  else
    repaired "Selected NVIDIA Vulkan ICD: $SELECTED_ICD"
  fi
}

check_conda_environments() {
  local listing
  if ! listing="$("$CONDA_BIN" env list --json 2>&1)"; then
    printf '%s\n' "$listing" >>"$REPORT_DIR/environment.txt"
    fail_check "Unable to list Conda environments; run: bash scripts/setup_autodl.sh"
    return 1
  fi
  printf '\nConda environments:\n%s\n' "$listing" >>"$REPORT_DIR/environment.txt"
  local missing=0
  grep -Fq "/${NAVDP_ENV_NAME}\"" <<<"$listing" || missing=1
  grep -Fq "/${ISAACLAB_ENV_NAME}\"" <<<"$listing" || missing=1
  if ((missing)); then
    fail_check "Required Conda environments are missing; run: bash scripts/setup_autodl.sh"
    return 1
  fi
  pass "NavDP and IsaacLab Conda environments found"
}

check_torch_cuda() {
  if "$CONDA_BIN" run --no-capture-output -n "$ISAACLAB_ENV_NAME" python - \
      >"$REPORT_DIR/torch-cuda.txt" 2>&1 <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
if torch.cuda.device_count() < 1:
    raise RuntimeError("no CUDA devices")
name = torch.cuda.get_device_name(0)
if "NVIDIA" not in name.upper():
    raise RuntimeError(f"device 0 is not NVIDIA: {name}")
x = torch.ones(1, device="cuda:0")
if float(x.item()) != 1.0:
    raise RuntimeError("CUDA tensor result is incorrect")
torch.cuda.synchronize()
print("CUDA_OK", name)
PY
  then
    pass "PyTorch CUDA validation passed"
  else
    fail_check "PyTorch CUDA validation failed; repair the $ISAACLAB_ENV_NAME environment with scripts/setup_autodl.sh"
    return 1
  fi
}

check_navdp_imports() {
  if "$CONDA_BIN" run --no-capture-output -n "$NAVDP_ENV_NAME" python - \
      >>"$REPORT_DIR/environment.txt" 2>&1 <<'PY'
import torch
import flask
import diffusers

print("NAVDP_IMPORTS_OK")
PY
  then
    pass "NavDP Python imports passed"
  else
    fail_check "NavDP import validation failed; repair the $NAVDP_ENV_NAME environment with scripts/setup_autodl.sh"
    return 1
  fi
}

check_runtime_contract() {
  local eval_file="$REPO_ROOT/run_scripts/eval_pointgoal_wheeled.py"
  local backend_file="$REPO_ROOT/experiments/simulators/isaac_navdp_backend.py"
  local token cli_token flag ok=1 help_output
  local -a tokens=(
    minco_start_validation_exemption_radius
    minco_penalty_weight_attractor
    navdp_seeds
    raw_controller
    experiment_variant
  )
  {
    printf 'Runtime source hashes:\n'
    sha256sum "$eval_file"
    sha256sum "$backend_file"
    printf '\nStatic token checks:\n'
  } >>"$REPORT_DIR/runtime-contract.txt"

  for token in "${tokens[@]}"; do
    cli_token="${token//_/-}"
    if ! grep -Fq "$token" "$eval_file" &&
       ! grep -Fq "$cli_token" "$eval_file"; then
      printf 'MISSING eval %s\n' "$token" >>"$REPORT_DIR/runtime-contract.txt"
      ok=0
    fi
  done

  if ! NAVDP_REPAIR_PY_MODE=validate-plan \
      "$CONDA_BIN" run --no-capture-output -n "$NAVDP_ENV_NAME" \
      python - "$backend_file" "$eval_file" \
      >>"$REPORT_DIR/runtime-contract.txt" 2>&1 <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
eval_path = pathlib.Path(sys.argv[2])
eval_tree = ast.parse(
    eval_path.read_text(encoding="utf-8"),
    filename=str(eval_path),
)
parse_lines = [
    node.lineno
    for node in ast.walk(eval_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "parse_args"
]
launcher_lines = [
    node.lineno
    for node in ast.walk(eval_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "AppLauncher"
]
if (
    not parse_lines
    or not launcher_lines
    or min(parse_lines) >= min(launcher_lines)
):
    raise SystemExit("EVAL_PARSE_ORDER_INVALID: parse_args must precede AppLauncher")
print("EVAL_PARSE_ORDER_OK")

backend_class = next(
    (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IsaacNavDPBackend"
    ),
    None,
)
if backend_class is None:
    raise SystemExit("MISSING backend class IsaacNavDPBackend")
build_command = next(
    (
        node
        for node in backend_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_command"
    ),
    None,
)
if build_command is None:
    raise SystemExit("MISSING backend method build_command")
command_sources = []
for node in ast.walk(build_command):
    if isinstance(node, ast.Return) and node.value is not None:
        command_sources.append(node.value)
    elif isinstance(node, ast.Assign):
        if any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets):
            command_sources.append(node.value)
    elif isinstance(node, ast.AnnAssign):
        if (
            isinstance(node.target, ast.Name)
            and node.target.id == "command"
            and node.value is not None
        ):
            command_sources.append(node.value)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "command"
            and node.func.attr in {"append", "extend"}
        ):
            command_sources.extend(node.args)
constants = {
    node.value
    for source in command_sources
    for node in ast.walk(source)
    if isinstance(node, ast.Constant)
    and isinstance(node.value, str)
    and node.value.startswith("--")
}
required = {
    "--minco_start_validation_exemption_radius",
    "--minco_penalty_weight_attractor",
    "--navdp-seeds",
    "--raw-controller",
    "--experiment-variant",
}
missing = sorted(required - constants)
if missing:
    raise SystemExit("MISSING backend build_command constants: " + ", ".join(missing))
print("BACKEND_AST_OK")
PY
  then
    ok=0
  fi

  if help_output="$("$CONDA_BIN" run --no-capture-output -n "$ISAACLAB_ENV_NAME" \
      bash "$ISAACLAB_DIR/isaaclab.sh" -p "$eval_file" --help 2>&1)"; then
    printf '\nEval help:\n%s\n' "$help_output" >>"$REPORT_DIR/runtime-contract.txt"
    for flag in \
      --minco_start_validation_exemption_radius \
      --minco_penalty_weight_attractor \
      --navdp-seeds \
      --raw-controller \
      --experiment-variant; do
      if ! grep -Fq -- "$flag" <<<"$help_output"; then
        printf 'MISSING help %s\n' "$flag" >>"$REPORT_DIR/runtime-contract.txt"
        ok=0
      fi
    done
  else
    printf '\nEval help failed:\n%s\n' "$help_output" >>"$REPORT_DIR/runtime-contract.txt"
    ok=0
  fi

  if ((ok)); then
    pass "Runtime source and eval parser contracts passed"
  else
    printf 'Runtime SHA-256:\n' >&2
    sha256sum "$eval_file" "$backend_file" >&2
    fail_check "远端运行时版本混用：请重新同步仓库或可信更新包；脚本不会覆盖运行时代码"
    return 1
  fi
}

smoke_log_has_fatal() {
  grep -Eiq \
    'Multiple Installable Client Drivers|Failed to create any GPU|GPU Foundation is not initialized|no suitable CUDA GPU|Fatal|Segmentation' \
    "$REPORT_DIR/isaac-smoke.log"
}

smoke_log_has_ready_gpu() {
  grep -Eiq 'app ready|Simulation App Startup Complete' \
    "$REPORT_DIR/isaac-smoke.log" &&
    grep -Ei 'NVIDIA|RTX' "$REPORT_DIR/isaac-smoke.log" |
      grep -Eiq '(^|[[:space:]|])Active([[:space:]|]|$)|Yes:[[:space:]]*[0-9]+'
}

classify_isaac_smoke() {
  if smoke_log_has_fatal; then
    fail_check "Isaac smoke reported a fatal GPU error"
    return 1
  fi
  if smoke_log_has_ready_gpu; then
    pass "Isaac headless GPU smoke passed"
    return 0
  fi
  fail_check "Isaac smoke did not reach app ready with an active NVIDIA GPU"
  return 1
}

start_isaac_smoke() {
  if ! command -v setsid >/dev/null 2>&1; then
    fail_check "Isaac smoke requires setsid for isolated process-group cleanup"
    return 1
  fi
  if ! : >"$REPORT_DIR/isaac-smoke.log"; then
    fail_check "Unable to initialize Isaac smoke log"
    return 1
  fi
  local -a command=(
    "$CONDA_BIN" run --no-capture-output -n "$ISAACLAB_ENV_NAME"
    bash "$ISAACLAB_DIR/isaaclab.sh" -p
    "$ISAACLAB_DIR/source/standalone/tutorials/00_sim/create_empty.py"
    --headless
  )
  local identity_file="$REPORT_DIR/isaac-smoke.identity"
  local reported_pid reported_pgid reported_sid extra attempt
  SMOKE_CLEANUP_REPORTED=0
  setsid python3 - "$identity_file" "${command[@]}" \
    >"$REPORT_DIR/isaac-smoke.log" 2>&1 <<'PY' &
import os
import sys

identity_path = sys.argv[1]
command = sys.argv[2:]
fd = os.open(identity_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="ascii") as handle:
    handle.write(f"{os.getpid()} {os.getpgid(0)} {os.getsid(0)}\n")
    handle.flush()
    os.fsync(handle.fileno())
os.execvp(command[0], command)
PY
  SMOKE_PID=$!

  for attempt in {1..100}; do
    [[ -s "$identity_file" ]] && break
    python3 -c 'import time; time.sleep(0.01)' || break
  done
  if ! IFS=' ' read -r reported_pid reported_pgid reported_sid extra \
      <"$identity_file" 2>/dev/null ||
     [[ -n "${extra:-}" ||
        ! "$reported_pid" =~ ^[0-9]+$ ||
        ! "$reported_pgid" =~ ^[0-9]+$ ||
        ! "$reported_sid" =~ ^[0-9]+$ ||
        "$reported_pid" != "$SMOKE_PID" ||
        "$reported_pgid" != "$SMOKE_PID" ||
        "$reported_sid" != "$SMOKE_PID" ]]; then
    fail_check "setsid did not establish the required Isaac smoke PID/PGID/SID identity"
    if smoke_leader_is_alive; then
      signal_backend -TERM "$SMOKE_PID" 2>/dev/null || true
    fi
    reap_smoke_leader_if_exited
    return 1
  fi
  SMOKE_PGID="$reported_pgid"
  SMOKE_SID="$reported_sid"
  SMOKE_GROUPED=1
  printf 'SMOKE_IDENTITY pid=%s pgid=%s sid=%s\n' \
    "$reported_pid" "$reported_pgid" "$reported_sid" \
    >>"$REPORT_DIR/isaac-smoke.log"
  return 0
}

wait_for_smoke_result() {
  local started now
  started="$(date +%s)"
  while true; do
    if smoke_log_has_fatal; then
      if ! stop_smoke_group; then
        fail_check "Failed to clean up Isaac smoke process group"
      fi
      classify_isaac_smoke
      return
    fi
    if smoke_log_has_ready_gpu; then
      if ! stop_smoke_group; then
        fail_check "Failed to clean up Isaac smoke process group"
        return 1
      fi
      classify_isaac_smoke
      return
    fi
    reap_smoke_leader_if_exited
    if [[ -z "$SMOKE_PID" ]]; then
      if smoke_group_is_alive; then
        :
      else
        SMOKE_PGID=""
        SMOKE_SID=""
        SMOKE_GROUPED=0
        classify_isaac_smoke
        return
      fi
    fi
    now="$(date +%s)"
    if ((now - started >= SMOKE_TIMEOUT)); then
      printf '\nSMOKE_TIMEOUT after %s seconds\n' "$SMOKE_TIMEOUT" \
        >>"$REPORT_DIR/isaac-smoke.log"
      if ! stop_smoke_group; then
        fail_check "Failed to clean up Isaac smoke process group"
      fi
      classify_isaac_smoke
      return
    fi
    "$SLEEP_BIN" 0.25
  done
}

run_isaac_smoke() {
  if ! start_isaac_smoke; then
    return 1
  fi
  wait_for_smoke_result
}

locate_dry_run_plan() {
  local info
  if ! info="$(NAVDP_REPAIR_PY_MODE=config \
      "$CONDA_BIN" run --no-capture-output -n "$NAVDP_ENV_NAME" \
      python - "$CONFIG" 2>>"$REPORT_DIR/experiment-dry-run.txt" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1]).resolve()
data = json.loads(config_path.read_text())
suite_id = data.get("suite_id")
if not suite_id:
    raise SystemExit("suite config has no suite_id")
output_root = pathlib.Path(data.get("output_root", "results"))
if not output_root.is_absolute():
    output_root = (config_path.parent / output_root).resolve()
print(suite_id)
print(output_root)
print(output_root / suite_id / "dry_run_plan.json")
PY
  )"; then
    fail_check "Unable to resolve suite dry-run plan location"
    return 1
  fi
  SUITE_ID="$(sed -n '1p' <<<"$info")"
  SUITE_OUTPUT_ROOT="$(sed -n '2p' <<<"$info")"
  DRY_RUN_PLAN="$(sed -n '3p' <<<"$info")"
  [[ -n "$DRY_RUN_PLAN" ]]
}

validate_dry_run_plan() {
  NAVDP_REPAIR_PY_MODE=validate-plan \
    "$CONDA_BIN" run --no-capture-output -n "$NAVDP_ENV_NAME" \
    python - "$DRY_RUN_PLAN" >>"$REPORT_DIR/experiment-dry-run.txt" 2>&1 <<'PY'
import json
import pathlib
import sys

plan_path = pathlib.Path(sys.argv[1])
plan = json.loads(plan_path.read_text())


class ValidationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def value_after(command, option):
    try:
        index = command.index(option)
    except ValueError as exc:
        raise ValidationError(f"missing {option}") from exc
    if index + 1 >= len(command) or str(command[index + 1]).startswith("--"):
        raise ValidationError(f"{option} has no separate value")
    return str(command[index + 1])


def values_after(command, option):
    try:
        index = command.index(option)
    except ValueError as exc:
        raise ValidationError(f"missing {option}") from exc
    values = []
    for value in command[index + 1:]:
        if str(value).startswith("--"):
            break
        values.append(str(value))
    if not values:
        raise ValidationError(f"{option} has no separate values")
    return values


single_value_options = {
    "--experiment-config",
    "--experiment-run-dir",
    "--experiment-variant",
    "--scenario-manifest",
    "--scene-path",
    "--scene-id",
    "--warm-start-mode",
    "--seed",
    "--navdp-seed",
    "--num_envs",
    "--num_episodes",
    "--speed",
    "--mpc_max_yaw_rate",
    "--use_robot_base_frame",
    "--port",
    "--raw-controller",
    "--minco_initial_top_k",
    "--minco_max_top_k",
    "--minco_candidate_time_budget_ms",
    "--minco_optimization_safe_dist",
    "--minco_validation_safe_dist",
    "--minco_start_validation_exemption_radius",
    "--minco_sample_dt",
    "--minco_max_vel",
    "--minco_max_acc",
    "--minco_max_iterations",
    "--minco_penalty_weight_pos",
    "--minco_penalty_weight_vel",
    "--minco_penalty_weight_acc",
    "--minco_penalty_weight_attractor",
    "--minco_time_weight",
    "--minco_time_barrier_weight",
    "--esdf_resolution",
    "--esdf_padding",
    "--esdf_cache_name",
    "--esdf_obstacle_min_height",
    "--esdf_obstacle_max_height",
    "--esdf_fill_footprint",
    "--esdf_footprint_inflate_cells",
    "--mpc_max_yaw_acc",
    "--mpc_max_wheel_speed",
    "--scene_scale",
    "--video-fps",
    "--video-crf",
    "--video-scale",
}
multi_value_options = {"--episode-uids", "--navdp-seeds"}
flag_options = {
    "--headless",
    "--save-video",
    "--no-save-video",
    "--save-debug-visuals",
    "--eval-monitor",
    "--save-planning-trace",
    "--no-save-planning-trace",
    "--esdf_force_rebuild",
    "--enable_minco",
    "--no-enable_minco",
}
known_options = single_value_options | multi_value_options | flag_options


def parse_eval_options(command, start, command_index):
    parsed = {}
    flags = set()
    cursor = start
    while cursor < len(command):
        option = command[cursor]
        require(option.startswith("--"), f"command {command_index} unexpected argv: {option}")
        require(
            option in known_options,
            f"command {command_index} unknown or glued argument: {option}",
        )
        require(
            option not in parsed and option not in flags,
            f"command {command_index} duplicate option: {option}",
        )
        if option in flag_options:
            flags.add(option)
            cursor += 1
            continue
        if option in single_value_options:
            require(
                cursor + 1 < len(command)
                and not command[cursor + 1].startswith("--"),
                f"command {command_index} {option} has no separate value",
            )
            parsed[option] = [command[cursor + 1]]
            cursor += 2
            continue
        values = []
        cursor += 1
        while cursor < len(command) and not command[cursor].startswith("--"):
            values.append(command[cursor])
            cursor += 1
        require(values, f"command {command_index} {option} has no separate values")
        parsed[option] = values
    return parsed, flags


def validate_server_command(command, command_index):
    require(
        not any(value.startswith("--port=") for value in command),
        f"server command {command_index} contains a glued --port option",
    )
    require(
        command.count("--port") == 1,
        f"server command {command_index} must contain --port exactly once",
    )
    if len(command) >= 2 and command[0] == "python":
        require(
            pathlib.Path(command[1]).name == "navdp_server.py",
            f"server command {command_index} has invalid python/navdp_server prefix",
        )
    else:
        required_prefix = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "navdp",
            "python",
        ]
        require(
            command[: len(required_prefix)] == required_prefix,
            f"server command {command_index} has invalid conda/navdp prefix",
        )
        cursor = len(required_prefix)
        if cursor < len(command) and command[cursor] == "-u":
            cursor += 1
        require(
            cursor < len(command)
            and pathlib.Path(command[cursor]).name == "navdp_server.py",
            f"server command {command_index} does not launch navdp_server.py",
        )
    return value_after(command, "--port")


try:
    require(
        type(plan.get("started_processes")) is int,
        "started_processes must be an integer",
    )
    require(plan.get("started_processes") == 0, "started_processes must be 0")
    require(plan.get("backend") == "isaac", "dry-run backend must be isaac")
    run_count = plan.get("run_count")
    require(type(run_count) is int and run_count > 0, "run_count must be > 0")
    commands = plan.get("commands")
    server_commands = plan.get("server_commands")
    require(isinstance(commands, list), "commands must be a list")
    require(isinstance(server_commands, list), "server_commands must be a list")
    require(len(commands) == run_count, "run_count must equal commands count")
    require(
        len(server_commands) == run_count,
        "run_count must equal server_commands count",
    )

    for index, (command, server_command) in enumerate(zip(commands, server_commands)):
        require(
            isinstance(command, list) and all(isinstance(v, str) for v in command),
            f"command {index} must be argv strings",
        )
        require(
            isinstance(server_command, list)
            and all(isinstance(v, str) for v in server_command),
            f"server command {index} must be argv strings",
        )
        required_eval_prefix = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "isaaclab",
            "bash",
        ]
        require(
            command[: len(required_eval_prefix)] == required_eval_prefix,
            f"command {index} has invalid conda/isaaclab prefix",
        )
        launcher_indexes = [
            position
            for position, value in enumerate(command)
            if pathlib.Path(value).name == "isaaclab.sh"
        ]
        require(
            len(launcher_indexes) == 1,
            f"command {index} must contain exactly one isaaclab.sh launcher",
        )
        launcher_index = launcher_indexes[0]
        require(
            launcher_index == len(required_eval_prefix),
            f"command {index} has unexpected argv before isaaclab.sh",
        )
        require(
            launcher_index + 2 < len(command)
            and command[launcher_index + 1] == "-p",
            f"command {index} requires isaaclab.sh immediately followed by -p",
        )
        require(
            pathlib.Path(command[launcher_index + 2]).name
            == "eval_pointgoal_wheeled.py",
            f"command {index} -p must launch eval_pointgoal_wheeled.py",
        )
        parsed, flags = parse_eval_options(command, launcher_index + 3, index)
        for required_option in (
            "--experiment-variant",
            "--episode-uids",
            "--navdp-seeds",
            "--num_envs",
            "--num_episodes",
            "--port",
            "--raw-controller",
            "--minco_start_validation_exemption_radius",
            "--minco_penalty_weight_attractor",
        ):
            require(
                required_option in parsed,
                f"command {index} missing {required_option}",
            )
        try:
            episodes = int(parsed["--num_episodes"][0])
            num_envs = int(parsed["--num_envs"][0])
        except ValueError as exc:
            raise ValidationError(
                f"command {index} num_envs/num_episodes must be integers"
            ) from exc
        require(episodes > 0, f"command {index} num_episodes must be > 0")
        require(num_envs == 1, f"command {index} num_envs must be 1")
        require(
            len(parsed["--navdp-seeds"]) == episodes,
            f"command {index} navdp-seeds count must equal num_episodes",
        )
        require(
            len(parsed["--episode-uids"]) == episodes,
            f"command {index} episode-uids count must equal num_episodes",
        )

        variant = parsed["--experiment-variant"][0]
        controller = parsed["--raw-controller"][0]
        require(
            not ({"--enable_minco", "--no-enable_minco"} <= flags),
            f"command {index} has conflicting MINCO switches",
        )
        if variant == "raw":
            require(
                controller == "original-navdp-mpc",
                f"RAW command {index} requires original-navdp-mpc",
            )
            require(
                "--no-enable_minco" in flags and "--enable_minco" not in flags,
                f"RAW command {index} requires only --no-enable_minco",
            )
        else:
            require(
                variant in {"minco-cold", "minco-hot"},
                f"command {index} has unsupported experiment variant: {variant}",
            )
            require(
                controller == "disabled",
                f"MINCO command {index} requires disabled raw controller",
            )
            require(
                "--enable_minco" in flags and "--no-enable_minco" not in flags,
                f"MINCO command {index} requires only --enable_minco",
            )

        eval_port = parsed["--port"][0]
        server_port = validate_server_command(server_command, index)
        require(
            eval_port == server_port,
            f"server/eval port mismatch at command {index}: "
            f"{server_port} != {eval_port}",
        )
except (ValidationError, ValueError, TypeError, KeyError) as exc:
    print(f"DRY_RUN_INVALID: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"DRY_RUN_OK runs={run_count} plan={plan_path}")
PY
}

run_experiment_dry_run() {
  if ! locate_dry_run_plan; then
    return 1
  fi
  local generation_marker="$REPORT_DIR/dry-run-generation.marker"
  local plan_dir lock_file lock_fd generated_hash validated_hash
  plan_dir="$(dirname "$DRY_RUN_PLAN")"
  lock_file="$plan_dir/.navdp-self-check-dry-run.lock"
  if ! mkdir -p "$plan_dir" || [[ -L "$lock_file" ]] ||
     ! exec {lock_fd}>"$lock_file" || ! flock -x "$lock_fd"; then
    fail_check "Unable to acquire the dry-run plan lock"
    return 1
  fi
  if ! : >"$generation_marker"; then
    flock -u "$lock_fd" || true
    exec {lock_fd}>&-
    fail_check "Unable to create dry-run generation marker"
    return 1
  fi
  {
    printf 'Command: conda run --no-capture-output -n %q python -m experiments run-suite ' \
      "$NAVDP_ENV_NAME"
    printf -- '--config %q --backend isaac --dry-run --skip-video\n' "$CONFIG"
  } >>"$REPORT_DIR/experiment-dry-run.txt"
  if ! (
    cd "$REPO_ROOT"
    "$CONDA_BIN" run --no-capture-output -n "$NAVDP_ENV_NAME" \
      python -m experiments run-suite \
      --config "$CONFIG" \
      --backend isaac \
      --dry-run \
      --skip-video
  ) >>"$REPORT_DIR/experiment-dry-run.txt" 2>&1; then
    flock -u "$lock_fd" || true
    exec {lock_fd}>&-
    fail_check "Experiment dry-run generation failed"
    return 1
  fi
  if [[ ! -f "$DRY_RUN_PLAN" ]]; then
    flock -u "$lock_fd" || true
    exec {lock_fd}>&-
    fail_check "Generated dry-run plan was not found"
    return 1
  fi
  if [[ ! "$DRY_RUN_PLAN" -nt "$generation_marker" ]]; then
    flock -u "$lock_fd" || true
    exec {lock_fd}>&-
    fail_check "Generated dry-run plan is stale; it was not produced by this run"
    return 1
  fi
  if ! generated_hash="$(sha256sum "$DRY_RUN_PLAN" | awk '{print $1}')"; then
    flock -u "$lock_fd" || true
    exec {lock_fd}>&-
    fail_check "Unable to hash the generated dry-run plan"
    return 1
  fi
  if [[ "${AUTODL_REPAIR_TESTING:-0}" == 1 &&
        -n "${AUTODL_REPAIR_TEST_PLAN_BEFORE_VALIDATE_HOOK:-}" ]]; then
    if ! "$AUTODL_REPAIR_TEST_PLAN_BEFORE_VALIDATE_HOOK"; then
      flock -u "$lock_fd" || true
      exec {lock_fd}>&-
      fail_check "Dry-run plan validation hook failed"
      return 1
    fi
  fi
  if validate_dry_run_plan; then
    :
  else
    flock -u "$lock_fd" || true
    exec {lock_fd}>&-
    fail_check "Experiment dry-run plan validation failed"
    return 1
  fi
  if ! validated_hash="$(sha256sum "$DRY_RUN_PLAN" | awk '{print $1}')" ||
     [[ "$validated_hash" != "$generated_hash" ]]; then
    flock -u "$lock_fd" || true
    exec {lock_fd}>&-
    fail_check "Generated dry-run plan changed after generation"
    return 1
  fi
  if ! flock -u "$lock_fd"; then
    exec {lock_fd}>&-
    fail_check "Unable to release the dry-run plan lock"
    return 1
  fi
  exec {lock_fd}>&-
  pass "Experiment dry-run contract passed"
}

find_latest_log() {
  local root="$1"
  local name="$2"
  [[ -d "$root" ]] || return 1
  find "$root" -type f -name "$name" -printf '%T@ %p\n' 2>/dev/null |
    sort -nr |
    sed -n '1{s/^[^ ]* //;p;}'
}

record_history_category() {
  local category="$1"
  local source="$2"
  printf '%s source=%s\n' "$category" "$source" \
    >>"$REPORT_DIR/historical-diagnostics.txt"
}

classify_historical_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  if grep -Eiq 'Multiple Installable Client Drivers' "$file" ||
     awk '
       tolower($0) ~ /devicename/ && tolower($0) ~ /nvidia/ { count++ }
       END { exit !(count > 1) }
     ' "$file"; then
    record_history_category DUPLICATE_VULKAN_ICD "$file"
  fi
  grep -Eiq 'Failed to create any GPU|GPU Foundation is not initialized|no suitable CUDA GPU' "$file" &&
    record_history_category GPU_FOUNDATION_FAILURE "$file"
  grep -Eiq 'unrecognized arguments|error: argument .*invalid|usage:.*eval_pointgoal' "$file" &&
    record_history_category ARGPARSE_MISMATCH "$file"
  grep -Eiq 'CUDA out of memory|CUDA OOM|OutOfMemoryError' "$file" &&
    record_history_category CUDA_OOM "$file"
  grep -Eiq 'OmniGraphSettings::getCudaDeviceOrdinal:.*Defaulting to GPU0' "$file" &&
    record_history_category OMNIGRAPH_GPU0_FALLBACK "$file"
  grep -Eiq 'Segmentation|(^|[^[:alpha:]])Fatal([^[:alpha:]]|$)' "$file" &&
    record_history_category SEGMENTATION_FATAL "$file"
  grep -Eiq 'app ready|Simulation App Startup Complete' "$file" &&
    record_history_category APP_READY "$file"
  grep -Fq '[FrameCheck]' "$file" &&
    record_history_category FRAME_CHECK "$file"
  grep -Fq '[ControlRef]' "$file" &&
    record_history_category CONTROL_REF "$file"
  grep -Fq '[EpisodeDone]' "$file" &&
    record_history_category EPISODE_DONE "$file"
  return 0
}

classify_historical_logs() {
  local root="$NAVDP_RESULTS_ROOT"
  if [[ -z "$root" ]]; then
    if [[ -z "$SUITE_OUTPUT_ROOT" ]]; then
      locate_dry_run_plan >/dev/null 2>&1 || true
    fi
    root="${SUITE_OUTPUT_ROOT:-$REPO_ROOT/results}"
  fi
  printf 'History root: %s\n' "$root" >>"$REPORT_DIR/historical-diagnostics.txt"

  local stderr_log stdout_log status_log found=0
  stderr_log="$(find_latest_log "$root" isaac_eval.stderr.log || true)"
  stdout_log="$(find_latest_log "$root" isaac_eval.stdout.log || true)"
  status_log="$(find_latest_log "$root" run_status.json || true)"
  if [[ -n "$stderr_log" ]]; then
    found=1
    classify_historical_file "$stderr_log"
  fi
  if [[ -n "$stdout_log" ]]; then
    found=1
    classify_historical_file "$stdout_log"
  fi
  if [[ -n "$status_log" ]]; then
    found=1
    if grep -Eq '"status"[[:space:]]*:[[:space:]]*"FAILED"' "$status_log"; then
      record_history_category RUN_STATUS_FAILED "$status_log"
      printf 'Retry command: python -m experiments run-suite --config %q --backend isaac --resume --retry-failed --allow-real-simulation --skip-video\n' \
        "$CONFIG" | tee -a "$REPORT_DIR/historical-diagnostics.txt"
    elif grep -Eq '"status"[[:space:]]*:[[:space:]]*"RUNNING"' "$status_log"; then
      record_history_category RUN_STATUS_RUNNING "$status_log"
    elif grep -Eq '"status"[[:space:]]*:[[:space:]]*"STALE"' "$status_log"; then
      record_history_category RUN_STATUS_STALE "$status_log"
    fi
  fi

  if ((found)); then
    warn "Historical experiment evidence classified (non-blocking)"
  else
    pass "No historical experiment logs found"
  fi
}

main() {
  if ! create_report_directory; then
    return 1
  fi
  local preflight_ok=1 conda_ok=1 icd_ok=1
  if ! preflight; then
    preflight_ok=0
  fi

  if ((preflight_ok)); then
    if scan_stale_processes; then
      signal_stale_processes || true
    fi
    record_gpu_environment
    if ! select_nvidia_icd; then
      icd_ok=0
    fi
    if ! check_conda_environments; then
      conda_ok=0
    fi
    if ((conda_ok)); then
      check_torch_cuda || true
      check_navdp_imports || true
      check_runtime_contract || true
      if [[ "$SKIP_SMOKE" == 1 ]]; then
        warn "Isaac smoke skipped by request"
      elif ((icd_ok)); then
        run_isaac_smoke || true
      else
        fail_check "Isaac smoke skipped because no valid NVIDIA ICD was selected"
      fi
      if [[ "$SKIP_DRY_RUN" == 1 ]]; then
        warn "Experiment dry-run skipped by request"
      else
        run_experiment_dry_run || true
      fi
    else
      fail_check "Conda-dependent CUDA, runtime, smoke, and dry-run checks could not run"
    fi
  else
    fail_check "Dependent checks skipped because preflight failed"
  fi

  classify_historical_logs
  if ! write_final_summary; then
    return 1
  fi
  ((FAIL_COUNT == 0))
}

main
exit $?
