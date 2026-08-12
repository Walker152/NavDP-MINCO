#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/autodl_self_check_repair.sh"
TEST_TMP="$(mktemp -d)"
trap 'rm -rf "$TEST_TMP"' EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_contains() {
  local file="$1"
  local expected="$2"
  grep -F -- "$expected" "$file" >/dev/null || {
    sed -n '1,260p' "$file" >&2 || true
    fail "expected '$expected' in $file"
  }
}

assert_not_contains() {
  local file="$1"
  local unexpected="$2"
  if grep -F -- "$unexpected" "$file" >/dev/null; then
    sed -n '1,260p' "$file" >&2 || true
    fail "did not expect '$unexpected' in $file"
  fi
}

assert_count() {
  local file="$1"
  local expected_count="$2"
  local pattern="$3"
  local actual_count
  actual_count="$(grep -F -c -- "$pattern" "$file" || true)"
  [[ "$actual_count" == "$expected_count" ]] || \
    fail "expected $expected_count occurrences of '$pattern' in $file, got $actual_count"
}

write_valid_plan() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
  cat >"$path" <<'JSON'
{
  "backend": "isaac",
  "run_count": 2,
  "started_processes": 0,
  "commands": [
    [
      "conda", "run", "--no-capture-output", "-n", "isaaclab",
      "bash", "/fixture/IsaacLab/isaaclab.sh", "-p", "/fixture/NavDP/run_scripts/eval_pointgoal_wheeled.py",
      "--experiment-variant", "raw",
      "--episode-uids", "ep_0", "ep_1",
      "--headless",
      "--navdp-seeds", "1000", "1001",
      "--num_envs", "1", "--num_episodes", "2",
      "--port", "8889",
      "--raw-controller", "original-navdp-mpc",
      "--minco_start_validation_exemption_radius", "0.35",
      "--minco_penalty_weight_attractor", "20.0",
      "--no-enable_minco"
    ],
    [
      "conda", "run", "--no-capture-output", "-n", "isaaclab",
      "bash", "/fixture/IsaacLab/isaaclab.sh", "-p", "/fixture/NavDP/run_scripts/eval_pointgoal_wheeled.py",
      "--experiment-variant", "minco-hot",
      "--episode-uids", "ep_0", "ep_1",
      "--headless",
      "--navdp-seeds", "1000", "1001",
      "--num_envs", "1", "--num_episodes", "2",
      "--port", "8889",
      "--raw-controller", "disabled",
      "--minco_start_validation_exemption_radius", "0.35",
      "--minco_penalty_weight_attractor", "20.0",
      "--enable_minco"
    ]
  ],
  "server_commands": [
    ["python", "navdp_server.py", "--port", "8889"],
    ["python", "navdp_server.py", "--port", "8889"]
  ]
}
JSON
}

write_invalid_plan() {
  local path="$1"
  local mode="$2"
  write_valid_plan "$path"
  case "$mode" in
    started)
      sed -i 's/"started_processes": 0/"started_processes": 1/' "$path"
      ;;
    glued)
      sed -i 's/"--minco_penalty_weight_attractor", "20.0"/"--minco_penalty_weight_attractor20.0"/' "$path"
      ;;
    seeds)
      sed -i '0,/"1000", "1001"/s//"1000"/' "$path"
      ;;
    controller)
      sed -i '0,/"original-navdp-mpc"/s//"disabled"/' "$path"
      ;;
    port)
      sed -i '0,/"--port", "8889"/s//"--port", "8890"/' "$path"
      ;;
    num-envs-glued)
      sed -i '0,/"--num_envs", "1"/s//"--num_envs1"/' "$path"
      ;;
    launcher-order)
      sed -i '0,\|"/fixture/IsaacLab/isaaclab.sh", "-p"|s||"/fixture/IsaacLab/isaaclab.sh", "--bogus-between-launcher-and-p", "-p"|' "$path"
      ;;
    conflicting-minco)
      sed -i 's/"--enable_minco"/"--no-enable_minco", "--enable_minco"/' "$path"
      ;;
    duplicate-server-port)
      sed -i '/\["python", "navdp_server.py"/s/"--port", "8889"/"--port", "8889", "--port", "8890"/' "$path"
      ;;
    invalid-server-prefix)
      sed -i '0,/\["python", "navdp_server.py", "--port", "8889"\]/s//["anything", "--port", "8889"]/' "$path"
      ;;
    invalid-eval-prefix)
      sed -i '0,/"conda", "run"/s//"anything", "conda", "run"/' "$path"
      ;;
    glued-server-port)
      sed -i '/\["python", "navdp_server.py"/s/]$/, "--port=8890"]/' "$path"
      ;;
    invalid-server-entry)
      sed -i 's/"navdp_server.py"/"not_navdp_server.py"/' "$path"
      ;;
    invalid-eval-entry)
      sed -i '0,/eval_pointgoal_wheeled.py/s//not_eval_pointgoal_wheeled.py/' "$path"
      ;;
    *)
      fail "unknown invalid plan mode: $mode"
      ;;
  esac
}

make_fixture_repo() {
  local root="$1"
  local output_root="$2"
  mkdir -p \
    "$root/configs/experiments" \
    "$root/experiments/simulators" \
    "$root/experiments/configs" \
    "$root/baselines/navdp" \
    "$root/run_scripts" \
    "$root/results"

  cat >"$root/run_scripts/eval_pointgoal_wheeled.py" <<'PY'
parser.add_argument("--minco_start_validation_exemption_radius")
parser.add_argument("--minco_penalty_weight_attractor")
parser.add_argument("--navdp-seeds")
parser.add_argument("--raw-controller")
parser.add_argument("--experiment-variant")
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True)
PY

  cat >"$root/experiments/simulators/isaac_navdp_backend.py" <<'PY'
class IsaacNavDPBackend:
    def build_command(self):
        return [
            "--minco_start_validation_exemption_radius",
            "--minco_penalty_weight_attractor",
            "--navdp-seeds",
            "--raw-controller",
            "--experiment-variant",
        ]
PY

  cat >"$root/configs/experiments/full_suite.json" <<JSON
{
  "suite_id": "fixture_suite",
  "backend": "isaac",
  "output_root": "$output_root",
  "scenario_manifest": "../../experiments/configs/real_pointgoal_scenarios.json",
  "runs": [
    {"experiment_id": "EXP", "variant": "raw", "warm_start_mode": "cold"}
  ]
}
JSON

  printf '{"manifest_id":"fixture","scenes":[]}\n' \
    >"$root/experiments/configs/real_pointgoal_scenarios.json"
  printf 'print("server")\n' >"$root/baselines/navdp/navdp_server.py"
}

make_fake_tools() {
  local bin_dir="$1"
  mkdir -p "$bin_dir"

  cat >"$bin_dir/nvidia-smi" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'nvidia-smi %s\n' "$*" >>"$FAKE_CALLS"
if [[ "$*" == *"--query-gpu="* ]]; then
  printf 'NVIDIA GeForce RTX 4090, 560.35.03, 24564 MiB, 0 %%\n'
else
  printf 'NVIDIA-SMI 560.35.03\n'
fi
SH

  cat >"$bin_dir/vulkaninfo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'vulkaninfo icd=%s\n' "${VK_ICD_FILENAMES:-unset}" >>"$FAKE_CALLS"
if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  cat <<'OUT'
GPU0:
        deviceName         = NVIDIA GeForce RTX 4090
        driverName         = NVIDIA
GPU1:
        deviceName         = llvmpipe (LLVM 15.0.7, 256 bits)
        driverName         = llvmpipe
GPU2:
        deviceName         = NVIDIA GeForce RTX 4090
        driverName         = NVIDIA
OUT
  exit 0
fi
if [[ "$VK_ICD_FILENAMES" == *"/etc/"* && "${FAKE_ETC_ICD_VALID:-1}" != 1 ]]; then
  printf 'GPU0:\n        deviceName = llvmpipe\n'
  exit 0
fi
if [[ "$VK_ICD_FILENAMES" == *"/usr/"* && "${FAKE_USR_ICD_VALID:-1}" != 1 ]]; then
  printf 'GPU0:\n        deviceName = llvmpipe\n'
  exit 0
fi
cat <<'OUT'
GPU0:
        deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        deviceName         = NVIDIA GeForce RTX 4090
        driverName         = NVIDIA
        driverInfo         = 560.35.03
OUT
SH

  cat >"$bin_dir/ps" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cat "$FAKE_PS_SNAPSHOT"
SH

  cat >"$bin_dir/fake-kill" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_KILL_LOG"
if [[ "${1:-}" == "-0" ]]; then
  if [[ "${FAKE_KILL_SURVIVES:-0}" == 1 ]]; then
    exit 0
  fi
  exit 1
fi
if [[ "${FAKE_KILL_SIGNAL_FAIL:-0}" == 1 ]]; then
  exit 1
fi
exit 0
SH

  cat >"$bin_dir/sleep" <<'SH'
#!/usr/bin/env bash
exit 0
SH

  cat >"$bin_dir/timeout" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
shift
exec "$@"
SH

  cat >"$bin_dir/conda" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'conda %s\n' "$*" >>"$FAKE_CALLS"

if [[ "${1:-}" == "env" && "${2:-}" == "list" && "${3:-}" == "--json" ]]; then
  printf '{"envs":["%s/conda/envs/navdp","%s/conda/envs/isaaclab"]}\n' \
    "$FAKE_AUTODL_WORK" "$FAKE_AUTODL_WORK"
  exit 0
fi

if [[ "$*" == *"python -m experiments run-suite"* ]]; then
  [[ "$*" == *"--backend isaac"* ]] || {
    printf 'unsafe run-suite invocation: missing --backend isaac\n' >&2
    exit 91
  }
  [[ "$*" == *"--dry-run"* ]] || {
    printf 'unsafe run-suite invocation: missing --dry-run\n' >&2
    exit 92
  }
  [[ "$*" == *"--skip-video"* ]] || {
    printf 'unsafe run-suite invocation: missing --skip-video\n' >&2
    exit 93
  }
  [[ "$*" != *"--allow-real-simulation"* ]] || {
    printf 'unsafe run-suite invocation: real simulation is forbidden\n' >&2
    exit 94
  }
  mkdir -p "$(dirname "$FAKE_DRY_RUN_DEST")"
  cp "$FAKE_DRY_RUN_SOURCE" "$FAKE_DRY_RUN_DEST"
  exit 0
fi

if [[ "$*" == *"create_empty.py --headless"* ]]; then
  case "${FAKE_SMOKE_MODE:-healthy}" in
    healthy)
      printf '| 0 | NVIDIA GeForce RTX 4090 | Yes: 0 |\n'
      printf '[3.557s] app ready\n'
      ;;
    fatal)
      printf 'Multiple Installable Client Drivers (ICDs)\n'
      printf 'Failed to create any GPU devices\n'
      printf 'GPU Foundation is not initialized!\n'
      ;;
    no-ready)
      printf '[ext: omni.gpu_foundation] startup\n'
      ;;
    inactive)
      printf '| 0 | NVIDIA GeForce RTX 4090 | Inactive |\n'
      printf '[3.557s] app ready\n'
      ;;
  esac
  exit 0
fi

if [[ "$*" == *"eval_pointgoal_wheeled.py --help"* ]]; then
  printf '%s\n' \
    '--minco_start_validation_exemption_radius' \
    '--minco_penalty_weight_attractor' \
    '--navdp-seeds' \
    '--raw-controller' \
    '--experiment-variant'
  exit 0
fi

if [[ "${NAVDP_REPAIR_PY_MODE:-}" == "config" || \
      "${NAVDP_REPAIR_PY_MODE:-}" == "validate-plan" ]]; then
  while (($#)); do
    if [[ "$1" == "python" ]]; then
      shift
      exec /usr/bin/python3 "$@"
    fi
    shift
  done
  exit 99
fi

if [[ "$*" == *"CUDA_OK"* || "$*" == *"python -"* && "$*" == *"-n isaaclab"* ]]; then
  if [[ "${FAKE_CUDA_OK:-1}" == 1 ]]; then
    printf 'CUDA_OK NVIDIA GeForce RTX 4090\n'
    exit 0
  fi
  printf 'CUDA unavailable\n' >&2
  exit 1
fi

if [[ "$*" == *"NAVDP_IMPORTS_OK"* || "$*" == *"-n navdp"* ]]; then
  if [[ "${FAKE_NAVDP_IMPORTS_OK:-1}" == 1 ]]; then
    printf 'NAVDP_IMPORTS_OK\n'
    exit 0
  fi
  printf 'diffusers import failed\n' >&2
  exit 1
fi

exit 0
SH

  chmod +x \
    "$bin_dir/nvidia-smi" \
    "$bin_dir/vulkaninfo" \
    "$bin_dir/ps" \
    "$bin_dir/fake-kill" \
    "$bin_dir/sleep" \
    "$bin_dir/timeout" \
    "$bin_dir/conda"
}

prepare_case() {
  local name="$1"
  CASE_DIR="$TEST_TMP/$name"
  CASE_REPO="$CASE_DIR/NavDP"
  CASE_HOME="$CASE_DIR/home"
  CASE_WORK="$CASE_DIR/autodl"
  CASE_ISAAC="$CASE_WORK/IsaacLab"
  CASE_BIN="$CASE_DIR/bin"
  CASE_REPORT_BASE="$CASE_DIR/reports"
  CASE_ENV_FILE="$CASE_HOME/.config/navdp/autodl-runtime.env"
  CASE_OUT="$CASE_DIR/out"
  CASE_CALLS="$CASE_DIR/calls"
  CASE_KILL_LOG="$CASE_DIR/kills"
  CASE_PS="$CASE_DIR/ps.txt"
  CASE_ETC_ICD="$CASE_DIR/etc/vulkan/icd.d"
  CASE_USR_ICD="$CASE_DIR/usr/share/vulkan/icd.d"
  CASE_RESULTS="$CASE_DIR/results"
  CASE_PLAN_SOURCE="$CASE_DIR/plan-source.json"
  CASE_PLAN_DEST="$CASE_RESULTS/fixture_suite/dry_run_plan.json"

  mkdir -p \
    "$CASE_HOME" \
    "$CASE_ISAAC/source/standalone/tutorials/00_sim" \
    "$CASE_ETC_ICD" \
    "$CASE_USR_ICD"
  : >"$CASE_CALLS"
  : >"$CASE_KILL_LOG"
  : >"$CASE_PS"
  : >"$CASE_HOME/.bashrc"
  : >"$CASE_ETC_ICD/nvidia_icd.json"
  : >"$CASE_USR_ICD/nvidia_icd.json"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$CASE_ISAAC/isaaclab.sh"
  chmod +x "$CASE_ISAAC/isaaclab.sh"
  printf 'print("fixture")\n' \
    >"$CASE_ISAAC/source/standalone/tutorials/00_sim/create_empty.py"

  make_fixture_repo "$CASE_REPO" "$CASE_RESULTS"
  make_fake_tools "$CASE_BIN"
  write_valid_plan "$CASE_PLAN_SOURCE"
}

run_case() {
  local name="$1"
  shift
  prepare_case "$name"

  set +e
  env \
    PATH="$CASE_BIN:/usr/bin:/bin" \
    HOME="$CASE_HOME" \
    AUTODL_REPAIR_TESTING=1 \
    AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
    AUTODL_WORK_DIR="$CASE_WORK" \
    ISAACLAB_DIR="$CASE_ISAAC" \
    CONDA_BIN="$CASE_BIN/conda" \
    CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
    NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
    NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
    NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
    NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" \
    VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
    PS_BIN="$CASE_BIN/ps" \
    KILL_BIN="${KILL_BIN_OVERRIDE:-$CASE_BIN/fake-kill}" \
    SLEEP_BIN="$CASE_BIN/sleep" \
    TIMEOUT_BIN="$CASE_BIN/timeout" \
    FAKE_CALLS="$CASE_CALLS" \
    FAKE_KILL_LOG="$CASE_KILL_LOG" \
    FAKE_PS_SNAPSHOT="$CASE_PS" \
    FAKE_AUTODL_WORK="$CASE_WORK" \
    FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" \
    FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
    FAKE_ETC_ICD_VALID="${FAKE_ETC_ICD_VALID:-1}" \
    FAKE_USR_ICD_VALID="${FAKE_USR_ICD_VALID:-1}" \
    FAKE_CUDA_OK="${FAKE_CUDA_OK:-1}" \
    FAKE_NAVDP_IMPORTS_OK="${FAKE_NAVDP_IMPORTS_OK:-1}" \
    FAKE_SMOKE_MODE="${FAKE_SMOKE_MODE:-healthy}" \
    FAKE_KILL_SURVIVES="${FAKE_KILL_SURVIVES:-0}" \
    FAKE_KILL_SIGNAL_FAIL="${FAKE_KILL_SIGNAL_FAIL:-0}" \
    FAKE_KILL_LIVENESS_FAIL="${FAKE_KILL_LIVENESS_FAIL:-0}" \
    FAKE_KILL_TERM_ALREADY_EXITED="${FAKE_KILL_TERM_ALREADY_EXITED:-0}" \
    HOSTILE_KILL_MARKER="${HOSTILE_KILL_MARKER:-}" \
    AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK="${AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK:-}" \
    AUTODL_REPAIR_TEST_PLAN_BEFORE_VALIDATE_HOOK="${AUTODL_REPAIR_TEST_PLAN_BEFORE_VALIDATE_HOOK:-}" \
    bash "$SCRIPT" \
      --config "$CASE_REPO/configs/experiments/full_suite.json" \
      --report-dir "$CASE_REPORT_BASE" \
      --smoke-timeout 2 \
      "$@" >"$CASE_OUT" 2>&1
  CASE_STATUS=$?
  set -e

  CASE_REPORT="$(sed -n 's/^Report: //p' "$CASE_OUT" | tail -n 1)"
  if [[ -z "$CASE_REPORT" ]]; then
    CASE_REPORT="$(find "$CASE_REPORT_BASE" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1{s/^[^ ]* //;p;}')"
  fi
}

run_prepared_case() {
  set +e
  env \
    PATH="$CASE_BIN:/usr/bin:/bin" \
    HOME="$CASE_HOME" \
    AUTODL_REPAIR_TESTING=1 \
    AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
    AUTODL_WORK_DIR="$CASE_WORK" \
    ISAACLAB_DIR="$CASE_ISAAC" \
    CONDA_BIN="$CASE_BIN/conda" \
    CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
    NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
    NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
    NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
    NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" \
    VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
    PS_BIN="$CASE_BIN/ps" \
    KILL_BIN="${KILL_BIN_OVERRIDE:-$CASE_BIN/fake-kill}" \
    SLEEP_BIN="$CASE_BIN/sleep" \
    TIMEOUT_BIN="$CASE_BIN/timeout" \
    FAKE_CALLS="$CASE_CALLS" \
    FAKE_KILL_LOG="$CASE_KILL_LOG" \
    FAKE_PS_SNAPSHOT="$CASE_PS" \
    FAKE_AUTODL_WORK="$CASE_WORK" \
    FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" \
    FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
    FAKE_ETC_ICD_VALID="${FAKE_ETC_ICD_VALID:-1}" \
    FAKE_USR_ICD_VALID="${FAKE_USR_ICD_VALID:-1}" \
    FAKE_CUDA_OK="${FAKE_CUDA_OK:-1}" \
    FAKE_NAVDP_IMPORTS_OK="${FAKE_NAVDP_IMPORTS_OK:-1}" \
    FAKE_SMOKE_MODE="${FAKE_SMOKE_MODE:-healthy}" \
    FAKE_KILL_SURVIVES="${FAKE_KILL_SURVIVES:-0}" \
    FAKE_KILL_SIGNAL_FAIL="${FAKE_KILL_SIGNAL_FAIL:-0}" \
    FAKE_KILL_LIVENESS_FAIL="${FAKE_KILL_LIVENESS_FAIL:-0}" \
    FAKE_KILL_TERM_ALREADY_EXITED="${FAKE_KILL_TERM_ALREADY_EXITED:-0}" \
    HOSTILE_KILL_MARKER="${HOSTILE_KILL_MARKER:-}" \
    AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK="${AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK:-}" \
    AUTODL_REPAIR_TEST_PLAN_BEFORE_VALIDATE_HOOK="${AUTODL_REPAIR_TEST_PLAN_BEFORE_VALIDATE_HOOK:-}" \
    PYTHONOPTIMIZE="${PYTHONOPTIMIZE:-}" \
    bash "$SCRIPT" \
      --config "$CASE_REPO/configs/experiments/full_suite.json" \
      --report-dir "$CASE_REPORT_BASE" \
      --smoke-timeout 2 \
      "$@" >"$CASE_OUT" 2>&1
  CASE_STATUS=$?
  set -e

  CASE_REPORT="$(sed -n 's/^Report: //p' "$CASE_OUT" | tail -n 1)"
  if [[ -z "$CASE_REPORT" ]]; then
    CASE_REPORT="$(find "$CASE_REPORT_BASE" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1{s/^[^ ]* //;p;}')"
  fi
}

[[ -f "$SCRIPT" ]] || fail "repair script is missing: $SCRIPT"
bash -n "$SCRIPT"
assert_not_contains "$SCRIPT" "assert torch.cuda"
assert_not_contains "$SCRIPT" "Older Python/kernel combinations fall back"
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$SCRIPT"
fi

bash "$SCRIPT" --help >"$TEST_TMP/help.out" 2>&1
assert_contains "$TEST_TMP/help.out" "Usage:"
assert_contains "$TEST_TMP/help.out" "--check-only"
assert_contains "$TEST_TMP/help.out" "--skip-smoke"
assert_contains "$TEST_TMP/help.out" "--skip-dry-run"
assert_contains "$TEST_TMP/help.out" "autodl-strict-history"
assert_contains "$TEST_TMP/help.out" "/root/NavDP"

set +e
bash "$SCRIPT" --unknown >"$TEST_TMP/unknown.out" 2>&1
unknown_status=$?
set -e
[[ "$unknown_status" == 2 ]] || fail "unknown option should return 2, got $unknown_status"
assert_contains "$TEST_TMP/unknown.out" "unknown option"

for option in --config --report-dir; do
  set +e
  bash "$SCRIPT" "$option" --skip-smoke \
    >"$TEST_TMP/missing-value-${option#--}.out" 2>&1
  missing_value_status=$?
  set -e
  [[ "$missing_value_status" == 2 ]] || \
    fail "$option followed by another option should return 2, got $missing_value_status"
  assert_contains "$TEST_TMP/missing-value-${option#--}.out" "requires a path"
done

set +e
AUTODL_REPAIR_TESTING=1 \
AUTODL_REPAIR_REPO_ROOT="$TEST_TMP/nonexistent-root" \
KILL_BIN=/bin/kill \
  bash "$SCRIPT" --check-only --skip-smoke --skip-dry-run \
    --report-dir "$TEST_TMP/real-kill-report" \
    >"$TEST_TMP/real-kill-backend.out" 2>&1
real_kill_status=$?
set -e
[[ "$real_kill_status" == 2 ]] || \
  fail "test mode should reject the real kill backend, got $real_kill_status"
assert_contains "$TEST_TMP/real-kill-backend.out" \
  "KILL_BIN must not resolve to the real system kill"

set +e
AUTODL_REPAIR_REPO_ROOT="$TEST_TMP/forbidden-root" \
  bash "$SCRIPT" --help >"$TEST_TMP/forbidden-root-override.out" 2>&1
root_override_status=$?
set -e
[[ "$root_override_status" == 2 ]] || \
  fail "repo-root test override should require test mode, got $root_override_status"
assert_contains "$TEST_TMP/forbidden-root-override.out" \
  "AUTODL_REPAIR_REPO_ROOT is only allowed with AUTODL_REPAIR_TESTING=1"

strict_report_base="$TEST_TMP/strict-production-reports"
mkdir -p "$TEST_TMP/strict-home"
set +e
env \
  HOME="$TEST_TMP/strict-home" \
  AUTODL_WORK_DIR="$TEST_TMP/local-autodl" \
  ISAACLAB_DIR="$TEST_TMP/local-autodl/IsaacLab" \
  CONDA_BIN="$TEST_TMP/local-conda" \
  CONDA_ENVS_PATH="$TEST_TMP/local-conda-envs" \
  NAVDP_RUNTIME_ENV_FILE="$TEST_TMP/local-runtime.env" \
  bash "$SCRIPT" --check-only --skip-smoke --skip-dry-run \
    --report-dir "$strict_report_base" \
    >"$TEST_TMP/strict-production.out" 2>&1
strict_status=$?
set -e
[[ "$strict_status" == 1 ]] || \
  fail "non-AutoDL production execution should fail, got $strict_status"
strict_report="$(sed -n 's/^Report: //p' "$TEST_TMP/strict-production.out" | tail -n 1)"
[[ -n "$strict_report" ]] || fail "strict production failure did not create a report"
assert_contains "$TEST_TMP/strict-production.out" \
  "Strict AutoDL repository mismatch: expected /root/NavDP"
assert_contains "$TEST_TMP/strict-production.out" \
  "Strict AutoDL path override rejected: AUTODL_WORK_DIR"
assert_contains "$strict_report/environment.txt" \
  "Profile: autodl-strict-history"
assert_contains "$strict_report/environment.txt" \
  "Expected repository: /root/NavDP"
assert_contains "$strict_report/environment.txt" \
  "Actual repository: $REPO_ROOT"
assert_contains "$strict_report/environment.txt" \
  "AutoDL work directory: /root/autodl-tmp/navdp"
assert_contains "$strict_report/environment.txt" \
  "IsaacLab directory: /root/autodl-tmp/navdp/IsaacLab"
assert_contains "$strict_report/environment.txt" \
  "Conda: /root/miniconda3/bin/conda"
assert_contains "$strict_report/environment.txt" \
  "Conda envs path: /root/autodl-tmp/navdp/conda/envs"
assert_contains "$strict_report/summary.txt" \
  "Profile: autodl-strict-history"

run_case healthy --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || {
  sed -n '1,300p' "$CASE_OUT" >&2
  fail "healthy repair should pass"
}
assert_contains "$CASE_OUT" "[REPAIRED] Selected NVIDIA Vulkan ICD"
assert_contains "$CASE_OUT" "Report:"
assert_contains "$CASE_ENV_FILE" "export VK_ICD_FILENAMES="
assert_contains "$CASE_ENV_FILE" "export VK_DRIVER_FILES="
assert_contains "$CASE_ENV_FILE" "export CONDA_ENVS_PATH="
assert_contains "$CASE_ENV_FILE" "export CONDA_BIN=$CASE_BIN/conda"
assert_contains "$CASE_ENV_FILE" "$CASE_ETC_ICD/nvidia_icd.json"
assert_not_contains "$CASE_ENV_FILE" "CUDA_VISIBLE_DEVICES"
assert_not_contains "$CASE_ENV_FILE" "NVIDIA_VISIBLE_DEVICES"
assert_not_contains "$CASE_ENV_FILE" "NVIDIA_DRIVER_CAPABILITIES"
assert_not_contains "$CASE_ENV_FILE" "CUDA_HOME"
assert_count "$CASE_HOME/.bashrc" 1 "# >>> NavDP AutoDL runtime >>>"
assert_count "$CASE_HOME/.bashrc" 1 "# <<< NavDP AutoDL runtime <<<"
[[ -f "$CASE_REPORT/summary.txt" ]] || fail "summary report missing"
assert_contains "$CASE_REPORT/summary.txt" "Profile: testing"
for required_report in \
  environment.txt \
  process-scan.txt \
  vulkan-before.txt \
  torch-cuda.txt \
  runtime-contract.txt \
  isaac-smoke.log \
  experiment-dry-run.txt \
  historical-diagnostics.txt; do
  [[ -f "$CASE_REPORT/$required_report" ]] || \
    fail "required report missing: $required_report"
done

cat >"$CASE_PS" <<EOF
910101 1 910101 910101 900 $CASE_BIN/conda run -n isaaclab bash $CASE_ISAAC/isaaclab.sh -p $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
910102 910101 910101 910101 890 $CASE_WORK/conda/envs/isaaclab/bin/python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
910201 1 910201 910201 800 python $CASE_REPO/baselines/navdp/navdp_server.py --port 8889 --api-token SUPERSECRET --authorization BEARERSECRET https://example.invalid/?token=URLSECRET
910301 1 910301 910301 700 python /other/NavDP/eval_pointgoal_wheeled.py
910401 1 910401 910401 600 python unrelated_worker.py
EOF

# Re-run the same prepared case directly so the custom process snapshot is kept.
set +e
env \
  PATH="$CASE_BIN:/usr/bin:/bin" HOME="$CASE_HOME" \
  AUTODL_REPAIR_TESTING=1 AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
  AUTODL_WORK_DIR="$CASE_WORK" ISAACLAB_DIR="$CASE_ISAAC" \
  CONDA_BIN="$CASE_BIN/conda" CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
  NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
  NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
  NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
  NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
  PS_BIN="$CASE_BIN/ps" KILL_BIN="$CASE_BIN/fake-kill" \
  SLEEP_BIN="$CASE_BIN/sleep" TIMEOUT_BIN="$CASE_BIN/timeout" \
  FAKE_CALLS="$CASE_CALLS" FAKE_KILL_LOG="$CASE_KILL_LOG" \
  FAKE_PS_SNAPSHOT="$CASE_PS" FAKE_AUTODL_WORK="$CASE_WORK" \
  FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
  bash "$SCRIPT" --config "$CASE_REPO/configs/experiments/full_suite.json" \
    --report-dir "$CASE_REPORT_BASE" --skip-smoke --skip-dry-run \
    >"$CASE_DIR/process-run.out" 2>&1
process_status=$?
set -e
[[ "$process_status" == 0 ]] || fail "process cleanup run should pass"
process_report="$(sed -n 's/^Report: //p' "$CASE_DIR/process-run.out" | tail -n 1)"
assert_contains "$CASE_KILL_LOG" "-TERM"
assert_contains "$CASE_KILL_LOG" "910101"
assert_contains "$CASE_KILL_LOG" "910201"
assert_not_contains "$CASE_KILL_LOG" "910301"
assert_not_contains "$CASE_KILL_LOG" "910401"
assert_not_contains "$process_report/process-scan.txt" "SUPERSECRET"
assert_not_contains "$process_report/process-scan.txt" "BEARERSECRET"
assert_not_contains "$process_report/process-scan.txt" "URLSECRET"
child_term_line="$(grep -n -- '-TERM 910102' "$CASE_KILL_LOG" | head -n 1 | cut -d: -f1)"
parent_term_line="$(grep -n -- '-TERM 910101' "$CASE_KILL_LOG" | head -n 1 | cut -d: -f1)"
[[ -n "$child_term_line" && -n "$parent_term_line" &&
   "$child_term_line" -lt "$parent_term_line" ]] || \
  fail "stale descendants must be signaled before their parent roots"

prepare_case active-and-readers
cat >"$CASE_PS" <<EOF
920100 920000 920100 920100 10 python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
920200 1 920200 920200 900 vim $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
920300 1 920300 920300 900 tail -f $CASE_REPO/baselines/navdp/navdp_server.py
EOF
run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "active/read-only process exclusion should pass"
[[ ! -s "$CASE_KILL_LOG" ]] || {
  sed -n '1,120p' "$CASE_KILL_LOG" >&2
  fail "active or read-only processes were incorrectly signaled"
}

prepare_case target-path-as-data
cat >"$CASE_PS" <<EOF
923100 1 923100 923100 900 python -c print_data $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
923200 1 923200 923200 900 bash -c echo_data $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
923300 1 923300 923300 900 conda run -n isaaclab python -c print_data $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
EOF
run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "target paths used only as data should be excluded"
[[ ! -s "$CASE_KILL_LOG" ]] || \
  fail "a process using the target path only as data was incorrectly signaled"

prepare_case young-orphan
cat >"$CASE_PS" <<EOF
925100 1 925100 925100 10 python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
EOF
run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "young orphan process exclusion should pass"
[[ ! -s "$CASE_KILL_LOG" ]] || fail "young orphan process was incorrectly signaled"
assert_contains "$CASE_REPORT/process-scan.txt" "YOUNG_ORPHAN_EXCLUDED"

prepare_case stubborn-stale
cat >"$CASE_PS" <<EOF
930100 1 930100 930100 900 python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
EOF
FAKE_KILL_SURVIVES=1 FAKE_KILL_SIGNAL_FAIL=1 \
  run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "un-killable stale process should fail"
assert_contains "$CASE_OUT" "Failed to terminate stale"
assert_not_contains "$CASE_OUT" "[REPAIRED] Terminated"

prepare_case stale-exits-before-term
cat >"$CASE_PS" <<EOF
930500 1 930500 930500 900 python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
EOF
FAKE_KILL_TERM_ALREADY_EXITED=1 \
  run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "naturally exited stale process should not fail repair"
assert_contains "$CASE_REPORT/process-scan.txt" "ALREADY_EXITED_BEFORE_TERM"

prepare_case unknown-stale-liveness
cat >"$CASE_PS" <<EOF
931100 1 931100 931100 900 python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
EOF
FAKE_KILL_LIVENESS_FAIL=1 \
  run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "unknown stale liveness should fail closed"
assert_contains "$CASE_OUT" "Unable to verify stale process liveness"
assert_not_contains "$CASE_OUT" "[REPAIRED] Terminated"

run_case check-only --check-only --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "healthy check-only should pass"
[[ ! -e "$CASE_ENV_FILE" ]] || fail "check-only wrote runtime environment"
[[ ! -s "$CASE_KILL_LOG" ]] || fail "check-only sent process signals"
assert_not_contains "$CASE_HOME/.bashrc" "# >>> NavDP AutoDL runtime >>>"

prepare_case check-only-with-stale
cat >"$CASE_PS" <<EOF
935100 1 935100 935100 900 python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
EOF
run_prepared_case --check-only --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "check-only with stale candidate should pass"
[[ ! -s "$CASE_KILL_LOG" ]] || fail "check-only with stale candidate sent signals"
assert_contains "$CASE_REPORT/process-scan.txt" "STALE_ROOT"
assert_contains "$CASE_OUT" "Check-only: found 1 stale"

prepare_case test-signal-isolation
cat >"$CASE_PS" <<EOF
936100 1 936100 936100 900 python $CASE_REPO/run_scripts/eval_pointgoal_wheeled.py
EOF
cat >"$CASE_BIN/hostile-kill-wrapper" <<'SH'
#!/usr/bin/env bash
printf 'EXECUTED\n' >>"$HOSTILE_KILL_MARKER"
exec /bin/kill "$@"
SH
chmod +x "$CASE_BIN/hostile-kill-wrapper"
HOSTILE_KILL_MARKER="$CASE_DIR/hostile-kill-executed"
KILL_BIN_OVERRIDE="$CASE_BIN/hostile-kill-wrapper" \
  run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "isolated test signal backend should pass"
[[ ! -e "$HOSTILE_KILL_MARKER" ]] || fail "test mode executed an external kill wrapper"
assert_contains "$CASE_KILL_LOG" "-TERM 936100"

FAKE_ETC_ICD_VALID=0 run_case usr-fallback --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "valid /usr ICD fallback should pass"
assert_contains "$CASE_ENV_FILE" "$CASE_USR_ICD/nvidia_icd.json"

FAKE_ETC_ICD_VALID=0 FAKE_USR_ICD_VALID=0 \
  run_case no-valid-icd --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "missing valid NVIDIA ICD should fail"
assert_contains "$CASE_OUT" "No valid single-GPU NVIDIA Vulkan ICD"

FAKE_CUDA_OK=0 run_case cuda-failure --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "CUDA failure should fail"
assert_contains "$CASE_OUT" "PyTorch CUDA validation failed"

FAKE_NAVDP_IMPORTS_OK=0 run_case navdp-import-failure --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "NavDP import failure should fail"
assert_contains "$CASE_OUT" "NavDP import validation failed"

prepare_case runtime-mismatch
sed -i '/minco_start_validation_exemption_radius/d' \
  "$CASE_REPO/experiments/simulators/isaac_navdp_backend.py"
set +e
env \
  PATH="$CASE_BIN:/usr/bin:/bin" HOME="$CASE_HOME" \
  AUTODL_REPAIR_TESTING=1 AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
  AUTODL_WORK_DIR="$CASE_WORK" ISAACLAB_DIR="$CASE_ISAAC" \
  CONDA_BIN="$CASE_BIN/conda" CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
  NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
  NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
  NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
  NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
  PS_BIN="$CASE_BIN/ps" KILL_BIN="$CASE_BIN/fake-kill" \
  SLEEP_BIN="$CASE_BIN/sleep" TIMEOUT_BIN="$CASE_BIN/timeout" \
  FAKE_CALLS="$CASE_CALLS" FAKE_KILL_LOG="$CASE_KILL_LOG" \
  FAKE_PS_SNAPSHOT="$CASE_PS" FAKE_AUTODL_WORK="$CASE_WORK" \
  FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
  bash "$SCRIPT" --config "$CASE_REPO/configs/experiments/full_suite.json" \
    --report-dir "$CASE_REPORT_BASE" --skip-smoke --skip-dry-run \
    >"$CASE_OUT" 2>&1
CASE_STATUS=$?
set -e
[[ "$CASE_STATUS" == 1 ]] || fail "runtime mismatch should fail"
assert_contains "$CASE_OUT" "远端运行时版本混用"

prepare_case eval-parser-order-mismatch
cat >"$CASE_REPO/run_scripts/eval_pointgoal_wheeled.py" <<'PY'
parser.add_argument("--minco_start_validation_exemption_radius")
parser.add_argument("--minco_penalty_weight_attractor")
parser.add_argument("--navdp-seeds")
parser.add_argument("--raw-controller")
parser.add_argument("--experiment-variant")
app_launcher = AppLauncher(headless=True)
args_cli = parser.parse_args()
PY
run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "AppLauncher before parse_args should fail runtime contract"
assert_contains "$CASE_OUT" "远端运行时版本混用"
assert_contains "$CASE_REPORT/runtime-contract.txt" "EVAL_PARSE_ORDER_INVALID"

prepare_case backend-dead-token
cat >"$CASE_REPO/experiments/simulators/isaac_navdp_backend.py" <<'PY'
DEAD_TEXT = "--minco_start_validation_exemption_radius --minco_penalty_weight_attractor --navdp-seeds --raw-controller --experiment-variant"

class IsaacNavDPBackend:
    def build_command(self):
        return ["conda", "run"]
PY
run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "dead backend token text should not satisfy runtime contract"
assert_contains "$CASE_OUT" "远端运行时版本混用"

FAKE_SMOKE_MODE=healthy run_case smoke-healthy --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || {
  sed -n '1,300p' "$CASE_OUT" >&2
  fail "healthy Isaac smoke should pass"
}
assert_contains "$CASE_OUT" "Isaac headless GPU smoke passed"

FAKE_SMOKE_MODE=fatal run_case smoke-fatal --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "fatal Isaac smoke should fail"
assert_contains "$CASE_OUT" "Isaac smoke reported a fatal GPU error"

FAKE_SMOKE_MODE=inactive run_case smoke-inactive --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "inactive GPU smoke row should fail"
assert_contains "$CASE_OUT" "did not reach app ready with an active NVIDIA GPU"

FAKE_SMOKE_MODE=no-ready run_case smoke-no-ready --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "smoke without readiness marker should fail"
assert_contains "$CASE_OUT" "did not reach app ready with an active NVIDIA GPU"

run_case dry-run-valid --skip-smoke
[[ "$CASE_STATUS" == 0 ]] || {
  sed -n '1,300p' "$CASE_OUT" >&2
  fail "valid experiment dry-run should pass"
}
assert_contains "$CASE_OUT" "Experiment dry-run contract passed"

prepare_case dry-run-concurrent-replacement
cat >"$CASE_DIR/plan-replacement-hook" <<SH
#!/usr/bin/env bash
sed -i 's/"backend": "isaac"/"backend": "isaac", "race_marker": true/' \
  "$CASE_PLAN_DEST"
SH
chmod +x "$CASE_DIR/plan-replacement-hook"
AUTODL_REPAIR_TEST_PLAN_BEFORE_VALIDATE_HOOK="$CASE_DIR/plan-replacement-hook" \
  run_prepared_case --skip-smoke
[[ "$CASE_STATUS" == 1 ]] || fail "concurrent dry-run plan replacement should fail closed"
assert_contains "$CASE_OUT" "dry-run plan changed after generation"

prepare_case dry-run-glued
write_invalid_plan "$CASE_PLAN_SOURCE" glued
set +e
env \
  PATH="$CASE_BIN:/usr/bin:/bin" HOME="$CASE_HOME" \
  AUTODL_REPAIR_TESTING=1 AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
  AUTODL_WORK_DIR="$CASE_WORK" ISAACLAB_DIR="$CASE_ISAAC" \
  CONDA_BIN="$CASE_BIN/conda" CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
  NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
  NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
  NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
  NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
  PS_BIN="$CASE_BIN/ps" KILL_BIN="$CASE_BIN/fake-kill" \
  SLEEP_BIN="$CASE_BIN/sleep" TIMEOUT_BIN="$CASE_BIN/timeout" \
  FAKE_CALLS="$CASE_CALLS" FAKE_KILL_LOG="$CASE_KILL_LOG" \
  FAKE_PS_SNAPSHOT="$CASE_PS" FAKE_AUTODL_WORK="$CASE_WORK" \
  FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
  bash "$SCRIPT" --config "$CASE_REPO/configs/experiments/full_suite.json" \
    --report-dir "$CASE_REPORT_BASE" --skip-smoke >"$CASE_OUT" 2>&1
CASE_STATUS=$?
set -e
[[ "$CASE_STATUS" == 1 ]] || fail "glued dry-run argument should fail"
assert_contains "$CASE_OUT" "dry-run plan validation failed"

for invalid_mode in \
  started \
  seeds \
  controller \
  port \
  num-envs-glued \
  launcher-order \
  conflicting-minco \
  duplicate-server-port \
  invalid-server-prefix \
  invalid-eval-prefix \
  glued-server-port \
  invalid-server-entry \
  invalid-eval-entry; do
  prepare_case "dry-run-$invalid_mode"
  write_invalid_plan "$CASE_PLAN_SOURCE" "$invalid_mode"
  set +e
  env \
    PATH="$CASE_BIN:/usr/bin:/bin" HOME="$CASE_HOME" \
    AUTODL_REPAIR_TESTING=1 AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
    AUTODL_WORK_DIR="$CASE_WORK" ISAACLAB_DIR="$CASE_ISAAC" \
    CONDA_BIN="$CASE_BIN/conda" CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
    NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
    NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
    NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
    NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
    PS_BIN="$CASE_BIN/ps" KILL_BIN="$CASE_BIN/fake-kill" \
    SLEEP_BIN="$CASE_BIN/sleep" TIMEOUT_BIN="$CASE_BIN/timeout" \
    FAKE_CALLS="$CASE_CALLS" FAKE_KILL_LOG="$CASE_KILL_LOG" \
    FAKE_PS_SNAPSHOT="$CASE_PS" FAKE_AUTODL_WORK="$CASE_WORK" \
    FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
    bash "$SCRIPT" --config "$CASE_REPO/configs/experiments/full_suite.json" \
      --report-dir "$CASE_REPORT_BASE" --skip-smoke >"$CASE_OUT" 2>&1
  CASE_STATUS=$?
  set -e
  [[ "$CASE_STATUS" == 1 ]] || \
    fail "$invalid_mode dry-run invariant should fail"
  assert_contains "$CASE_OUT" "dry-run plan validation failed"
done

prepare_case optimized-invalid-plan
write_invalid_plan "$CASE_PLAN_SOURCE" started
PYTHONOPTIMIZE=1 run_prepared_case --skip-smoke
[[ "$CASE_STATUS" == 1 ]] || \
  fail "optimized Python must not bypass dry-run validation"
assert_contains "$CASE_OUT" "dry-run plan validation failed"

prepare_case broken-bashrc
printf 'before\n# >>> NavDP AutoDL runtime >>>\nkeep-after-broken-marker\n' \
  >"$CASE_HOME/.bashrc"
run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "broken bashrc managed markers should fail safely"
assert_contains "$CASE_HOME/.bashrc" "keep-after-broken-marker"
assert_contains "$CASE_OUT" "Malformed NavDP runtime block"

prepare_case concurrent-bashrc-update
printf 'original-content\n' >"$CASE_HOME/.bashrc"
cat >"$CASE_DIR/concurrent-bashrc-hook" <<SH
#!/usr/bin/env bash
printf 'concurrent-content\\n' >>"$CASE_HOME/.bashrc"
SH
chmod +x "$CASE_DIR/concurrent-bashrc-hook"
AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK="$CASE_DIR/concurrent-bashrc-hook" \
  run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "concurrent bashrc update should fail closed"
assert_contains "$CASE_HOME/.bashrc" "original-content"
assert_contains "$CASE_HOME/.bashrc" "concurrent-content"
assert_not_contains "$CASE_HOME/.bashrc" "# >>> NavDP AutoDL runtime >>>"
assert_contains "$CASE_OUT" "changed concurrently"
[[ ! -e "$CASE_ENV_FILE" ]] || \
  fail "runtime environment was not rolled back after bashrc concurrency failure"

prepare_case concurrent-runtime-env-update
mkdir -p "$(dirname "$CASE_ENV_FILE")"
printf 'old-runtime-env\n' >"$CASE_ENV_FILE"
printf 'original-bashrc\n' >"$CASE_HOME/.bashrc"
cat >"$CASE_DIR/concurrent-runtime-hook" <<SH
#!/usr/bin/env bash
printf 'concurrent-runtime-env\\n' >"$CASE_ENV_FILE"
printf 'concurrent-bashrc\\n' >>"$CASE_HOME/.bashrc"
SH
chmod +x "$CASE_DIR/concurrent-runtime-hook"
AUTODL_REPAIR_TEST_BASHRC_BEFORE_REPLACE_HOOK="$CASE_DIR/concurrent-runtime-hook" \
  run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "concurrent runtime env update should fail closed"
assert_contains "$CASE_ENV_FILE" "concurrent-runtime-env"
assert_contains "$CASE_OUT" "runtime environment changed concurrently"

prepare_case unwritable-runtime-parent
printf 'not-a-directory\n' >"$CASE_HOME/runtime-parent"
CASE_ENV_FILE="$CASE_HOME/runtime-parent/autodl-runtime.env"
run_prepared_case --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 1 ]] || fail "runtime environment write failure should fail"
assert_not_contains "$CASE_OUT" "[REPAIRED] Selected NVIDIA Vulkan ICD"

run_case idempotent --skip-smoke --skip-dry-run
[[ "$CASE_STATUS" == 0 ]] || fail "first idempotence run failed"
first_env_hash="$(sha256sum "$CASE_ENV_FILE" | awk '{print $1}')"

set +e
env \
  PATH="$CASE_BIN:/usr/bin:/bin" HOME="$CASE_HOME" \
  AUTODL_REPAIR_TESTING=1 AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
  AUTODL_WORK_DIR="$CASE_WORK" ISAACLAB_DIR="$CASE_ISAAC" \
  CONDA_BIN="$CASE_BIN/conda" CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
  NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
  NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
  NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
  NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
  PS_BIN="$CASE_BIN/ps" KILL_BIN="$CASE_BIN/fake-kill" \
  SLEEP_BIN="$CASE_BIN/sleep" TIMEOUT_BIN="$CASE_BIN/timeout" \
  FAKE_CALLS="$CASE_CALLS" FAKE_KILL_LOG="$CASE_KILL_LOG" \
  FAKE_PS_SNAPSHOT="$CASE_PS" FAKE_AUTODL_WORK="$CASE_WORK" \
  FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
  bash "$SCRIPT" --config "$CASE_REPO/configs/experiments/full_suite.json" \
    --report-dir "$CASE_REPORT_BASE" --skip-smoke --skip-dry-run \
    >"$CASE_DIR/idempotent-second.out" 2>&1
second_status=$?
set -e
[[ "$second_status" == 0 ]] || fail "second idempotence run failed"
assert_count "$CASE_HOME/.bashrc" 1 "# >>> NavDP AutoDL runtime >>>"
[[ "$(sha256sum "$CASE_ENV_FILE" | awk '{print $1}')" == "$first_env_hash" ]] || \
  fail "runtime environment changed on identical second run"

history_dir="$CASE_RESULTS/fixture_suite/run/logs"
mkdir -p "$history_dir"
cat >"$history_dir/isaac_eval.stderr.log" <<'LOG'
GPU0:
        deviceName         = NVIDIA GeForce RTX 4090
GPU2:
        deviceName         = NVIDIA GeForce RTX 4090
GPU Foundation is not initialized!
eval_pointgoal_wheeled.py: error: unrecognized arguments: --old-option
CUDA out of memory
OmniGraphSettings::getCudaDeviceOrdinal: unable to get a valid CUDA device id from the renderer. Defaulting to GPU0.
LOG
cat >"$history_dir/isaac_eval.stdout.log" <<'LOG'
[3.557s] app ready
[FrameCheck] camera-base offset
[ControlRef] env=0
[EpisodeDone] env=0
LOG
printf '{"status":"FAILED"}\n' >"$CASE_RESULTS/fixture_suite/run/run_status.json"

set +e
env \
  PATH="$CASE_BIN:/usr/bin:/bin" HOME="$CASE_HOME" \
  AUTODL_REPAIR_TESTING=1 AUTODL_REPAIR_REPO_ROOT="$CASE_REPO" \
  AUTODL_WORK_DIR="$CASE_WORK" ISAACLAB_DIR="$CASE_ISAAC" \
  CONDA_BIN="$CASE_BIN/conda" CONDA_ENVS_PATH="$CASE_WORK/conda/envs" \
  NAVDP_RUNTIME_ENV_FILE="$CASE_ENV_FILE" \
  NAVDP_VULKAN_ICD_DIRS="$CASE_ETC_ICD:$CASE_USR_ICD" \
  NAVDP_RESULTS_ROOT="$CASE_RESULTS" \
  NVIDIA_SMI_BIN="$CASE_BIN/nvidia-smi" VULKANINFO_BIN="$CASE_BIN/vulkaninfo" \
  PS_BIN="$CASE_BIN/ps" KILL_BIN="$CASE_BIN/fake-kill" \
  SLEEP_BIN="$CASE_BIN/sleep" TIMEOUT_BIN="$CASE_BIN/timeout" \
  FAKE_CALLS="$CASE_CALLS" FAKE_KILL_LOG="$CASE_KILL_LOG" \
  FAKE_PS_SNAPSHOT="$CASE_PS" FAKE_AUTODL_WORK="$CASE_WORK" \
  FAKE_DRY_RUN_SOURCE="$CASE_PLAN_SOURCE" FAKE_DRY_RUN_DEST="$CASE_PLAN_DEST" \
  bash "$SCRIPT" --config "$CASE_REPO/configs/experiments/full_suite.json" \
    --report-dir "$CASE_REPORT_BASE" --skip-smoke --skip-dry-run \
    >"$CASE_DIR/history.out" 2>&1
history_status=$?
set -e
[[ "$history_status" == 0 ]] || fail "historical errors must not override healthy current checks"
latest_report="$(sed -n 's/^Report: //p' "$CASE_DIR/history.out" | tail -n 1)"
assert_contains "$latest_report/historical-diagnostics.txt" "DUPLICATE_VULKAN_ICD"
assert_contains "$latest_report/historical-diagnostics.txt" "ARGPARSE_MISMATCH"
assert_contains "$latest_report/historical-diagnostics.txt" "CUDA_OOM"
assert_contains "$latest_report/historical-diagnostics.txt" "OMNIGRAPH_GPU0_FALLBACK"
assert_contains "$latest_report/historical-diagnostics.txt" "EPISODE_DONE"
assert_contains "$CASE_DIR/history.out" "--resume --retry-failed --allow-real-simulation"

if grep -R -F -- \
  "python -m experiments run-suite" "$TEST_TMP"/*/calls 2>/dev/null |
  grep -F -- "--allow-real-simulation" >/dev/null; then
  fail "self-check executed a real simulation command"
fi

printf 'PASS: AutoDL self-check repair integration tests\n'
