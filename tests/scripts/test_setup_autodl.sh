#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/setup_autodl.sh"
TEST_TMP="$(mktemp -d)"
trap 'rm -rf "$TEST_TMP"' EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_contains() {
  local file="$1"
  local expected="$2"
  grep -F -- "$expected" "$file" >/dev/null || fail "expected '$expected' in $file"
}

assert_not_contains() {
  local file="$1"
  local unexpected="$2"
  if grep -F -- "$unexpected" "$file" >/dev/null; then
    sed -n '1,240p' "$file" >&2
    fail "did not expect '$unexpected' in $file"
  fi
}

make_fake_tools() {
  local bin_dir="$1"
  mkdir -p "$bin_dir"

  cat >"$bin_dir/conda" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'conda %s\n' "$*" >>"$FAKE_CALLS"
printf 'conda-envs-path %s\n' "${CONDA_ENVS_PATH:-unset}" >>"$FAKE_CALLS"
printf 'conda-pkgs-dirs %s\n' "${CONDA_PKGS_DIRS:-unset}" >>"$FAKE_CALLS"
printf 'pip-cache-dir %s\n' "${PIP_CACHE_DIR:-unset}" >>"$FAKE_CALLS"
if [[ "${1:-}" == "env" && "${2:-}" == "list" && "${3:-}" == "--json" ]]; then
  printf '{"envs":["/fake/base"'
  if [[ ",${FAKE_EXISTING_ENVS:-}," == *,navdp,* ]]; then printf ',"/fake/envs/navdp"'; fi
  if [[ ",${FAKE_EXISTING_ENVS:-}," == *,isaaclab,* ]]; then printf ',"/fake/envs/isaaclab"'; fi
  printf ']}\n'
  exit 0
fi
if [[ "${1:-}" == "run" ]]; then
  if [[ "$*" == *"pip freeze"* ]]; then
    printf 'fake-package==1.0\n'
  fi
  exit 0
fi
exit 0
EOF

  cat >"$bin_dir/df" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'df %s\n' "$*" >>"$FAKE_CALLS"
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n'
if [[ "${@: -1}" == "$FAKE_DATA_DISK"* ]]; then
  printf '/dev/data 104857600 128 104857472 1%% /data\n'
else
  printf '/dev/root 31457280 29457280 2000000 94%% /\n'
fi
EOF

  cat >"$bin_dir/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'git %s\n' "$*" >>"$FAKE_CALLS"
if [[ "${1:-}" == "clone" ]]; then
  target="${@: -1}"
  mkdir -p "$target/.git"
  cat >"$target/isaaclab.sh" <<'SCRIPT'
#!/usr/bin/env bash
exit 0
SCRIPT
  chmod +x "$target/isaaclab.sh"
fi
if [[ "$*" == *"status --porcelain"* ]]; then
  exit 0
fi
exit 0
EOF

  cat >"$bin_dir/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'nvidia-smi %s\n' "$*" >>"$FAKE_CALLS"
printf 'NVIDIA RTX 4090, 550.54.15, 24564 MiB\n'
EOF

  cat >"$bin_dir/timeout" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'timeout %s\n' "$*" >>"$FAKE_CALLS"
exit 0
EOF

  chmod +x "$bin_dir/conda" "$bin_dir/git" "$bin_dir/nvidia-smi" "$bin_dir/timeout" "$bin_dir/df"
}

run_script() {
  local case_dir="$1"
  shift
  mkdir -p "$case_dir/bin" "$case_dir/home" "$case_dir/data"
  : >"$case_dir/calls"
  make_fake_tools "$case_dir/bin"
  env \
    PATH="$case_dir/bin:/usr/bin:/bin" \
    HOME="$case_dir/home" \
    FAKE_CALLS="$case_dir/calls" \
    AUTODL_MIN_FREE_GB=0 \
    AUTODL_EXPORT_DIR="$case_dir/export" \
    AUTODL_WORK_DIR="$case_dir/data" \
    FAKE_DATA_DISK="$case_dir/data" \
    "$@"
}

[[ -f "$SCRIPT" ]] || fail "setup script is missing: $SCRIPT"

help_dir="$TEST_TMP/help"
mkdir -p "$help_dir"
run_script "$help_dir" bash "$SCRIPT" --help >"$help_dir/out" 2>&1
assert_contains "$help_dir/out" "Usage:"
assert_contains "$help_dir/out" "--check-only"

invalid_dir="$TEST_TMP/invalid"
mkdir -p "$invalid_dir"
if run_script "$invalid_dir" bash "$SCRIPT" --unknown >"$invalid_dir/out" 2>&1; then
  fail "unknown option should fail"
fi
assert_contains "$invalid_dir/out" "unknown option"

check_dir="$TEST_TMP/check"
mkdir -p "$check_dir"
run_script "$check_dir" bash "$SCRIPT" --check-only >"$check_dir/out" 2>&1
assert_contains "$check_dir/out" "Preflight checks passed"
assert_contains "$check_dir/calls" "df -Pk $check_dir/data"
assert_not_contains "$check_dir/calls" "env create"
assert_not_contains "$check_dir/calls" "pip install"

autodl_disk_dir="$TEST_TMP/autodl-disk"
mkdir -p "$autodl_disk_dir"
run_script "$autodl_disk_dir" \
  AUTODL_MIN_FREE_GB=35 \
  bash "$SCRIPT" --check-only >"$autodl_disk_dir/out" 2>&1
assert_contains "$autodl_disk_dir/out" "Preflight checks passed"
assert_contains "$autodl_disk_dir/calls" "df -Pk $autodl_disk_dir/data"

missing_dir="$TEST_TMP/missing"
mkdir -p "$missing_dir/bin" "$missing_dir/home"
: >"$missing_dir/calls"
make_fake_tools "$missing_dir/bin"
rm "$missing_dir/bin/conda"
if env PATH="$missing_dir/bin:/usr/bin:/bin" HOME="$missing_dir/home" FAKE_CALLS="$missing_dir/calls" \
  AUTODL_MIN_FREE_GB=0 bash "$SCRIPT" --check-only >"$missing_dir/out" 2>&1; then
  fail "missing conda should fail"
fi
assert_contains "$missing_dir/out" "conda"

full_dir="$TEST_TMP/full"
mkdir -p "$full_dir"
run_script "$full_dir" \
  ISAACLAB_DIR="$full_dir/IsaacLab" \
  bash "$SCRIPT" >"$full_dir/out" 2>&1
assert_contains "$full_dir/calls" "env create -n navdp"
assert_contains "$full_dir/calls" "env create -n isaaclab"
assert_contains "$full_dir/calls" "conda run --no-capture-output -n navdp python -m pip install"
assert_contains "$full_dir/calls" "conda run --no-capture-output -n isaaclab python -m pip install"
assert_contains "$full_dir/calls" "conda-envs-path $full_dir/data/conda/envs"
assert_contains "$full_dir/calls" "conda-pkgs-dirs $full_dir/data/conda/pkgs"
assert_contains "$full_dir/calls" "pip-cache-dir $full_dir/data/pip-cache"
assert_contains "$full_dir/calls" "isaacsim==4.2.0.2"
assert_contains "$full_dir/calls" "isaacsim-extscache-physics==4.2.0.2"
assert_contains "$full_dir/calls" "checkout --detach v1.2.0"
assert_contains "$full_dir/calls" "baselines/navdp/requirements.txt"
assert_contains "$full_dir/out" "Installing NavDP benchmark requirements"
assert_contains "$full_dir/calls" "pip freeze"
assert_contains "$full_dir/calls" "timeout"
assert_contains "$full_dir/out" "Setup complete"

custom_dir="$TEST_TMP/custom"
mkdir -p "$custom_dir"
run_script "$custom_dir" \
  NAVDP_ENV_NAME=model \
  ISAACLAB_ENV_NAME=sim \
  ISAACLAB_DIR="$custom_dir/existing-lab" \
  FAKE_EXISTING_ENVS=navdp,isaaclab \
  bash "$SCRIPT" --skip-verify >"$custom_dir/out" 2>&1
assert_contains "$custom_dir/calls" "env create -n model"
assert_contains "$custom_dir/calls" "env create -n sim"
assert_not_contains "$custom_dir/calls" "timeout"

reuse_dir="$TEST_TMP/reuse"
mkdir -p "$reuse_dir/IsaacLab/.git"
cat >"$reuse_dir/IsaacLab/isaaclab.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$reuse_dir/IsaacLab/isaaclab.sh"
run_script "$reuse_dir" \
  ISAACLAB_DIR="$reuse_dir/IsaacLab" \
  FAKE_EXISTING_ENVS=navdp,isaaclab \
  bash "$SCRIPT" --skip-verify >"$reuse_dir/out" 2>&1
assert_not_contains "$reuse_dir/calls" "env create"
assert_not_contains "$reuse_dir/calls" "git clone"
assert_contains "$reuse_dir/out" "Reusing Conda environment: navdp"
assert_contains "$reuse_dir/out" "Reusing IsaacLab checkout"

printf 'PASS: setup_autodl integration tests\n'
