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

assert_count() {
  local file="$1"
  local expected_count="$2"
  local pattern="$3"
  local actual_count
  actual_count="$(grep -F -c -- "$pattern" "$file" || true)"
  [[ "$actual_count" == "$expected_count" ]] || \
    fail "expected $expected_count occurrences of '$pattern' in $file, got $actual_count"
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
printf 'omni-kit-accept-eula %s\n' "${OMNI_KIT_ACCEPT_EULA:-unset}" >>"$FAKE_CALLS"
if [[ "${1:-}" == "env" && "${2:-}" == "list" && "${3:-}" == "--json" ]]; then
  printf '{"envs":["/fake/base"'
  if [[ ",${FAKE_EXISTING_ENVS:-}," == *,navdp,* ]]; then printf ',"/fake/envs/navdp"'; fi
  if [[ ",${FAKE_EXISTING_ENVS:-}," == *,isaaclab,* ]]; then printf ',"/fake/envs/isaaclab"'; fi
  printf ']}\n'
  exit 0
fi
if [[ "${1:-}" == "run" ]]; then
  if [[ "$*" == *"pip install -r /tmp/"* && -n "${FAKE_CAPTURED_REQUIREMENTS:-}" ]]; then
    requirements_path="${@: -1}"
    cp "$requirements_path" "$FAKE_CAPTURED_REQUIREMENTS"
  fi
  if [[ "$*" == *"isaaclab.sh -i"* && "$*" != *"isaaclab.sh -i none"* && "${FAKE_RSL_RL_TLS_FAILURE:-0}" == 1 ]]; then
    printf "Running command git clone https://github.com/leggedrobotics/rsl_rl.git\n" >&2
    printf "fatal: unable to access rsl_rl.git: GnuTLS recv error (-110)\n" >&2
    printf "ERROR: Failed to build 'rsl-rl' when git clone failed\n" >&2
    exit 1
  fi
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
  count=0
  if [[ -f "${FAKE_GIT_COUNTER:-}" ]]; then
    count="$(cat "$FAKE_GIT_COUNTER")"
  fi
  if ((count < ${FAKE_GIT_CLONE_FAILURES:-0})); then
    printf '%s\n' "$((count + 1))" >"$FAKE_GIT_COUNTER"
    printf 'fatal: early EOF\n' >&2
    exit 128
  fi
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

  cat >"$bin_dir/sleep" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'sleep %s\n' "$*" >>"$FAKE_CALLS"
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
if [[ "$*" == *"vulkaninfo --summary"* ]]; then
  printf 'vulkaninfo icd=%s\n' "${VK_ICD_FILENAMES:-unset}" >>"$FAKE_CALLS"
  case "${FAKE_VULKAN_MODE:-healthy}" in
    healthy)
      printf 'deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n'
      printf 'deviceName = NVIDIA RTX 4090\n'
      printf 'driverName = NVIDIA\n'
      ;;
    glx-fails-egl-works)
      if [[ -n "${VK_ICD_FILENAMES:-}" &&
            -f "$VK_ICD_FILENAMES" &&
            "$(grep -c 'libEGL_nvidia.so.0' "$VK_ICD_FILENAMES" || true)" == 1 ]]; then
        printf 'deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n'
        printf 'deviceName = NVIDIA RTX 4090\n'
        printf 'driverName = NVIDIA\n'
      else
        printf "ERROR: Could not get 'vkCreateInstance' for ICD libGLX_nvidia.so.0\n"
        printf 'deviceType = PHYSICAL_DEVICE_TYPE_CPU\n'
        printf 'deviceName = llvmpipe\n'
      fi
      ;;
    repair-still-llvmpipe)
      printf 'deviceType = PHYSICAL_DEVICE_TYPE_CPU\n'
      printf 'deviceName = llvmpipe\n'
      ;;
    *)
      printf 'unknown FAKE_VULKAN_MODE=%s\n' "$FAKE_VULKAN_MODE" >&2
      exit 2
      ;;
  esac
fi
exit 0
EOF

  cat >"$bin_dir/vulkaninfo" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

  cat >"$bin_dir/ldconfig" <<'EOF'
#!/usr/bin/env bash
printf 'libGLU.so.1\nlibSM.so.6\nlibXt.so.6\n'
if [[ "${FAKE_EGL_AVAILABLE:-1}" == 1 ]]; then
  printf 'libEGL_nvidia.so.0 (libc6,x86-64) => %s\n' "$FAKE_EGL_LIBRARY_PATH"
fi
EOF

  cat >"$bin_dir/apt-get" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'apt-get %s\n' "$*" >>"$FAKE_CALLS"
if [[ " $* " == *" libeigen3-dev "* ]]; then
  mkdir -p "$NAVDP_EIGEN_INCLUDE_DIR/Eigen"
  : >"$NAVDP_EIGEN_INCLUDE_DIR/Eigen/Core"
fi
EOF

  chmod +x "$bin_dir/conda" "$bin_dir/git" "$bin_dir/nvidia-smi" "$bin_dir/timeout" "$bin_dir/vulkaninfo" "$bin_dir/ldconfig" "$bin_dir/apt-get" "$bin_dir/df" "$bin_dir/sleep"
}

run_script() {
  local case_dir="$1"
  shift
  mkdir -p "$case_dir/bin" "$case_dir/home" "$case_dir/data" "$case_dir/lib" \
    "$case_dir/system/include/eigen3/Eigen"
  : >"$case_dir/calls"
  : >"$case_dir/lib/libEGL_nvidia.so.0"
  : >"$case_dir/system/include/eigen3/Eigen/Core"
  make_fake_tools "$case_dir/bin"
  env \
    PATH="$case_dir/bin:/usr/bin:/bin" \
    HOME="$case_dir/home" \
    FAKE_CALLS="$case_dir/calls" \
    AUTODL_MIN_FREE_GB=0 \
    AUTODL_EXPORT_DIR="$case_dir/export" \
    AUTODL_WORK_DIR="$case_dir/data" \
    FAKE_DATA_DISK="$case_dir/data" \
    FAKE_EGL_LIBRARY_PATH="$case_dir/lib/libEGL_nvidia.so.0" \
    NAVDP_HEADLESS_VULKAN_ICD_PATH="$case_dir/etc/vulkan/icd.d/navdp_nvidia_headless_icd.json" \
    NAVDP_EIGEN_INCLUDE_DIR="$case_dir/system/include/eigen3" \
    "$@"
}

[[ -f "$SCRIPT" ]] || fail "setup script is missing: $SCRIPT"

help_dir="$TEST_TMP/help"
mkdir -p "$help_dir"
run_script "$help_dir" bash "$SCRIPT" --help >"$help_dir/out" 2>&1
assert_contains "$help_dir/out" "Usage:"
assert_contains "$help_dir/out" "--check-only"
assert_contains "$help_dir/out" "--accept-isaac-eula"

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
[[ ! -e "$check_dir/etc/vulkan/icd.d/navdp_nvidia_headless_icd.json" ]] || \
  fail "healthy Vulkan preflight must not create a replacement ICD"

headless_check_dir="$TEST_TMP/headless-check"
mkdir -p "$headless_check_dir"
if ! run_script "$headless_check_dir" \
  FAKE_VULKAN_MODE=glx-fails-egl-works \
  bash "$SCRIPT" --check-only >"$headless_check_dir/out" 2>&1; then
  sed -n '1,240p' "$headless_check_dir/out" >&2
  fail "AutoDL headless EGL repair should be discoverable in --check-only mode"
fi
assert_contains "$headless_check_dir/out" "AutoDL headless NVIDIA Vulkan repair is available"
assert_contains "$headless_check_dir/out" "Preflight checks passed"
[[ ! -e "$headless_check_dir/etc/vulkan/icd.d/navdp_nvidia_headless_icd.json" ]] || \
  fail "--check-only must not persist the generated EGL ICD"

headless_repair_dir="$TEST_TMP/headless-repair"
mkdir -p "$headless_repair_dir"
run_script "$headless_repair_dir" \
  FAKE_VULKAN_MODE=glx-fails-egl-works \
  ISAACLAB_DIR="$headless_repair_dir/IsaacLab" \
  bash "$SCRIPT" --skip-verify >"$headless_repair_dir/out" 2>&1
headless_icd="$headless_repair_dir/etc/vulkan/icd.d/navdp_nvidia_headless_icd.json"
[[ -f "$headless_icd" ]] || fail "normal setup must persist the generated EGL ICD"
assert_contains "$headless_icd" '"library_path": "'"$headless_repair_dir/lib/libEGL_nvidia.so.0"'"'
assert_contains "$headless_repair_dir/out" "Configured AutoDL headless NVIDIA Vulkan ICD"
assert_contains "$headless_repair_dir/calls" "vulkaninfo icd=$headless_icd"
first_icd_hash="$(sha256sum "$headless_icd" | awk '{print $1}')"
run_script "$headless_repair_dir" \
  FAKE_VULKAN_MODE=glx-fails-egl-works \
  ISAACLAB_DIR="$headless_repair_dir/IsaacLab" \
  bash "$SCRIPT" --skip-verify >"$headless_repair_dir/second-out" 2>&1
second_icd_hash="$(sha256sum "$headless_icd" | awk '{print $1}')"
[[ "$first_icd_hash" == "$second_icd_hash" ]] || \
  fail "repeated setup must leave the generated EGL ICD unchanged"

missing_egl_dir="$TEST_TMP/missing-egl"
mkdir -p "$missing_egl_dir"
if run_script "$missing_egl_dir" \
  FAKE_VULKAN_MODE=glx-fails-egl-works \
  FAKE_EGL_AVAILABLE=0 \
  bash "$SCRIPT" --check-only >"$missing_egl_dir/out" 2>&1; then
  fail "Vulkan repair without libEGL_nvidia.so.0 must fail closed"
fi
assert_contains "$missing_egl_dir/out" "libEGL_nvidia.so.0 was not found"

failed_repair_dir="$TEST_TMP/failed-repair"
mkdir -p "$failed_repair_dir"
if run_script "$failed_repair_dir" \
  FAKE_VULKAN_MODE=repair-still-llvmpipe \
  bash "$SCRIPT" --check-only >"$failed_repair_dir/out" 2>&1; then
  fail "Vulkan repair that still exposes llvmpipe must fail closed"
fi
assert_contains "$failed_repair_dir/out" "EGL ICD validation failed"

eigen_install_dir="$TEST_TMP/eigen-install"
mkdir -p "$eigen_install_dir"
missing_eigen_include="$eigen_install_dir/missing/include/eigen3"
run_script "$eigen_install_dir" \
  NAVDP_EIGEN_INCLUDE_DIR="$missing_eigen_include" \
  ISAACLAB_DIR="$eigen_install_dir/IsaacLab" \
  bash "$SCRIPT" --skip-verify >"$eigen_install_dir/out" 2>&1
assert_contains "$eigen_install_dir/calls" "apt-get install -y libeigen3-dev"
[[ -f "$missing_eigen_include/Eigen/Core" ]] || \
  fail "native dependency installation must provide the Eigen/Core header"
assert_contains "$eigen_install_dir/out" "Installing missing native build dependencies: libeigen3-dev"

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
  FAKE_CAPTURED_REQUIREMENTS="$full_dir/benchmark-requirements" \
  bash "$SCRIPT" --accept-isaac-eula >"$full_dir/out" 2>&1
assert_contains "$full_dir/calls" "env create -n navdp"
assert_contains "$full_dir/calls" "env create -n isaaclab"
assert_contains "$full_dir/calls" "conda run --no-capture-output -n navdp python -m pip install"
assert_contains "$full_dir/calls" "conda run --no-capture-output -n isaaclab python -m pip install"
assert_contains "$full_dir/calls" "conda-envs-path $full_dir/data/conda/envs"
assert_contains "$full_dir/calls" "conda-pkgs-dirs $full_dir/data/conda/pkgs"
assert_contains "$full_dir/calls" "pip-cache-dir $full_dir/data/pip-cache"
assert_contains "$full_dir/calls" "omni-kit-accept-eula YES"
assert_contains "$full_dir/calls" "isaacsim==4.2.0.2"
assert_contains "$full_dir/calls" "isaacsim-extscache-physics==4.2.0.2"
assert_contains "$full_dir/calls" "checkout --detach v1.2.0"
assert_contains "$full_dir/calls" "baselines/navdp/requirements.txt"
assert_contains "$full_dir/out" "Installing NavDP benchmark requirements"
assert_not_contains "$full_dir/benchmark-requirements" "isaaclab=="
assert_not_contains "$full_dir/benchmark-requirements" "rsl-rl-lib=="
assert_not_contains "$full_dir/benchmark-requirements" "triton=="
assert_not_contains "$full_dir/benchmark-requirements" "warp-lang=="
assert_not_contains "$full_dir/benchmark-requirements" "packaging=="
assert_contains "$full_dir/benchmark-requirements" "s3transfer==0.6.1"
assert_contains "$full_dir/calls" "python -m pip install torch==2.4.0 triton==3.0.0"
assert_contains "$full_dir/calls" "python -m pip install rsl-rl-lib==2.3.1"
assert_contains "$full_dir/calls" "python -m pip install packaging>=24.0 boto3==1.26.76 botocore==1.29.76 s3transfer==0.6.1"
assert_contains "$full_dir/calls" "python -m pip check"
assert_contains "$full_dir/calls" "pip freeze"
assert_contains "$full_dir/calls" "timeout"
assert_contains "$full_dir/calls" "bash $full_dir/IsaacLab/isaaclab.sh -p"
assert_contains "$full_dir/out" "Setup complete"

retry_dir="$TEST_TMP/retry"
mkdir -p "$retry_dir"
if ! run_script "$retry_dir" \
  ISAACLAB_DIR="$retry_dir/IsaacLab" \
  FAKE_GIT_CLONE_FAILURES=2 \
  FAKE_GIT_COUNTER="$retry_dir/git-counter" \
  bash "$SCRIPT" --skip-verify >"$retry_dir/out" 2>&1; then
  sed -n '1,240p' "$retry_dir/out" >&2
  fail "clone retry scenario should recover"
fi
assert_count "$retry_dir/calls" 3 "git clone"
assert_contains "$retry_dir/calls" "git clone --branch v1.2.0 --depth 1 --single-branch"
assert_contains "$retry_dir/out" "Git operation failed (attempt 1/3)"
assert_contains "$retry_dir/out" "Git operation failed (attempt 2/3)"
[[ -x "$retry_dir/IsaacLab/isaaclab.sh" ]] || fail "retried clone was not installed atomically"

incomplete_dir="$TEST_TMP/incomplete"
mkdir -p "$incomplete_dir/IsaacLab/.git"
if ! run_script "$incomplete_dir" \
  ISAACLAB_DIR="$incomplete_dir/IsaacLab" \
  bash "$SCRIPT" --skip-verify >"$incomplete_dir/out" 2>&1; then
  sed -n '1,240p' "$incomplete_dir/out" >&2
  fail "incomplete clone should be preserved and replaced"
fi
assert_contains "$incomplete_dir/out" "Preserving incomplete IsaacLab checkout"
compgen -G "$incomplete_dir/IsaacLab.incomplete.*" >/dev/null || \
  fail "incomplete checkout archive was not created"
[[ -x "$incomplete_dir/IsaacLab/isaaclab.sh" ]] || fail "incomplete checkout was not replaced"

local_source_dir="$TEST_TMP/local-source"
mkdir -p "$local_source_dir/IsaacLab/.git/objects/pack" "$local_source_dir/IsaacLab/source"
cat >"$local_source_dir/IsaacLab/isaaclab.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
# Deliberately leave isaaclab.sh non-executable: it is invoked through bash.
if ! run_script "$local_source_dir" \
  ISAACLAB_DIR="$local_source_dir/IsaacLab" \
  ISAACLAB_USE_LOCAL_SOURCE=1 \
  bash "$SCRIPT" --skip-verify >"$local_source_dir/out" 2>&1; then
  sed -n '1,240p' "$local_source_dir/out" >&2
  fail "manual IsaacLab source mode should ignore corrupt Git metadata"
fi
assert_contains "$local_source_dir/out" "Using manually provided IsaacLab source"
assert_not_contains "$local_source_dir/calls" "git -C $local_source_dir/IsaacLab"
assert_not_contains "$local_source_dir/calls" "git clone"
assert_contains "$local_source_dir/calls" "bash $local_source_dir/IsaacLab/isaaclab.sh -i none"

rsl_tls_dir="$TEST_TMP/rsl-tls"
mkdir -p "$rsl_tls_dir/IsaacLab/source"
cat >"$rsl_tls_dir/IsaacLab/isaaclab.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
if ! run_script "$rsl_tls_dir" \
  ISAACLAB_DIR="$rsl_tls_dir/IsaacLab" \
  ISAACLAB_USE_LOCAL_SOURCE=1 \
  FAKE_RSL_RL_TLS_FAILURE=1 \
  bash "$SCRIPT" --skip-verify >"$rsl_tls_dir/out" 2>&1; then
  sed -n '1,240p' "$rsl_tls_dir/out" >&2
  fail "rsl-rl-only TLS clone failure should follow the README exception"
fi
assert_contains "$rsl_tls_dir/calls" "bash $rsl_tls_dir/IsaacLab/isaaclab.sh -i none"
assert_not_contains "$rsl_tls_dir/out" "leggedrobotics/rsl_rl.git"
assert_contains "$rsl_tls_dir/out" "Setup complete"

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
assert_not_contains "$custom_dir/calls" "isaaclab.sh -p"

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
