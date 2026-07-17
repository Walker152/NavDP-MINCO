# AutoDL Self-Check Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify one idempotent AutoDL self-check/repair command that fixes duplicate NVIDIA Vulkan ICD selection, removes stale NavDP/Isaac processes, validates CUDA/Isaac/runtime command contracts, performs a zero-process experiment dry-run, and writes an auditable report.

**Architecture:** A single Bash entrypoint exposes small stage functions and injects external command paths so shell integration tests can exercise real control flow without a GPU. The script persists only reversible runtime environment configuration, never edits driver manifests or runtime source, and delegates JSON contract validation to Python inside the configured Conda environment.

**Tech Stack:** Bash 5.1, standard Linux CLI tools, Conda, Python 3 JSON validation, existing `experiments` CLI, shell integration tests.

---

## File map

- Create `scripts/autodl_self_check_repair.sh`: production CLI, stage orchestration, repair logic, report generation.
- Create `tests/scripts/test_autodl_self_check_repair.sh`: fake-tool integration harness and regression scenarios.
- Modify `README.md`: one-command usage, options, persistent environment file, reports, exit codes.
- Reference `docs/superpowers/specs/2026-07-17-autodl-self-check-repair-design.md`: approved behavioral contract.

### Task 1: CLI, report lifecycle, and test harness

**Files:**
- Create: `tests/scripts/test_autodl_self_check_repair.sh`
- Create: `scripts/autodl_self_check_repair.sh`

- [ ] **Step 1: Write the failing CLI/report tests**

Create a shell harness with `fail`, `assert_contains`, `assert_not_contains`, `assert_count`, `make_fixture_repo`, `make_fake_tools`, and `run_repair`. The first cases must assert:

```bash
[[ -f "$SCRIPT" ]] || fail "repair script is missing: $SCRIPT"

bash "$SCRIPT" --help >"$TEST_TMP/help.out" 2>&1
assert_contains "$TEST_TMP/help.out" "Usage:"
assert_contains "$TEST_TMP/help.out" "--check-only"
assert_contains "$TEST_TMP/help.out" "--skip-smoke"
assert_contains "$TEST_TMP/help.out" "--skip-dry-run"

if bash "$SCRIPT" --unknown >"$TEST_TMP/unknown.out" 2>&1; then
  fail "unknown option should return exit 2"
else
  [[ "$?" == 2 ]] || fail "unknown option did not return exit 2"
fi

run_repair healthy --skip-smoke --skip-dry-run
assert_contains "$CASE_OUT" "[PASS]"
assert_contains "$CASE_OUT" "Report:"
[[ -f "$CASE_REPORT/summary.txt" ]] || fail "summary report missing"
```

The fixture repository must contain minimal files for every path checked by preflight, including a fixture `eval_pointgoal_wheeled.py`, backend file, suite JSON, `isaaclab.sh`, and `create_empty.py`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: `FAIL: repair script is missing`.

- [ ] **Step 3: Implement the CLI and report skeleton**

Create the production script with:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECK_ONLY=0
SKIP_SMOKE=0
SKIP_DRY_RUN=0
SMOKE_TIMEOUT=180
CONFIG="$REPO_ROOT/configs/experiments/full_suite.json"
REPORT_BASE="$REPO_ROOT/results/autodl_self_check"
PASS_COUNT=0
REPAIR_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
CURRENT_STAGE="startup"

usage() {
  cat <<'EOF'
Usage: bash scripts/autodl_self_check_repair.sh [OPTIONS]

Options:
  --check-only             Diagnose without writing configuration or killing processes
  --kill-stale             Compatibility flag; stale cleanup is already enabled by default
  --skip-smoke             Skip the Isaac headless GPU smoke test
  --skip-dry-run           Skip experiment command generation and validation
  --config PATH            Suite config (default: configs/experiments/full_suite.json)
  --report-dir PATH        Base report directory
  --smoke-timeout SECONDS  Isaac smoke timeout (default: 180)
  -h, --help               Show this help
EOF
}
```

Parse all arguments before creating reports. Unknown options and invalid positive integer timeouts return 2. Create a UTC-stamped report directory, initialize every report file from the specification, and implement `pass`, `repaired`, `warn`, and `fail_check` counters. Add `print_summary` on normal exit and a trap returning 130 on interruption.

Allow command injection through:

```bash
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
VULKANINFO_BIN="${VULKANINFO_BIN:-vulkaninfo}"
PS_BIN="${PS_BIN:-ps}"
KILL_BIN="${KILL_BIN:-kill}"
SLEEP_BIN="${SLEEP_BIN:-sleep}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
```

Use `AUTODL_REPAIR_REPO_ROOT` only in tests to replace the detected repository root; reject it unless `AUTODL_REPAIR_TESTING=1`.

- [ ] **Step 4: Run the CLI tests and verify GREEN**

Run:

```bash
bash -n scripts/autodl_self_check_repair.sh
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: initial CLI/report cases pass and the test stops at the next unimplemented behavior.

- [ ] **Step 5: Commit**

```bash
git add scripts/autodl_self_check_repair.sh
git add -f tests/scripts/test_autodl_self_check_repair.sh
git commit -m "test: define AutoDL repair CLI contract"
```

### Task 2: Safe stale-process cleanup

**Files:**
- Modify: `tests/scripts/test_autodl_self_check_repair.sh`
- Modify: `scripts/autodl_self_check_repair.sh`

- [ ] **Step 1: Add failing process-selection tests**

Feed a fake `ps` snapshot containing:

```text
101 1 101 101 900 /root/miniconda3/bin/conda run -n isaaclab bash /root/autodl-tmp/navdp/IsaacLab/isaaclab.sh -p /fixture/NavDP/eval_pointgoal_wheeled.py
102 101 101 101 890 /root/autodl-tmp/navdp/conda/envs/isaaclab/bin/python /fixture/NavDP/eval_pointgoal_wheeled.py
201 1 201 201 800 python /fixture/NavDP/baselines/navdp/navdp_server.py --port 8889
301 1 301 301 700 python /other/NavDP/eval_pointgoal_wheeled.py
401 1 401 401 600 python unrelated_worker.py
```

Use the internal test-mode signal recorder. Assert default mode records signals
for old PPID-1 roots 101/201 and matching descendant 102, never 301/401, never
executes an external kill backend, and check-only records candidates without
any signal:

```bash
assert_contains "$CASE_KILL_LOG" "-TERM"
assert_not_contains "$CASE_KILL_LOG" "301"
assert_not_contains "$CASE_KILL_LOG" "401"
assert_contains "$CASE_REPORT/process-scan.txt" "STALE_CANDIDATE"
```

- [ ] **Step 2: Run process tests and verify RED**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: no stale-process signals are recorded.

- [ ] **Step 3: Implement ancestry exclusion and cleanup**

Add functions:

```bash
collect_ancestor_pids()
scan_stale_processes()
signal_stale_processes()
```

`scan_stale_processes` parses `ps -eo pid=,ppid=,pgid=,sid=,etimes=,args=` and requires both:

```bash
[[ "$args" == *"$REPO_ROOT"* || "$args" == *"$ISAACLAB_DIR"* ]]
[[ "$args" =~ eval_pointgoal_wheeled\.py|omni\.kit|isaac-sim|navdp_server\.py|isaaclab\.sh ]]
```

Build the ancestor set by following PPID from the same snapshot. Exclude PID 1, `$$`, `$PPID`, all ancestors, and all non-target paths.

Require a PPID-1 root to be at least
`NAVDP_STALE_MIN_AGE_SECONDS` old, then add matching descendants through the
PPID graph. Revalidate each PID identity and use pidfd in production to send
`TERM`, wait up to five one-second polls, then send `KILL` only to still-live
unchanged targets. Do not signal a snapshot PGID. Test mode uses an internal
recorder/simulator and must never execute an external kill backend.
`--check-only` records candidates without sending any signal.

- [ ] **Step 4: Verify process tests pass**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: stale candidates are killed, unrelated and ancestor PIDs are preserved.

- [ ] **Step 5: Commit**

```bash
git add scripts/autodl_self_check_repair.sh
git add -f tests/scripts/test_autodl_self_check_repair.sh
git commit -m "feat: clean stale NavDP and Isaac processes"
```

### Task 3: Duplicate Vulkan ICD repair and persistence

**Files:**
- Modify: `tests/scripts/test_autodl_self_check_repair.sh`
- Modify: `scripts/autodl_self_check_repair.sh`

- [ ] **Step 1: Add failing Vulkan selection tests**

Create temporary `/etc`-like and `/usr/share`-like fixture directories and fake manifests. Configure `NAVDP_VULKAN_ICD_DIRS` as a colon-delimited test override.

Fake `vulkaninfo` must return:

- no ICD env: GPU0 RTX 4090, GPU1 llvmpipe, GPU2 RTX 4090;
- `/etc/.../nvidia_icd.json`: one RTX 4090;
- `/usr/.../nvidia_icd.json`: one RTX 4090;
- selected failure scenario: nonzero or llvmpipe only.

Assert:

```bash
assert_contains "$CASE_OUT" "[REPAIRED] Selected NVIDIA Vulkan ICD"
assert_contains "$CASE_ENV_FILE" "export VK_ICD_FILENAMES="
assert_contains "$CASE_ENV_FILE" "export VK_DRIVER_FILES="
assert_count "$CASE_HOME/.bashrc" 1 "# >>> NavDP AutoDL runtime >>>"
assert_count "$CASE_HOME/.bashrc" 1 "# <<< NavDP AutoDL runtime <<<"
```

Add cases for `/etc` preference, `/usr` fallback, no valid candidate failure, check-only no writes, and a second run that leaves exactly one managed block.

- [ ] **Step 2: Run Vulkan tests and verify RED**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: no ICD is selected and no runtime environment file exists.

- [ ] **Step 3: Implement ICD probing and atomic persistence**

Add:

```bash
record_gpu_environment()
discover_nvidia_icds()
probe_nvidia_icd()
select_nvidia_icd()
write_runtime_environment()
update_bashrc()
```

`probe_nvidia_icd` executes:

```bash
VK_ICD_FILENAMES="$candidate" \
VK_DRIVER_FILES="$candidate" \
"$TIMEOUT_BIN" 30 "$VULKANINFO_BIN" --summary
```

Validate exactly one NVIDIA `deviceName` line and reject duplicate names and llvmpipe-only output. Candidate ordering follows configured directories, defaulting to:

```bash
/etc/vulkan/icd.d:/usr/share/vulkan/icd.d
```

Write the runtime file through `mktemp` + `mv`, shell-quote all values with `printf '%q'`, then export the selected ICD in the current process. Rewrite the marked bashrc block with `awk`, preserving all unrelated content and creating one `.before-navdp-autodl` backup if none exists.

- [ ] **Step 4: Verify Vulkan tests pass**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: all ICD selection, fallback, check-only, and idempotence cases pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/autodl_self_check_repair.sh
git add -f tests/scripts/test_autodl_self_check_repair.sh
git commit -m "feat: repair duplicate NVIDIA Vulkan ICD selection"
```

### Task 4: Conda, CUDA, and runtime-source contracts

**Files:**
- Modify: `tests/scripts/test_autodl_self_check_repair.sh`
- Modify: `scripts/autodl_self_check_repair.sh`

- [ ] **Step 1: Add failing environment and source-contract tests**

Extend fake Conda behavior for:

- `conda env list --json`;
- IsaacLab torch CUDA success/failure;
- NavDP import success/failure;
- eval `--help` output.

Add cases asserting CUDA unavailable fails, missing NavDP imports fail, and a fixture backend missing `minco_start_validation_exemption_radius` reports mixed runtime versions. Assert SHA-256 lines are present.

- [ ] **Step 2: Run contract tests and verify RED**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: unhealthy fake Conda/source cases incorrectly pass or lack the required diagnostic.

- [ ] **Step 3: Implement preflight, CUDA, and source checks**

Add:

```bash
resolve_conda()
preflight()
check_conda_environments()
check_torch_cuda()
check_navdp_imports()
check_runtime_contract()
```

The CUDA Python program must exit nonzero unless all conditions hold:

```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
assert "NVIDIA" in torch.cuda.get_device_name(0).upper()
x = torch.ones(1, device="cuda:0")
assert float(x.item()) == 1.0
torch.cuda.synchronize()
print("CUDA_OK", torch.cuda.get_device_name(0))
```

The source contract checks every required token in both eval and backend where applicable, records:

```bash
sha256sum "$REPO_ROOT/eval_pointgoal_wheeled.py"
sha256sum "$REPO_ROOT/experiments/simulators/isaac_navdp_backend.py"
```

and runs eval `--help` in IsaacLab. Missing tokens must produce `远端运行时版本混用`.

- [ ] **Step 4: Verify environment and contract tests pass**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: healthy fixtures pass; each unhealthy fixture exits 1 with a specific report.

- [ ] **Step 5: Commit**

```bash
git add scripts/autodl_self_check_repair.sh
git add -f tests/scripts/test_autodl_self_check_repair.sh
git commit -m "feat: verify AutoDL CUDA and runtime contracts"
```

### Task 5: Isaac smoke lifecycle and fatal-log classification

**Files:**
- Modify: `tests/scripts/test_autodl_self_check_repair.sh`
- Modify: `scripts/autodl_self_check_repair.sh`

- [ ] **Step 1: Add failing smoke tests**

Make fake Conda emit configurable smoke logs:

Healthy:

```text
| 0 | NVIDIA GeForce RTX 4090 | Yes: 0 |
[3.557s] app ready
```

Fatal:

```text
Multiple Installable Client Drivers (ICDs)
Failed to create any GPU devices
GPU Foundation is not initialized!
```

Add one timeout-without-ready case. Assert healthy smoke passes and the launched process is cleaned up; fatal and no-ready cases fail.

- [ ] **Step 2: Run smoke tests and verify RED**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: smoke result is not classified.

- [ ] **Step 3: Implement monitored smoke execution**

Add:

```bash
start_isaac_smoke()
wait_for_smoke_result()
stop_smoke_group()
classify_isaac_smoke()
```

Launch the Conda command with `setsid` when available, redirecting output to `isaac-smoke.log`. Poll at 250 ms intervals until:

- a fatal pattern appears: stop and fail;
- both readiness and active NVIDIA patterns appear: stop and pass;
- process exits: classify accumulated log;
- timeout expires: stop and fail.

Fatal pattern matching is case-insensitive and includes all specification strings. Treat default-display, pre-SimulationApp import, and crash-reporter warnings as nonfatal.

- [ ] **Step 4: Verify smoke tests pass**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: healthy, fatal, and timeout scenarios classify correctly and leave no fake smoke process.

- [ ] **Step 5: Commit**

```bash
git add scripts/autodl_self_check_repair.sh
git add -f tests/scripts/test_autodl_self_check_repair.sh
git commit -m "feat: add monitored Isaac GPU smoke test"
```

### Task 6: Experiment dry-run and argv validation

**Files:**
- Modify: `tests/scripts/test_autodl_self_check_repair.sh`
- Modify: `scripts/autodl_self_check_repair.sh`

- [ ] **Step 1: Add failing dry-run contract tests**

Create valid and invalid `dry_run_plan.json` fixtures. The valid fixture includes one RAW and one MINCO command with 2 episodes. Invalid fixtures cover:

- `started_processes: 1`;
- `--minco_penalty_weight_attractor20.0`;
- 2 episodes but one NavDP seed;
- controller/variant mismatch;
- server/eval port mismatch.

Assert valid passes and every invalid fixture exits 1 with the corresponding invariant name.

- [ ] **Step 2: Run dry-run tests and verify RED**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: malformed plans are not rejected.

- [ ] **Step 3: Implement dry-run generation and validator**

Add:

```bash
run_experiment_dry_run()
locate_dry_run_plan()
validate_dry_run_plan()
```

Invoke the existing experiments CLI through NavDP Conda. Pass the generated plan to a Python heredoc that implements helpers:

```python
def value_after(command, option):
    index = command.index(option)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise AssertionError(f"{option} has no separate value")
    return command[index + 1]

def values_between(command, option, next_option):
    start = command.index(option) + 1
    end = command.index(next_option, start)
    return command[start:end]
```

Validate every invariant from the specification, including exact seed/episode counts and port equality. Print one `DRY_RUN_OK` line only after all commands pass.

- [ ] **Step 4: Verify dry-run tests pass**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: valid plan passes; all malformed plans fail with precise messages.

- [ ] **Step 5: Commit**

```bash
git add scripts/autodl_self_check_repair.sh
git add -f tests/scripts/test_autodl_self_check_repair.sh
git commit -m "feat: validate real experiment dry-run commands"
```

### Task 7: Historical diagnostics, documentation, and final audit

**Files:**
- Modify: `tests/scripts/test_autodl_self_check_repair.sh`
- Modify: `scripts/autodl_self_check_repair.sh`
- Modify: `README.md`

- [ ] **Step 1: Add failing historical-classification tests**

Create historical logs containing duplicate ICD, argparse mismatch, OOM, fatal, `app ready`, `FrameCheck`, `ControlRef`, and `EpisodeDone`, plus FAILED/RUNNING status JSON. Assert all categories are recorded, current healthy checks still return 0, and retry guidance contains:

```text
--resume --retry-failed --allow-real-simulation
```

- [ ] **Step 2: Run historical tests and verify RED**

Run:

```bash
bash tests/scripts/test_autodl_self_check_repair.sh
```

Expected: historical diagnostics report is empty or incomplete.

- [ ] **Step 3: Implement historical scanning and final summary**

Add:

```bash
find_latest_log()
classify_historical_logs()
write_final_summary()
```

Search under the configured suite output root, classify the exact patterns from the specification, and use warnings—not failures—for historical-only evidence. Write final counters, selected ICD, config, and report path to `summary.txt`. Exit 1 iff `FAIL_COUNT > 0`.

- [ ] **Step 4: Document the one-command workflow**

Add a README section with:

```bash
bash scripts/autodl_self_check_repair.sh
source "$HOME/.config/navdp/autodl-runtime.env"
```

Document check-only, skip flags, default stale-process termination, report location, safety boundaries, exit codes, and the generated retry command. Explicitly state that the script never deletes ICD JSON files and never launches a real experiment.

- [ ] **Step 5: Run all verification**

Run:

```bash
bash -n scripts/autodl_self_check_repair.sh
bash -n tests/scripts/test_autodl_self_check_repair.sh
bash tests/scripts/test_autodl_self_check_repair.sh
bash tests/scripts/test_setup_autodl.sh
python3 -m unittest experiments.tests.test_real_backend_static experiments.tests.test_pipeline_contracts -v
```

Expected: all commands pass. If `shellcheck` is installed, also run:

```bash
shellcheck scripts/autodl_self_check_repair.sh tests/scripts/test_autodl_self_check_repair.sh
```

Expected: no ShellCheck findings.

- [ ] **Step 6: Perform requirement-by-requirement completion audit**

Record evidence for every approved specification section:

- CLI and reports;
- strict stale process matching/default cleanup;
- duplicate ICD selection and persistence;
- CUDA/Conda;
- runtime source consistency;
- Isaac smoke;
- zero-process dry-run and argv invariants;
- historical diagnostics;
- idempotence and safety boundaries.

Confirm with:

```bash
git diff --check
git status --short
```

Ensure only the intended script, test, README, specification, and plan are attributed to this feature; preserve unrelated user changes.

- [ ] **Step 7: Commit**

```bash
git add scripts/autodl_self_check_repair.sh README.md
git add -f tests/scripts/test_autodl_self_check_repair.sh
git commit -m "feat: deliver AutoDL self-check repair workflow"
```
