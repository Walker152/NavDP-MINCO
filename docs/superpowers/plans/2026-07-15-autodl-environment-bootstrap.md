# AutoDL Environment Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested one-command AutoDL bootstrap that creates isolated NavDP and IsaacLab environments from the repository README.

**Architecture:** A strict Bash orchestrator delegates environment creation to two small Conda YAML files and invokes all environment commands through `conda run`. A Bash integration test injects fake host tools through `PATH`, allowing preflight, idempotency, version pins, exports, and options to be tested without network or GPU access.

**Tech Stack:** Bash, Conda environment YAML, fake-command integration tests, Markdown documentation

---

### Task 1: Define observable CLI behavior

**Files:**
- Create: `tests/scripts/test_setup_autodl.sh`

- [ ] Write a fake-tool integration harness that creates temporary `conda`, `git`, `nvidia-smi`, and `timeout` executables and records calls.
- [ ] Add failing checks for `--help`, unknown arguments, `--check-only`, missing required tools, default configuration, overridden configuration, version pins, environment reuse, verification skipping, and snapshot export.
- [ ] Run `bash tests/scripts/test_setup_autodl.sh` and confirm it fails because `scripts/setup_autodl.sh` does not exist.

### Task 2: Implement preflight and CLI orchestration

**Files:**
- Create: `scripts/setup_autodl.sh`

- [ ] Implement strict-mode argument parsing and documented environment-variable defaults.
- [ ] Resolve the repository root from the script location and validate Linux, Conda, Git, NVIDIA tooling, input files, environment names, target directory, and disk capacity.
- [ ] Implement `--help` and mutation-free `--check-only` behavior.
- [ ] Run the focused tests and confirm the CLI/preflight cases pass.

### Task 3: Add reproducible Conda definitions and installation stages

**Files:**
- Create: `configs/environments/navdp-autodl.yml`
- Create: `configs/environments/isaaclab-autodl.yml`
- Modify: `scripts/setup_autodl.sh`
- Modify: `tests/scripts/test_setup_autodl.sh`

- [ ] Add Python 3.10 Conda definitions with pip packaging prerequisites.
- [ ] Test then implement environment existence detection and creation from the definitions.
- [ ] Test then implement NavDP requirements installation.
- [ ] Test then implement Isaac Sim `4.2.0.2` installation through NVIDIA's extra index.
- [ ] Test then implement non-destructive IsaacLab clone/tag selection at `v1.2.0` and installation in the IsaacLab environment.
- [ ] Test then implement root benchmark requirements installation.
- [ ] Run the integration test after each behavior and keep all cases green.

### Task 4: Add verification, export, and operator documentation

**Files:**
- Modify: `scripts/setup_autodl.sh`
- Modify: `tests/scripts/test_setup_autodl.sh`
- Modify: `README.md`
- Create at runtime: `requirements/autodl/navdp-freeze.txt`
- Create at runtime: `requirements/autodl/isaaclab-freeze.txt`

- [ ] Test then implement Python package/import checks and a bounded headless Isaac Sim smoke test.
- [ ] Test then implement `--skip-verify`.
- [ ] Test then implement deterministic `pip freeze` snapshot output without overwriting source requirement files.
- [ ] Document the one-line AutoDL setup, options, configuration variables, exclusions, and launch commands in README.
- [ ] Run `bash -n scripts/setup_autodl.sh tests/scripts/test_setup_autodl.sh` and the complete integration test.
- [ ] Run ShellCheck if installed, inspect `git diff --check`, and review the final diff for unrelated changes.
