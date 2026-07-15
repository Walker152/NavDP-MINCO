# AutoDL Environment Bootstrap Design

## Goal

Provide a repeatable, one-command setup for running NavDP and its Isaac Sim benchmark on an AutoDL GPU server. The setup creates two isolated Conda environments, installs the versions documented by this repository, verifies the installation without requiring a desktop session, and exports reproducible dependency snapshots.

Scene-N1 assets and NavDP checkpoints are explicitly out of scope.

## Supported Host Assumptions

- The host is a Linux AutoDL GPU instance.
- Conda is already installed and callable.
- A compatible NVIDIA driver is installed by AutoDL; the script must inspect it but must not install or upgrade it.
- The repository has already been cloned, and the setup command is run from any directory while resolving the repository from the script location.
- Network access to Conda, PyPI, NVIDIA PyPI, and GitHub is available during installation.

## Chosen Approach

Use a hybrid layout:

- A Bash entry point owns preflight checks, orchestration, Git-based IsaacLab installation, verification, logging, and dependency export.
- Two Conda environment definition files record the environment names, Python version, and pip/packaging prerequisites.
- A non-networked test script exercises argument parsing, command construction, version pins, and idempotent control flow using fake executables.

This is preferable to a single monolithic script because the core Conda definitions remain easy to inspect and reproduce. It is preferable to pure environment YAML because NVIDIA's extra package index, the pinned IsaacLab checkout, and simulator verification require procedural steps.

## Environment Layout

### `navdp`

- Python 3.10.
- Dependencies from `baselines/navdp/requirements.txt`.
- Intended to run `baselines/navdp/navdp_server.py` independently from the simulator.

### `isaaclab`

- Python 3.10.
- Isaac Sim packages pinned to `4.2.0.2`:
  - `isaacsim`
  - `isaacsim-extscache-physics`
  - `isaacsim-extscache-kit`
  - `isaacsim-extscache-kit-sdk`
- IsaacLab checked out at tag `v1.2.0` and installed through its documented installer.
- Repository benchmark dependencies from the root `requirements.txt`.

The two environments remain separate because the upstream README defines that isolation and their Python packages may otherwise conflict.

## Script Interface

The primary command is:

```bash
bash scripts/setup_autodl.sh
```

Supported options:

- `--check-only`: run host and repository preflight checks without changing environments or downloading packages.
- `--skip-verify`: install everything but skip runtime smoke checks.
- `--help`: print usage and configuration variables.

Configuration is provided through environment variables:

- `NAVDP_ENV_NAME`, default `navdp`.
- `ISAACLAB_ENV_NAME`, default `isaaclab`.
- `ISAACLAB_DIR`, default `${AUTODL_ROOT:-$HOME}/IsaacLab`.
- `PIP_INDEX_URL`, optional primary Python package mirror. NVIDIA packages always retain `https://pypi.nvidia.com` as an extra index.

The script uses `conda run -n NAME` rather than `conda activate`, so it works in non-interactive shells and AutoDL startup jobs.

## Installation Flow

1. Resolve the repository root from the script path.
2. Validate Linux, Bash, Conda, Git, `nvidia-smi`, required requirement files, and available disk space.
3. Print detected GPU, driver, Conda, and target path information.
4. Create each Conda environment only when it does not already exist.
5. Upgrade pip tooling inside each environment.
6. Install NavDP model dependencies into `navdp`.
7. Install the four pinned Isaac Sim packages from NVIDIA's package index into `isaaclab`.
8. Clone IsaacLab when absent. When present, verify that it is a Git checkout and safely select the pinned `v1.2.0` tag without overwriting unrelated local changes.
9. Run the IsaacLab installer in the `isaaclab` environment, tolerating only the README-documented unavailable `rsl-rl` condition rather than masking arbitrary failures.
10. Install the root benchmark requirements into `isaaclab`.
11. Unless verification is disabled, run import/version checks and an Isaac Sim headless smoke test with a bounded timeout.
12. Export installed-package snapshots for both environments under `requirements/autodl/` and print follow-up commands.

Repeated execution is supported: existing environments and a valid pinned IsaacLab checkout are reused, while pip installation and verification are safe to repeat.

## Safety and Failure Handling

- Use strict Bash error handling and identify the failing installation stage.
- Never install system packages, change NVIDIA drivers, download scene assets, or download model checkpoints.
- Never delete or reset an existing IsaacLab directory. If its state is incompatible with selecting `v1.2.0`, stop with a remediation message.
- Validate environment names and paths before interpolating them into commands.
- Keep generated snapshots separate from the repository's source requirement files.
- Preserve existing repository changes and avoid modifying experimental configuration.
- The Isaac Sim smoke test must be headless and time-bounded so an AutoDL SSH session cannot hang indefinitely.

## Outputs

Successful completion produces:

- Conda environment `navdp`.
- Conda environment `isaaclab`.
- A pinned IsaacLab `v1.2.0` checkout at `ISAACLAB_DIR`.
- `requirements/autodl/navdp-freeze.txt`.
- `requirements/autodl/isaaclab-freeze.txt`.
- Terminal instructions for launching the NavDP server and rerunning verification.

## Testing Strategy

Tests run without network access or real package installation. A temporary fake toolchain records invocations of Conda, Git, `nvidia-smi`, and timeout commands. Tests verify:

- `--help` and invalid arguments.
- `--check-only` performs checks but no mutation.
- Missing Conda or GPU tooling fails with an actionable message.
- Default and overridden environment names and IsaacLab paths.
- Exact Isaac Sim `4.2.0.2` and IsaacLab `v1.2.0` pins.
- Existing environments and checkout reuse the idempotent path.
- Snapshot export and verification can be independently observed or skipped.

Shell syntax checks are also run with `bash -n`; ShellCheck is used when available but is not a mandatory host dependency.

## Acceptance Criteria

- One documented command configures both environments on a compatible AutoDL host.
- A second execution does not recreate environments, reclone IsaacLab, or destroy local state.
- Installation versions match the repository README.
- Runtime verification is suitable for a headless server and cannot block forever.
- No assets, checkpoints, drivers, or unrelated baseline environments are installed.
- Tests validate orchestration without requiring a GPU, Conda environment creation, or network access.
