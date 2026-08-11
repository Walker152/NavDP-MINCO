# NavDP One-Click Research Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair Tasks 01–07 and deliver reproducible local-static, simulation, and one-click experiment workflows that generate traceable publication-quality results without relying on deleted historical artifacts.

**Architecture:** Python owns parameter resolution, workflow stages, validation, analysis, receipts, and reports. Three thin shell entry points locate the environment and invoke the Python workflow API. Every stage writes immutable, hash-addressed evidence; real Isaac execution remains behind explicit authorization while local static experiments and dynamic dry-run execute by default.

**Tech Stack:** Python 3.10, unittest, NumPy, SciPy, Matplotlib, ImageIO/OpenCV, Bash, CMake/C++17, pybind11, IsaacLab/Isaac Sim for explicitly authorized runs.

---

## File structure

New focused modules:

- `experiments/core/parameter_receipt.py`: validated defaults/overrides/effective parameter resolution.
- `experiments/core/artifact_receipt.py`: SHA-256 inventory helpers shared by validators and workflows.
- `experiments/orchestrators/research_workflow.py`: static, simulation, and all-workflow stage orchestration.
- `experiments/analyzers/static_comparison.py`: paired legacy/safe tables and boundary summaries.
- `experiments/analyzers/paper_report.py`: data-driven PNG/PDF/backing-data/caption generation.
- `experiments/visualizers/paper_style.py`: one publication plotting style and figure-save receipt helper.
- `experiments/integration/dynamic_sanity.py`: pure helpers for dynamic hashes, initial-state and ESDF sanity.
- `scripts/run_static_experiments.sh`: local-only workflow entry.
- `scripts/run_simulation_experiments.sh`: dry-run by default, real simulation only with gate.
- `experiments/tests/fixtures.py`: self-contained trace/video/static fixtures independent of `results/`.

Modified modules keep their current responsibilities; no broad rewrite of the evaluator or MINCO pipeline is planned.

### Task 1: Repair moved evaluator paths and make tests result-independent

**Files:**
- Modify: `experiments/simulators/isaac_navdp_backend.py:28-100`
- Modify: `scripts/autodl_self_check_repair.sh`
- Modify: `experiments/tests/test_real_backend_static.py`
- Modify: `experiments/tests/test_paired_real_experiment.py`
- Modify: `experiments/tests/test_eval_hooks_video.py`
- Create: `experiments/tests/fixtures.py`
- Modify: `experiments/tests/test_static_benchmark.py`
- Modify: `experiments/tests/test_closure.py`
- Modify: `experiments/analyzers/readonly.py`
- Modify: `experiments/analyzers/comparison_study.py`
- Modify: `experiments/visualizers/paired_video.py`
- Modify: `experiments/tests/test_readonly_comparison.py`
- Modify: `experiments/tests/test_trace_evidence.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write a failing backend-path test**

```python
def test_backend_uses_versioned_run_script(self):
    backend = IsaacNavDPBackend(repo_root=Path("."))
    run = RunSpec("suite", "experiment", "raw", "cold", "SPARSE", "scene", 0, "run")
    self.assertEqual(
        backend.evaluator_path,
        Path("run_scripts/eval_pointgoal_wheeled.py").resolve(),
    )
    self.assertEqual(backend.validate_static_configuration(run), [])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_real_backend_static.RealBackendStaticTests.test_backend_uses_versioned_run_script -v
```

Expected: FAIL because `evaluator_path` does not exist and validation still checks the repository root.

- [ ] **Step 3: Implement one evaluator path in the backend**

Add in `IsaacNavDPBackend.__init__`:

```python
self.evaluator_path = (
    self.repo_root / "run_scripts" / "eval_pointgoal_wheeled.py"
).resolve()
```

Use `self.evaluator_path` in validation and command construction. Update source-inspection tests and AutoDL checks to the same path.

- [ ] **Step 4: Verify the path tests GREEN**

Run:

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_real_backend_static \
  experiments.tests.test_paired_real_experiment \
  experiments.tests.test_eval_hooks_video -v
```

Expected: all evaluator path and dry-run tests pass.

- [ ] **Step 5: Add self-contained result fixtures**

Create helpers with this public interface:

```python
def write_minimal_trace_case(root: Path, case_uid: str = "fixture_case") -> Path:
    """Write numeric-only NPZ, metadata JSON, and case-index JSON with hashes."""

def write_minimal_closure_inputs(root: Path) -> dict[str, Path]:
    """Write minimal valid legacy, safe, selection, and readiness directories."""
```

Use `tempfile.TemporaryDirectory()` in closure and trace-import tests instead of repository `results/`.

- [ ] **Step 6: Prove deleted historical results are no longer required**

Run:

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_static_benchmark.TraceImportTests \
  experiments.tests.test_closure -v
```

Expected: tests pass when `results/navdp_minco_longterm_20260726` is absent.

- [ ] **Step 7: Make task code/tests/docs trackable**

Add targeted `.gitignore` exceptions after the broad ignore rules:

```gitignore
!experiments/tests/
!experiments/tests/**
!docs/navdp_codex_longterm_tasks/
!docs/navdp_codex_longterm_tasks/**
!docs/superpowers/specs/
!docs/superpowers/specs/**
!docs/superpowers/plans/
!docs/superpowers/plans/**
!run_scripts/
!run_scripts/**
```

- [ ] **Step 8: Add failing read-only provenance and synchronization tests**

```python
def test_readonly_manifest_hashes_every_consumed_input(self):
    suite = write_minimal_real_suite(self.root)
    output = self.root / "comparison"
    generate_comparison_study(suite, output, max_trace_cases=1)
    manifest = json.loads((output / "study_manifest.json").read_text())
    hashed = {row["path"] for row in manifest["consumed_inputs"]}
    self.assertIn("experiments/e/s/raw/seed_0/run_x/episode_metrics.csv", hashed)
    self.assertTrue(any(path.endswith(".npz") for path in hashed))
    self.assertTrue(any(path.endswith(".mp4") for path in hashed))

def test_monotonic_frame_timestamps_are_relative_not_exact_wall_clock(self):
    clock = read_video_clock(video, receipt_with_monotonic_timestamps)
    self.assertEqual(clock.method, "RECORDED_RELATIVE_TIMESTAMPS")
    self.assertFalse(clock.exact_wall_clock)
```

- [ ] **Step 9: Hash consumed source evidence and correct clock semantics**

Record path, size and SHA-256 for every CSV, trace, video and sidecar actually consumed by comparison generation. Treat timestamps as exact wall clock only when the receipt contains an explicit common clock-domain identifier and absolute epoch timestamps; otherwise align by episode-relative time and state the error bound. Select trace cases with deterministic tag-stratified coverage before stable UID tie-break.

- [ ] **Step 10: Verify Task01 tests GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_readonly_comparison \
  experiments.tests.test_trace_evidence -v
```

- [ ] **Step 11: Commit the isolated repair**

```bash
git add .gitignore run_scripts experiments/simulators/isaac_navdp_backend.py \
  experiments/analyzers/readonly.py experiments/analyzers/comparison_study.py \
  experiments/visualizers/paired_video.py experiments/tests \
  scripts/autodl_self_check_repair.sh
git commit -m "fix: restore versioned evaluator and self-contained tests"
```

### Task 2: Make calibration the effective-parameter truth source

**Files:**
- Create: `experiments/core/parameter_receipt.py`
- Modify: `experiments/core/effective_parameters.py`
- Modify: `experiments/orchestrators/suite_runner.py`
- Modify: `experiments/recorders/run_manifest.py`
- Modify: `configs/experiments/full_suite.json`
- Modify: `experiments/tests/test_robot_calibration.py`
- Create: `experiments/tests/test_parameter_receipt.py`

- [ ] **Step 1: Write failing tests for calibrated defaults and invalid overrides**

```python
class ParameterReceiptTests(unittest.TestCase):
    def test_calibration_supplies_default_safety_and_wheel_values(self):
        receipt = resolve_parameter_receipt(video_enabled=True, overrides={})
        calibration = receipt["effective"]["robot_calibration"]
        minco = receipt["effective"]["minco"]
        self.assertEqual(
            minco["validation_safe_distance_m"],
            calibration["validation_safe_dist_m"],
        )
        self.assertEqual(
            minco["optimization_safe_distance_m"],
            calibration["optimization_safe_dist_m"],
        )
        self.assertGreater(
            receipt["effective"]["minco_mpc"]["max_wheel_speed_radps"], 0.0
        )

    def test_unsafe_distance_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "validation_safe_distance_m"):
            resolve_parameter_receipt(
                video_enabled=True,
                overrides={"minco": {"validation_safe_distance_m": 0.01}},
            )
```

- [ ] **Step 2: Run and verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_parameter_receipt -v
```

Expected: import failure because `resolve_parameter_receipt` is not implemented.

- [ ] **Step 3: Implement parameter receipt resolution**

Expose:

```python
def resolve_parameter_receipt(
    *, video_enabled: bool, overrides: Mapping[str, Mapping[str, object]] | None
) -> dict[str, object]:
    defaults = calibrated_defaults()
    normalized_overrides = normalize_overrides(overrides or {})
    effective = merge_sections(defaults, normalized_overrides)
    validate_effective_parameters(effective)
    effective["video"]["enabled"] = bool(video_enabled)
    return {
        "schema_version": 1,
        "defaults": defaults,
        "overrides": normalized_overrides,
        "effective": effective,
        "calibration_sha256": effective["robot_calibration"]["sha256"],
    }
```

Validation must enforce finite positive wheel geometry/limits, calibrated collision radius, and `optimization >= validation >= circumscribed radius` for `safe_corridor_v1`.

- [ ] **Step 4: Preserve the existing compatibility API**

Implement:

```python
def effective_parameters(video_enabled=True, overrides=None):
    return resolve_parameter_receipt(
        video_enabled=video_enabled,
        overrides=overrides,
    )["effective"]
```

Add `parameter_receipt` beside `effective_parameters` in `run_config.json` and `run_manifest.json`.

- [ ] **Step 5: Update the full-suite config**

Set the full suite to calibrated values and remove the null wheel override:

```json
"optimization_safe_distance_m": 0.4293079057,
"validation_safe_distance_m": 0.2793079057,
"constraint_profile": "safe_corridor_v1",
"max_wheel_speed_radps": 100.0
```

Keep the separate static legacy profile unchanged for controlled legacy comparison.

- [ ] **Step 6: Verify parameter and backend integration GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_parameter_receipt \
  experiments.tests.test_robot_calibration \
  experiments.tests.test_real_backend_static -v
```

- [ ] **Step 7: Commit calibrated parameter plumbing**

```bash
git add configs/experiments/full_suite.json experiments/core \
  experiments/orchestrators/suite_runner.py experiments/recorders/run_manifest.py \
  experiments/tests/test_parameter_receipt.py experiments/tests/test_robot_calibration.py
git commit -m "fix: use calibration as experiment parameter truth"
```

### Task 3: Honor suite resume, retry, and analysis semantics

**Files:**
- Modify: `experiments/designers/suite.py`
- Modify: `experiments/orchestrators/suite_runner.py`
- Modify: `experiments/cli/main.py`
- Create: `experiments/tests/test_suite_config_semantics.py`

- [ ] **Step 1: Write failing precedence tests**

```python
def test_suite_flags_apply_when_cli_is_unspecified(self):
    config = SuiteConfig(
        suite_id="suite",
        output_root=Path("/tmp/results"),
        manifest_path=Path("/tmp/manifest.json"),
        runs=({"experiment_id": "e", "variant": "raw", "warm_start_mode": "cold"},),
        analysis={"enabled": False},
        retry={"failed": True},
        resume=True,
    )
    behavior = resolve_suite_behavior(
        config,
        resume=None,
        retry_failed=None,
        analysis_enabled=None,
    )
    self.assertTrue(behavior.resume)
    self.assertTrue(behavior.retry_failed)
    self.assertFalse(behavior.analysis_enabled)

def test_explicit_cli_false_overrides_suite_true(self):
    config = SuiteConfig(
        suite_id="suite",
        output_root=Path("/tmp/results"),
        manifest_path=Path("/tmp/manifest.json"),
        runs=({"experiment_id": "e", "variant": "raw", "warm_start_mode": "cold"},),
        retry={"failed": True},
        resume=True,
    )
    behavior = resolve_suite_behavior(config, resume=False, retry_failed=False)
    self.assertFalse(behavior.resume)
    self.assertFalse(behavior.retry_failed)
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_suite_config_semantics -v
```

- [ ] **Step 3: Implement explicit precedence**

Add an immutable behavior model:

```python
@dataclass(frozen=True)
class SuiteBehavior:
    resume: bool
    retry_failed: bool
    analysis_enabled: bool

def resolve_suite_behavior(config, *, resume=None, retry_failed=None, analysis_enabled=None):
    return SuiteBehavior(
        resume=config.resume if resume is None else bool(resume),
        retry_failed=bool((config.retry or {}).get("failed", False))
        if retry_failed is None else bool(retry_failed),
        analysis_enabled=bool((config.analysis or {}).get("enabled", True))
        if analysis_enabled is None else bool(analysis_enabled),
    )
```

Use this model throughout `run_suite`. Store it in `suite_config.json`.

- [ ] **Step 4: Give CLI flags tri-state behavior**

Use paired switches with default `None`:

```python
run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
run.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=None)
run.add_argument("--analysis", action=argparse.BooleanOptionalAction, default=None)
```

- [ ] **Step 5: Verify resume lifecycle and analysis behavior**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_suite_config_semantics \
  experiments.tests.test_full_pipeline \
  experiments.tests.test_end_to_end -v
```

- [ ] **Step 6: Commit config semantics**

```bash
git add experiments/designers/suite.py experiments/orchestrators/suite_runner.py \
  experiments/cli/main.py experiments/tests/test_suite_config_semantics.py
git commit -m "fix: honor suite workflow flags"
```

### Task 4: Strengthen immutable static evidence validation

**Files:**
- Create: `experiments/core/artifact_receipt.py`
- Modify: `experiments/static/benchmark.py`
- Modify: `experiments/static/case_schema.py`
- Modify: `experiments/tests/test_static_benchmark.py`
- Create: `experiments/tests/test_static_validation.py`

- [ ] **Step 1: Write tamper-detection tests**

```python
def test_static_validator_rejects_modified_case_npz(self):
    root = self.run_one_case_benchmark()
    case_npz = next((root / "cases").glob("*.npz"))
    case_npz.write_bytes(case_npz.read_bytes() + b"tamper")
    errors = validate_static_benchmark(root)
    self.assertTrue(any("hash mismatch" in error for error in errors))

def test_static_validator_rejects_false_determinism(self):
    root = self.run_one_case_benchmark()
    manifest = json.loads((root / "legacy_baseline_manifest.json").read_text())
    manifest["all_deterministic"] = False
    (root / "legacy_baseline_manifest.json").write_text(json.dumps(manifest))
    self.assertIn("benchmark is not deterministic", validate_static_benchmark(root))
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_static_validation -v
```

- [ ] **Step 3: Implement file receipts**

Add helpers:

```python
def file_receipt(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

def validate_file_receipt(root: Path, receipt: Mapping[str, object]) -> list[str]:
    path = root / str(receipt["path"])
    if not path.is_file():
        return [f"missing artifact {receipt['path']}"]
    if path.stat().st_size != int(receipt["size_bytes"]):
        return [f"size mismatch {receipt['path']}"]
    if sha256_file(path) != receipt["sha256"]:
        return [f"hash mismatch {receipt['path']}"]
    return []
```

- [ ] **Step 4: Record and validate every static input/output**

Manifest case entries must include receipts for metadata JSON, numeric NPZ, metrics JSON, PNG/PDF/GIF artifacts, and ESDF array hash. Validate CSV exact headers, unique case UID, manifest/CSV case-set equality, repeat count and determinism.

- [ ] **Step 5: Verify tamper and regression tests GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_static_validation \
  experiments.tests.test_static_benchmark -v
```

- [ ] **Step 6: Commit stronger validation**

```bash
git add experiments/core/artifact_receipt.py experiments/static \
  experiments/tests/test_static_validation.py experiments/tests/test_static_benchmark.py
git commit -m "feat: validate immutable static evidence"
```

### Task 5: Complete MINCO corridor, adaptive validation, and diagnostics

**Files:**
- Modify: `minco_processor/include/minco_processor/guide_corridor.hpp`
- Modify: `minco_processor/src/guide_corridor.cpp`
- Modify: `minco_processor/include/minco_processor/minco_pipeline.hpp`
- Modify: `minco_processor/src/minco_processor/minco_pipeline.cpp`
- Modify: `minco_processor/include/traj_opt/minco_optimizer.hpp`
- Modify: `minco_processor/src/traj_opt/minco_optimizer.cpp`
- Modify: `minco_processor/bindings/minco_pybind.cpp`
- Modify: `minco_processor/tests/test_pure_algorithm_compile.cpp`
- Modify: `utils_tasks/navdp_minco_adapter.py`
- Modify: `experiments/core/schemas.py`
- Modify: `experiments/integration/eval_hooks.py`
- Modify: `experiments/tests/test_navdp_minco_adapter.py`

- [ ] **Step 1: Add native tests that fail on current behavior**

Add checks that:

```cpp
if (corridor.contains(point_on_segment_three, 0, &matched)) return 21;
if (depth_limited_report.reason != "VALIDATION_DEPTH_EXHAUSTED") return 22;
if (result.penalty_terms.count("guide_corridor") == 0U) return 23;
if (!finiteDifferenceTimeGradientMatches(optimizer, 1e-4)) return 24;
```

The progression test uses four separated segments and confirms a sample cannot jump from segment 0 directly to segment 3.

- [ ] **Step 2: Build and verify native RED**

```bash
cmake -S minco_processor -B minco_processor/build
cmake --build minco_processor/build -j2
minco_processor/build/minco_processor_compile_test
```

Expected: nonzero exit on the first newly asserted missing behavior.

- [ ] **Step 3: Restrict corridor progression and define junction margin**

Search only `previous-1` through `previous+1` and compute junction overlap as the minimum radius covering the shared junction. Reject nonfinite or noncoincident adjacent endpoints using the map-resolution tolerance.

```cpp
const int end = std::min(
  static_cast<int>(segments_.size()), previous_segment + 2);
for (int index = begin; index < end; ++index) {
  const auto & segment = segments_[static_cast<size_t>(index)];
  const double distance = std::sqrt(std::max(
    0.0,
    squaredDistanceAndGradient(point, segment.start, segment.end, nullptr)));
  const double margin = segment.radius - distance;
  if (margin >= -1e-9 && margin > best_margin) {
    best = index;
    best_margin = margin;
  }
}
```

- [ ] **Step 4: Fail closed on adaptive depth exhaustion**

Separate `needs_refine` from `can_refine`. When `needs_refine` is true and `depth >= adaptive_max_depth`, set `depth_exhausted` and reject with `VALIDATION_DEPTH_EXHAUSTED` plus measured interval displacement and configured spatial limit.

- [ ] **Step 5: Expose objective terms**

Copy optimizer penalty log into a named `std::map<std::string, double>` on `Result`:

```cpp
result.penalty_terms = {
  {"energy", log(0)}, {"position", log(1)}, {"velocity", log(2)},
  {"acceleration", log(3)}, {"attractor", log(4)},
  {"guide_corridor", log(5)}, {"time_barrier", log(6)},
};
```

Expose it through pybind and adapter records. Add `objective_terms_json` to planning/candidate/plan schemas.

- [ ] **Step 6: Add full finite-difference tests**

Test the integrated optimizer cost gradients for ESDF position, guide corridor position and segment time variables, including samples on integration boundaries. Retain explicit straight, acute-turn, short-segment, repeated-point, OOB and negative-ESDF cases.

- [ ] **Step 7: Verify native and Python diagnostics GREEN**

```bash
cmake --build minco_processor/build -j2
minco_processor/build/minco_processor_compile_test
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_navdp_minco_adapter \
  experiments.tests.test_eval_diagnostics_completeness -v
```

- [ ] **Step 8: Commit safety completion**

```bash
git add minco_processor utils_tasks/navdp_minco_adapter.py \
  experiments/core/schemas.py experiments/integration/eval_hooks.py \
  experiments/tests/test_navdp_minco_adapter.py
git commit -m "fix: complete MINCO hard-validation diagnostics"
```

### Task 6: Generate complete static boundary evidence and rankings

**Files:**
- Modify: `experiments/configs/static_boundary_selection_v1.json`
- Modify: `experiments/static/selection.py`
- Create: `experiments/analyzers/static_comparison.py`
- Create: `experiments/visualizers/paper_style.py`
- Create: `experiments/tests/test_static_paper_outputs.py`
- Modify: `experiments/tests/test_static_selection.py`

- [ ] **Step 1: Write failing output-contract tests**

```python
def test_selection_contains_full_rankings_and_pending_hot_start(self):
    frozen = run_boundary_selection(self.config, self.output)
    self.assertEqual(len(frozen["best_ranking"]), len(frozen["eligible_case_uids"]))
    self.assertEqual(len(frozen["worst_ranking"]), len(frozen["eligible_case_uids"]))
    self.assertEqual(frozen["hot_start_evidence"], "PENDING_DYNAMIC_VALIDATION")

def test_every_paper_figure_has_pdf_data_caption_and_receipt(self):
    generate_static_paper_outputs(self.selection_dir, self.paper_dir)
    for png in self.paper_dir.glob("figures/*.png"):
        stem = png.stem
        self.assertTrue((self.paper_dir / "figures" / f"{stem}.pdf").is_file())
        self.assertTrue((self.paper_dir / "tables" / f"{stem}.csv").is_file())
        self.assertTrue((self.paper_dir / "captions" / f"{stem}.md").is_file())
        self.assertTrue((self.paper_dir / "receipts" / f"{stem}.json").is_file())
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_static_paper_outputs -v
```

- [ ] **Step 3: Add explicit factor metadata to scan config**

Each state variant receives `factor_name` and numeric/categorical `factor_level`; for example:

```json
{
  "case_uid": "state_v_mid",
  "source_case_uid": "syn_straight",
  "velocity_xyz_mps": [0.5, 0.0, 0.0],
  "factor_name": "initial_speed_mps",
  "factor_level": 0.5,
  "tags": ["velocity_mid"]
}
```

Add a configured two-dimensional grid over `initial_speed_mps × yaw_error_rad` so the heatmap represents measured factor combinations rather than category counts.

- [ ] **Step 4: Persist complete rankings**

Store full ordered records:

```python
frozen["best_ranking"] = [ranking_record(row, rank) for rank, row in enumerate(best_ranked, 1)]
frozen["worst_ranking"] = [ranking_record(row, rank) for rank, row in enumerate(worst_ranked, 1)]
frozen["hot_start_evidence"] = "PENDING_DYNAMIC_VALIDATION"
```

- [ ] **Step 5: Implement publication figure saving**

`save_paper_figure` must write the bitmap, vector, backing table, caption and input receipt in one operation and close the Matplotlib figure in `finally`.

- [ ] **Step 6: Generate all static plots**

Produce data-driven single-factor curves, real two-factor heatmaps, transition matrix, failure stack, Pareto view, complete ranking table and Best2/Worst2 cards. Captions state profile, source, units, `n`, missing/failed denominator and interpretation.

- [ ] **Step 7: Verify static paper outputs GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_static_selection \
  experiments.tests.test_static_paper_outputs -v
```

- [ ] **Step 8: Commit boundary and paper output work**

```bash
git add experiments/configs/static_boundary_selection_v1.json \
  experiments/static/selection.py experiments/analyzers/static_comparison.py \
  experiments/visualizers/paper_style.py experiments/tests
git commit -m "feat: generate complete static research evidence"
```

### Task 7: Complete dynamic materialization and frame sanity

**Files:**
- Modify: `experiments/dynamic_pilot.py`
- Create: `experiments/integration/dynamic_sanity.py`
- Modify: `run_scripts/eval_pointgoal_wheeled.py`
- Modify: `experiments/simulators/isaac_navdp_backend.py`
- Modify: `experiments/tests/test_dynamic_pilot.py`
- Create: `experiments/tests/test_dynamic_sanity.py`

- [ ] **Step 1: Write failing scene/hash/sanity tests**

```python
def test_materialized_usd_declares_collision(self):
    text = _usda_scene([[1.0, 1.0, 2.0, 2.0]])
    self.assertGreaterEqual(text.count("PhysicsCollisionAPI"), 2)
    self.assertIn("physics:collisionEnabled = true", text)

def test_sanity_rejects_case_hash_mismatch(self):
    with self.assertRaisesRegex(RuntimeError, "case hash"):
        validate_dynamic_receipts(spec, observed, case_sha256="wrong")

def test_unmaterializable_main_uses_frozen_backup_with_receipt(self):
    selected = self.selection_with_unmaterializable_best1()
    chosen, substitutions = resolve_materializable_selection(selected)
    self.assertEqual(chosen[0], selected["best_backups"][0])
    self.assertEqual(substitutions[0]["reason"], "INITIAL_ACCELERATION_NOT_INJECTABLE")
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_dynamic_pilot \
  experiments.tests.test_dynamic_sanity -v
```

- [ ] **Step 3: Author collidable USD scenes**

Apply collision schemas to floor and obstacles in USDA:

```usda
prepend apiSchemas = ["PhysicsCollisionAPI"]
bool physics:collisionEnabled = true
```

Validate the generated stage with OpenUSD when bindings are present and with strict text/schema checks in lightweight tests.

- [ ] **Step 4: Implement pre-run receipt validation**

The pure helper validates calibration, case and scene hashes plus requested/observed state and ESDF clearance. The evaluator gathers observed values after environment reset and before starting the planner thread; any mismatch raises before an episode is recorded.

- [ ] **Step 5: Implement frozen substitution behavior**

Only predeclared backups may substitute an unmaterializable main case. Write `substitution_receipts.json` with slot, rejected UID/hash/reason and selected UID/hash. Once any real process starts, substitutions are disabled.

- [ ] **Step 6: Add readiness validation for eight directly executable commands**

Verify each command references existing evaluator, calibration, manifest, scene USD, pointgoal array and dynamic case spec. Verify command/profile/case pairing and `started_processes == 0`.

- [ ] **Step 7: Verify dynamic preparation GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_dynamic_pilot \
  experiments.tests.test_dynamic_sanity \
  experiments.tests.test_real_backend_static -v
```

- [ ] **Step 8: Commit dynamic readiness repair**

```bash
git add experiments/dynamic_pilot.py experiments/integration/dynamic_sanity.py \
  experiments/simulators/isaac_navdp_backend.py run_scripts/eval_pointgoal_wheeled.py \
  experiments/tests/test_dynamic_pilot.py experiments/tests/test_dynamic_sanity.py
git commit -m "fix: validate dynamic case materialization"
```

### Task 8: Capture machine termination, contact, wheel, and recovery truth

**Files:**
- Modify: `configs/tasks/wheeled_task.py`
- Modify: `utils_tasks/episode_diagnostics.py`
- Modify: `run_scripts/eval_pointgoal_wheeled.py`
- Modify: `experiments/integration/eval_hooks.py`
- Modify: `experiments/core/failure_taxonomy.py`
- Modify: `experiments/analyzers/validator.py`
- Modify: `experiments/tests/test_episode_diagnostics.py`
- Modify: `experiments/tests/test_eval_collection_contract.py`
- Modify: `experiments/tests/test_eval_hooks_video.py`
- Modify: `experiments/tests/test_strict_validation.py`
- Modify: `experiments/tests/test_failure_taxonomy.py`

- [ ] **Step 1: Write failing termination/contact tests**

```python
def test_pointnav_collision_term_is_configured(self):
    source = Path("configs/tasks/wheeled_task.py").read_text()
    block = source.split("class PointNavTerminationsCfg", 1)[1].split("class ImageNavTerminationsCfg", 1)[0]
    self.assertIn("base_contact = DoneTerm", block)

def test_episode_bridge_accumulates_hold_stop_and_saturation(self):
    bridge.record_control_step(0, "p", 0.0, 0.0, control_state="HOLD_LAST", wheel_saturated=True, sample_dt_s=0.05)
    bridge.record_control_step(1, "p", 0.0, 0.0, control_state="STOP", wheel_saturated=False, sample_dt_s=0.05)
    bridge.end_episode(success=False)
    row = sink.rows["episode_metrics"][0]
    self.assertEqual(row["hold_duration_s"], 0.05)
    self.assertEqual(row["stop_duration_s"], 0.05)
    self.assertEqual(row["wheel_saturation_count"], 1)

def test_unknown_failure_is_not_declared_safe(self):
    classified = classify_reason("NEW_UNMAPPED_FAILURE")
    self.assertEqual(classified["reason_source"], "UNMAPPED")
    self.assertFalse(classified["safe_failure"])
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_episode_diagnostics \
  experiments.tests.test_eval_collection_contract -v
```

- [ ] **Step 3: Add PointNav collision termination**

Add the same contact-based term used by exploration:

```python
base_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=DINGO_BASE_LINK),
        "threshold": DINGO_THRESHOLD,
    },
)
```

- [ ] **Step 4: Extract structured termination observation**

Return a dataclass containing canonical reason, raw active terms, contact flag, force and object when available. Do not infer force or object from text.

```python
@dataclass(frozen=True)
class TerminationObservation:
    reason: str
    raw_terms: str
    contact_detected: bool | None
    collision_object: str
    impact_force_n: float | None
```

- [ ] **Step 5: Record actual joint velocities and contact data**

Resolve left/right wheel joint indices once from `robot_articulation.find_joints`. At each control sample read `robot_articulation.data.joint_vel`, compute saturation against the calibrated limit, and pass actual values to the hook. Read the existing `contact_sensor` force tensor at termination and pass only finite available values.

- [ ] **Step 6: Accumulate recovery durations inside the bridge**

Maintain hold/stop seconds and wheel saturation count from control samples. Attach the most recent planning cycle UID to termination.

- [ ] **Step 7: Strengthen real-run validator**

For REAL Task06/full-suite runs require raw termination terms, termination frame/plan/cycle association for MINCO variants, contact consistency, wheel fields when calibration declares a limit, and nonblank recovery durations. Report unavailable optional sensor fields through an explicit availability receipt.

Unknown failure reasons remain `UNMAPPED`, are treated as unsafe/unknown rather than safe infrastructure failures, and cause REAL validation to fail until added to the versioned taxonomy.

- [ ] **Step 8: Verify collection and strict validation GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_episode_diagnostics \
  experiments.tests.test_eval_collection_contract \
  experiments.tests.test_eval_hooks_video \
  experiments.tests.test_strict_validation -v
```

- [ ] **Step 9: Commit machine-truth collection**

```bash
git add configs/tasks/wheeled_task.py utils_tasks/episode_diagnostics.py \
  run_scripts/eval_pointgoal_wheeled.py experiments/integration/eval_hooks.py \
  experiments/analyzers/validator.py experiments/tests
git commit -m "feat: record simulation and control machine truth"
```

### Task 9: Build data-driven publication analysis

**Files:**
- Create: `experiments/analyzers/paper_report.py`
- Modify: `experiments/visualizers/paper_style.py`
- Modify: `experiments/analyzers/statistics.py`
- Modify: `experiments/analyzers/data_quality.py`
- Modify: `experiments/analyzers/artifact_manifest.py`
- Create: `experiments/tests/test_paper_report.py`
- Modify: `scripts/gen_charts.py`
- Modify: `scripts/generate_navdp_charts.py`

- [ ] **Step 1: Write failing no-hardcoded-data and figure-bundle tests**

```python
def test_paper_report_uses_input_tables(self):
    write_episode_rows(self.root, raw_success=1, cold_success=0, hot_success=1)
    report = generate_paper_report(self.root, self.output)
    summary = json.loads((self.output / "tables/episode_summary.json").read_text())
    self.assertEqual(summary["minco-cold"]["success_count"], 0)
    self.assertEqual(report["data_source"], "SIMULATED")

def test_legacy_chart_scripts_contain_no_numeric_result_tables(self):
    for path in (Path("scripts/gen_charts.py"), Path("scripts/generate_navdp_charts.py")):
        source = path.read_text()
        self.assertNotIn("nav = {", source)
        self.assertNotIn("vals = [60.0, 45.0, 65.0]", source)
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_paper_report -v
```

- [ ] **Step 3: Implement statistical primitives**

Add Wilson intervals for binomial outcomes and deterministic paired bootstrap by episode/case key:

```python
def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)
```

Bootstrap seeds are fixed in the analysis receipt. Planning cycles are clustered within episode and never treated as independent trial rows.

- [ ] **Step 4: Generate complete figure bundles**

Generate only figures supported by available data. Static-only runs produce static evidence and explicitly mark dynamic panels unavailable. REAL validated runs add episode, control, safety, latency, recovery and static-vs-dynamic figures.

- [ ] **Step 5: Replace hardcoded chart scripts with compatibility wrappers**

Both scripts accept `--input` and `--output`, call `generate_paper_report`, and contain no embedded experimental values.

- [ ] **Step 6: Validate figure provenance**

Artifact manifest validation checks PNG/PDF/backing/caption/receipt groups and hashes every input/output. Captions state units, data source, paired key, `n`, denominator and limitations.

- [ ] **Step 7: Verify report tests GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_paper_report \
  experiments.tests.test_full_pipeline -v
```

- [ ] **Step 8: Commit publication analysis**

```bash
git add experiments/analyzers experiments/visualizers/paper_style.py \
  experiments/tests/test_paper_report.py scripts/gen_charts.py \
  scripts/generate_navdp_charts.py
git commit -m "feat: generate data-driven publication reports"
```

### Task 10: Implement reusable research workflow orchestration

**Files:**
- Create: `experiments/orchestrators/research_workflow.py`
- Modify: `experiments/closure.py`
- Modify: `experiments/cli/main.py`
- Modify: `experiments/__main__.py`
- Create: `experiments/tests/test_research_workflow.py`
- Modify: `experiments/tests/test_closure.py`

- [ ] **Step 1: Write failing stage/resume tests**

```python
def test_static_workflow_runs_from_empty_output(self):
    receipt = run_static_workflow(
        repo_root=self.repo,
        options=WorkflowOptions(output_root=self.output),
    )
    self.assertEqual(receipt["status"], "COMPLETE")
    self.assertEqual(receipt["stages"]["legacy_benchmark"]["status"], "COMPLETE")
    self.assertEqual(receipt["stages"]["safe_benchmark"]["status"], "COMPLETE")
    self.assertTrue((self.output / "paper/report.md").is_file())

def test_resume_rejects_changed_config_hash(self):
    run_static_workflow(
        repo_root=self.repo,
        options=WorkflowOptions(output_root=self.output),
    )
    self.config.write_text(self.config.read_text() + "\n")
    with self.assertRaisesRegex(RuntimeError, "input hash changed"):
        run_static_workflow(
            repo_root=self.repo,
            options=WorkflowOptions(output_root=self.output, resume=True),
        )
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_research_workflow -v
```

- [ ] **Step 3: Implement stage receipts**

Use this stable model:

```python
@dataclass(frozen=True)
class WorkflowOptions:
    output_root: Path
    resume: bool = False
    retry_failed: bool = False
    allow_real_simulation: bool = False
    full_suite: bool = False
    skip_video: bool = False

@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    started_at_utc: str
    ended_at_utc: str
    inputs: tuple[dict[str, object], ...]
    outputs: tuple[dict[str, object], ...]
    command: tuple[str, ...]
```

Write status atomically before and after each stage. Compatibility requires matching stage name, command and input hashes.

- [ ] **Step 4: Implement static workflow stages**

Call build/native test, USD extraction/report, legacy benchmark with `trace_limit=0`, safe benchmark, static paired comparison, boundary selection, paper report, validation and artifact indexing.

- [ ] **Step 5: Implement simulation workflow stages**

Always run dynamic preparation and dry-run. With permission run the dynamic suite; only with both permission and `full_suite=True` run the full suite. Validate and analyze each completed REAL suite separately.

- [ ] **Step 6: Make closure a compatibility facade**

Replace hardcoded dated paths with explicit workflow inputs. `run_codex_closure` delegates to workflow stages and never assumes existing evidence.

- [ ] **Step 7: Add CLI commands**

Add `run-static-workflow`, `run-simulation-workflow`, and `run-all-workflows` with shared arguments:

```text
--output
--resume
--retry-failed
--allow-real-simulation
--full-suite
--skip-video
```

- [ ] **Step 8: Verify orchestration GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_research_workflow \
  experiments.tests.test_closure \
  experiments.tests.test_full_pipeline -v
```

- [ ] **Step 9: Commit orchestration**

```bash
git add experiments/orchestrators/research_workflow.py experiments/closure.py \
  experiments/cli/main.py experiments/__main__.py experiments/tests
git commit -m "feat: add reproducible research workflows"
```

### Task 11: Add static, simulation, and all-experiment shell entries

**Files:**
- Create: `scripts/run_static_experiments.sh`
- Create: `scripts/run_simulation_experiments.sh`
- Modify: `scripts/run_all_experiments.sh`
- Create: `experiments/tests/test_experiment_scripts.py`

- [ ] **Step 1: Write failing shell contract tests**

```python
def test_shell_entries_are_safe_by_default(self):
    for name in (
        "run_static_experiments.sh",
        "run_simulation_experiments.sh",
        "run_all_experiments.sh",
    ):
        path = Path("scripts") / name
        self.assertTrue(path.is_file())
        completed = subprocess.run(["bash", str(path), "--help"], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
    source = Path("scripts/run_all_experiments.sh").read_text()
    self.assertNotIn("--allow-real-simulation\"", source)
```

- [ ] **Step 2: Verify RED**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_experiment_scripts -v
```

- [ ] **Step 3: Implement a common environment preamble in each script**

Use the script location, never the caller working directory:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${NAVDP_PYTHON:-}" ]]; then
  PYTHON_BIN="$NAVDP_PYTHON"
elif [[ -x "/home/alioth/miniforge3/envs/navdp/bin/python" ]]; then
  PYTHON_BIN="/home/alioth/miniforge3/envs/navdp/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi
export PYTHONPATH="$REPO_ROOT/minco_processor/build:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
```

- [ ] **Step 4: Implement the three commands**

Static calls `python -m experiments run-static-workflow`; simulation calls `run-simulation-workflow`; all calls `run-all-workflows`. Arguments are arrays, not `eval` strings. Default output is generated by Python. Real flags are forwarded only when present in user arguments.

- [ ] **Step 5: Verify shell syntax and safety GREEN**

```bash
bash -n scripts/run_static_experiments.sh
bash -n scripts/run_simulation_experiments.sh
bash -n scripts/run_all_experiments.sh
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_experiment_scripts -v
```

- [ ] **Step 6: Commit shell entries**

```bash
chmod +x scripts/run_static_experiments.sh scripts/run_simulation_experiments.sh \
  scripts/run_all_experiments.sh
git add scripts experiments/tests/test_experiment_scripts.py
git commit -m "feat: add one-click experiment scripts"
```

### Task 12: Document commands and scientific evidence boundaries

**Files:**
- Modify: `experiments/README.md`
- Modify: `experiments/README_PARAMETERS.md`
- Modify: `reports/codex_longterm/RUNBOOK.md`
- Modify: `reports/codex_longterm/final_completion.md`
- Create: `docs/navdp_codex_longterm_tasks/IMPLEMENTATION_STATUS.md`

- [ ] **Step 1: Add executable documentation checks**

Extend `test_experiment_scripts.py`:

```python
def test_documented_commands_match_cli(self):
    parser = build_parser()
    for command in (
        "run-static-workflow",
        "run-simulation-workflow",
        "run-all-workflows",
    ):
        args = parser.parse_args([command, "--output", "/tmp/navdp-doc-check"])
        self.assertEqual(args.command, command)
```

- [ ] **Step 2: Update documentation with exact commands**

Document:

```bash
scripts/run_static_experiments.sh \
  --output results/navdp_minco_paper_local_verification_20260812
scripts/run_simulation_experiments.sh \
  --output results/navdp_minco_paper_local_verification_20260812 --resume
scripts/run_all_experiments.sh
scripts/run_all_experiments.sh --allow-real-simulation
scripts/run_all_experiments.sh --allow-real-simulation --full-suite
```

Explain directory layout, resume/retry semantics, estimated scope, data-source labels, and why static/mock/dry-run evidence cannot support dynamic performance claims.

- [ ] **Step 3: Correct completion status**

Replace unconditional “all tasks completed” language with machine-derived workflow status. `IMPLEMENTATION_STATUS.md` maps each Task 01–07 requirement to code, test and generated artifact.

- [ ] **Step 4: Verify docs and CLI GREEN**

```bash
conda run --no-capture-output -n navdp python -m unittest \
  experiments.tests.test_experiment_scripts -v
```

- [ ] **Step 5: Commit documentation**

```bash
git add experiments/README.md experiments/README_PARAMETERS.md \
  reports/codex_longterm docs/navdp_codex_longterm_tasks/IMPLEMENTATION_STATUS.md
git commit -m "docs: publish reproducible experiment runbook"
```

### Task 13: Run the complete local workflow and final verification

**Files:**
- Generate: `results/navdp_minco_paper_<UTC>/`
- Modify only if failures reveal a tested defect: files named by the failing test

- [ ] **Step 1: Run formatting/static checks**

```bash
git diff --check
conda run --no-capture-output -n navdp python -m compileall -q \
  experiments utils_tasks run_scripts
```

Expected: both commands exit 0.

- [ ] **Step 2: Build and run native tests**

```bash
cmake -S minco_processor -B minco_processor/build
cmake --build minco_processor/build -j2
minco_processor/build/minco_processor_compile_test
```

Expected: build and executable exit 0.

- [ ] **Step 3: Run the complete Python test suite**

```bash
conda run --no-capture-output -n navdp python -m unittest discover \
  -s experiments/tests -p 'test_*.py' -v
```

Expected: zero failures and zero errors; Isaac-only runtime tests may be explicitly skipped with a reason.

- [ ] **Step 4: Execute the real local static workflow**

```bash
scripts/run_static_experiments.sh
```

Expected: calibration, both static profiles, paired comparison, boundary scan, rankings, paper figure bundles, validation and artifact manifest are COMPLETE.

- [ ] **Step 5: Execute mock plus dynamic dry-run**

```bash
scripts/run_simulation_experiments.sh
```

Expected: mock suite COMPLETE; dynamic readiness has 8 runs and `started_processes=0`; no Isaac/NavDP process is started.

- [ ] **Step 6: Execute the default all-workflow resume path**

```bash
scripts/run_all_experiments.sh --resume \
  --output results/navdp_minco_paper_local_verification_20260812
```

Expected: validated stages are skipped, receipts remain hash-compatible, and the final report/manifest validate without regeneration drift.

- [ ] **Step 7: Inspect generated figures and tables**

Open representative footprint, trajectory/ESDF, clearance, dynamics, factor curve, 2D heatmap, transition matrix, Pareto and Best/Worst cards. Confirm labels, units, legends, visible sample counts and caption statements match backing tables.

- [ ] **Step 8: Record final verification receipt**

The workflow writes test counts, native build status, static case counts, selection, dry-run count, artifact count and every validation error to `experiment_receipt.json`. Do not mark overall COMPLETE unless `validation_errors` is empty.

- [ ] **Step 9: Commit final verified state without generated results**

```bash
git add -u
git add configs experiments minco_processor run_scripts scripts utils_tasks docs reports
git status --short
git commit -m "feat: complete NavDP one-click research experiments"
```

Generated `results/` remain untracked and are referenced by their receipt and artifact hashes.
