# NavDP Rolling Paper Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic full-route rolling MINCO experiments for static and local dynamic obstacles, then generate a validated paper-ready showcase containing three-way plots, synchronized GIFs, yaw arrows, data, captions, and receipts.

**Architecture:** A focused `experiments/rolling` package owns immutable scenario/state records, deterministic obstacle sampling, local-guide selection, repeated MINCO calls, execution-prefix state propagation, and fail-closed termination. A separate showcase renderer consumes only recorded rollout evidence and never reconstructs or invents planning data. The research workflow runs rollout scenarios before paper reporting and validates a self-contained `paper_showcase` artifact tree.

**Tech Stack:** Python 3.10, NumPy, Matplotlib, imageio, existing `minco_processor` pybind extension, CSV/JSON/SHA256 receipts, `unittest`, Bash, optional IsaacLab validation.

---

## File map

- Create `experiments/rolling/models.py`: immutable state, obstacle, cycle, result, and configuration contracts.
- Create `experiments/rolling/scenarios.py`: static/dynamic scenario parsing, deterministic obstacle sampling, occupancy/ESDF materialization.
- Create `experiments/rolling/engine.py`: local-guide selection, repeated native planning, execution-prefix propagation, collision/goal/stall/limit termination.
- Create `experiments/rolling/serialization.py`: canonical cycle/result CSV, NPZ, JSON, and receipt I/O.
- Create `experiments/configs/rolling_showcase_v1.json`: scene matrix, method pairing, rollout limits, initial-state sweeps, visual settings.
- Create `experiments/visualizers/rolling_showcase.py`: three-panel/overlay/corridor/state figures, synchronized GIFs, captions, index, validator.
- Create `experiments/rolling/showcase.py`: scenario execution, fair method pairing, aggregate metrics, extreme-case selection, output assembly.
- Modify `experiments/orchestrators/research_workflow.py`: add rolling-showcase stage before unified paper report.
- Modify `scripts/run_static_experiments.sh`: expose showcase config and execute the new stage by default.
- Modify `scripts/run_all_experiments.sh`: include and validate showcase output in the one-click receipt.
- Modify `experiments/README.md`: concise output index and run commands.
- Create `experiments/tests/test_rolling_models.py`, `test_rolling_scenarios.py`, `test_rolling_engine.py`, `test_rolling_showcase.py`.
- Modify `experiments/tests/test_research_workflow.py`, `test_shell_entrypoints.py`.

### Task 1: Immutable rollout evidence contracts

**Files:**
- Create: `experiments/rolling/__init__.py`
- Create: `experiments/rolling/models.py`
- Test: `experiments/tests/test_rolling_models.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_robot_state_rejects_nonfinite_values():
    with self.assertRaisesRegex(ValueError, "finite"):
        RobotState(position_xyz=[0, 0, 0], velocity_xyz_mps=[float("nan"), 0, 0],
                   acceleration_xyz_mps2=[0, 0, 0], yaw_rad=0, yaw_rate_radps=0)

def test_cycle_record_requires_state_continuity():
    cycle = make_cycle(input_x=0.0, executed_end_x=0.25)
    following = make_cycle(index=1, input_x=0.2)
    self.assertIn("state discontinuity", validate_cycle_sequence([cycle, following])[0])

def test_success_requires_final_goal_tolerance():
    result = make_rollout_result(status="GOAL_REACHED", final_x=1.0, goal_x=3.0)
    self.assertIn("goal tolerance", result.validate()[0])
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_models -v`

Expected: import failure for `experiments.rolling.models`.

- [ ] **Step 3: Implement explicit contracts and validators**

```python
@dataclass(frozen=True)
class RobotState:
    position_xyz: np.ndarray
    velocity_xyz_mps: np.ndarray
    acceleration_xyz_mps2: np.ndarray
    yaw_rad: float
    yaw_rate_radps: float

@dataclass(frozen=True)
class RolloutCycle:
    cycle_index: int
    time_s: float
    input_state: RobotState
    local_guide_xyz: np.ndarray
    local_goal_xyz: np.ndarray
    candidate_samples: np.ndarray
    executed_samples: np.ndarray
    corridor_segments: np.ndarray
    obstacle_states: tuple[ObstacleState, ...]
    diagnostics: Mapping[str, object]

@dataclass(frozen=True)
class RolloutResult:
    scenario_uid: str
    method: str
    status: str
    cycles: tuple[RolloutCycle, ...]
    executed_samples: np.ndarray
    final_goal_xyz: np.ndarray
    metrics: Mapping[str, object]
```

Arrays must be finite, read-only and shape checked. `validate_cycle_sequence` must require consecutive indices, monotonic time, exact previous executed end/current input continuity within `1e-8`, and executed samples being a prefix of the corresponding candidate samples. `GOAL_REACHED` must require final error no larger than configured tolerance.

- [ ] **Step 4: Re-run tests and confirm GREEN**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_models -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the isolated contract change**

```bash
git add experiments/rolling/__init__.py experiments/rolling/models.py experiments/tests/test_rolling_models.py
git commit -m "feat: define rolling experiment evidence contracts"
```

### Task 2: Deterministic static and dynamic scenarios

**Files:**
- Create: `experiments/rolling/scenarios.py`
- Create: `experiments/configs/rolling_showcase_v1.json`
- Test: `experiments/tests/test_rolling_scenarios.py`

- [ ] **Step 1: Write failing scenario tests**

```python
def test_dynamic_obstacles_are_deterministic_at_same_time():
    scenario = load_scenarios(CONFIG)["dynamic_crossing"]
    self.assertEqual(scenario.obstacles_at(1.25), scenario.obstacles_at(1.25))

def test_dense_scene_has_more_occupied_area_than_sparse_scene():
    sparse = materialize_world(load_scenarios(CONFIG)["static_sparse"], 0.0)
    dense = materialize_world(load_scenarios(CONFIG)["static_dense"], 0.0)
    self.assertGreater(dense.occupancy.sum(), sparse.occupancy.sum())

def test_initial_state_sweeps_change_exactly_one_factor():
    for group in initial_state_sweeps(load_showcase_config(CONFIG)):
        self.assertEqual(changed_fields(group.baseline, group.variant), {group.factor})
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_scenarios -v`

Expected: missing `experiments.rolling.scenarios` APIs.

- [ ] **Step 3: Implement scenario schema and world materialization**

```python
@dataclass(frozen=True)
class MovingDisc:
    obstacle_uid: str
    radius_m: float
    keyframes: tuple[tuple[float, float, float], ...]

    def state_at(self, time_s: float) -> ObstacleState:
        # Piecewise-linear interpolation with clamped endpoints.
        ...

def materialize_world(scenario: RollingScenario, time_s: float) -> WorldSnapshot:
    obstacles = scenario.obstacles_at(time_s)
    occupancy = rasterize_rectangles_and_discs(scenario, obstacles)
    return WorldSnapshot(occupancy=occupancy,
                         esdf_distance=compute_esdf(occupancy, scenario.resolution_m),
                         obstacles=obstacles)
```

The JSON must define the eight required scene families, complete guides/final goals, static rectangles, dynamic obstacle keyframes, and single-factor initial-state sweeps. It must reject overlapping UIDs, nonmonotonic keyframes, obstacles outside the declared world, missing final goals, and dynamic motion outside the experiment time range.

- [ ] **Step 4: Re-run scenario tests and validate configuration**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_scenarios -v`

Expected: all tests pass and every required family is present.

- [ ] **Step 5: Commit scenarios**

```bash
git add experiments/rolling/scenarios.py experiments/configs/rolling_showcase_v1.json experiments/tests/test_rolling_scenarios.py
git commit -m "feat: add deterministic rolling showcase scenarios"
```

### Task 3: Full-route rolling MINCO engine

**Files:**
- Create: `experiments/rolling/engine.py`
- Modify: `experiments/static/runner.py`
- Test: `experiments/tests/test_rolling_engine.py`

- [ ] **Step 1: Write failing behavioral tests with a deterministic planner double**

```python
def test_rollout_executes_prefix_and_reaches_full_guide_goal():
    result = run_rollout(straight_scenario(length_m=4.0), planner=linear_planner(),
                         method="legacy", config=config(execute_duration_s=0.5))
    self.assertEqual(result.status, "GOAL_REACHED")
    self.assertGreater(len(result.cycles), 1)
    np.testing.assert_allclose(result.executed_samples[-1, 1:4], [4, 0, 0], atol=0.05)

def test_next_cycle_receives_previous_executed_end_state():
    result = run_rollout(turn_scenario(), planner=recording_planner(), method="legacy", config=config())
    for left, right in zip(result.cycles, result.cycles[1:]):
        np.testing.assert_allclose(right.input_state.position_xyz, left.executed_samples[-1, 1:4])
        np.testing.assert_allclose(right.input_state.velocity_xyz_mps, left.executed_samples[-1, 4:7])
        self.assertAlmostEqual(right.input_state.yaw_rad, left.executed_samples[-1, 13])

def test_dynamic_world_is_rebuilt_each_cycle():
    result = run_rollout(dynamic_crossing(), planner=linear_planner(), method="safe_corridor_v1", config=config())
    self.assertNotEqual(result.cycles[0].obstacle_states, result.cycles[-1].obstacle_states)
```

Also test `COLLISION`, `OPTIMIZATION_FAILED`, `STALLED`, `MAX_CYCLES`, `TIMEOUT`, local-guide progress without backtracking, and safe-corridor geometry capture.

- [ ] **Step 2: Run engine tests and confirm RED**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_engine -v`

Expected: missing rolling engine.

- [ ] **Step 3: Expose a reusable native single-cycle planner**

Refactor `experiments/static/runner.py` so existing `run_static_case` calls the same focused adapter used by the rolling engine:

```python
def run_native_plan(*, guide_path_xyz, world, state, terminal_goal_xyz,
                    profile, reset_history=False) -> StaticRunResult:
    """Run one evidenced native cycle without changing experiment semantics."""
```

Keep current static benchmark behavior backward compatible. Do not reset MINCO history between rolling cycles; reset once at rollout start so hot-start behavior remains measurable.

- [ ] **Step 4: Implement local-guide progress and state propagation**

```python
def run_rollout(scenario, *, method, profile, config, planner=run_native_plan):
    state = scenario.initial_state
    progress_s = 0.0
    for cycle_index in range(config.max_cycles):
        world = materialize_world(scenario, time_s)
        local_guide, progress_s = select_local_guide(
            scenario.guide_path_xyz, state.position_xyz, progress_s,
            config.local_horizon_m)
        plan = planner(guide_path_xyz=local_guide, world=world, state=state,
                       terminal_goal_xyz=scenario.final_goal_xyz,
                       profile=profile)
        executed = select_execution_prefix(plan.samples, config.execute_duration_s,
                                           scenario.final_goal_xyz,
                                           config.goal_tolerance_m)
        state = RobotState.from_minco_sample(executed[-1])
        # Record cycle, then evaluate collision/goal/stall/limits in fixed order.
```

The final goal may only be passed as a stopping constraint when it lies within the selected local guide horizon; otherwise the local guide endpoint is the cycle goal. This prevents the previous single-call “jump from start to distant final goal” ambiguity.

- [ ] **Step 5: Run engine and legacy static regression tests**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_engine experiments.tests.test_static_runner experiments.tests.test_static_benchmark -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the engine**

```bash
git add experiments/rolling/engine.py experiments/static/runner.py experiments/tests/test_rolling_engine.py
git commit -m "feat: execute full-route rolling MINCO plans"
```

### Task 4: Canonical rollout serialization and validation

**Files:**
- Create: `experiments/rolling/serialization.py`
- Test: `experiments/tests/test_rolling_serialization.py`

- [ ] **Step 1: Write failing round-trip and tamper tests**

```python
def test_rollout_round_trip_preserves_every_cycle(tmp_path):
    receipt = write_rollout_result(make_result(cycle_count=3), tmp_path)
    loaded = load_rollout_result(receipt.manifest_path)
    self.assertEqual(len(loaded.cycles), 3)
    self.assertEqual(validate_rollout_result(tmp_path), [])

def test_changed_cycle_csv_fails_closed(tmp_path):
    write_rollout_result(make_result(), tmp_path)
    append_text(tmp_path / "cycle_metrics.csv", "tamper")
    self.assertTrue(any("hash" in error for error in validate_rollout_result(tmp_path)))
```

- [ ] **Step 2: Run serialization tests and confirm RED**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_serialization -v`

Expected: missing serialization module.

- [ ] **Step 3: Implement canonical evidence output**

Write `run_manifest.json`, `cycle_metrics.csv`, `executed_trajectory.csv`, `candidate_trajectories.npz`, `corridor_segments.csv`, `obstacle_states.csv`, `metrics.json`, and `artifact_receipt.json`. Every row must carry `scenario_uid`, `method`, and either `cycle_index` or an explicit all-run scope. Store unavailable values as blank/`null`, not numeric zero.

- [ ] **Step 4: Re-run serialization tests**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_serialization -v`

Expected: round-trip and tamper cases pass.

- [ ] **Step 5: Commit serialization**

```bash
git add experiments/rolling/serialization.py experiments/tests/test_rolling_serialization.py
git commit -m "feat: serialize validated rolling experiment evidence"
```

### Task 5: Paper figures, yaw arrows, corridors, and synchronized GIFs

**Files:**
- Create: `experiments/visualizers/rolling_showcase.py`
- Test: `experiments/tests/test_rolling_showcase.py`

- [ ] **Step 1: Write failing renderer tests**

```python
def test_scene_package_contains_all_paper_outputs(tmp_path):
    package = render_scene_package(make_paired_results(), tmp_path)
    required = {"three_panel.png", "three_panel.pdf", "overlay.png", "overlay.pdf",
                "safe_corridor.png", "safe_corridor.pdf", "three_way.gif",
                "figure_data.csv", "caption.md", "caption_zh.md",
                "artifact_receipt.json", "validation.json"}
    self.assertTrue(required.issubset({path.name for path in package.files}))

def test_xy_figures_and_gif_frames_contain_heading_arrow_metadata(tmp_path):
    package = render_scene_package(make_paired_results(yaw=[0.0, 1.57]), tmp_path)
    manifest = json.loads(package.manifest_path.read_text())
    self.assertEqual(manifest["visual_contract"]["robot_heading"], "ARROW")
    self.assertGreater(manifest["visual_contract"]["sampled_heading_arrow_count"], 1)

def test_safe_corridor_image_requires_real_corridor_segments(tmp_path):
    with self.assertRaisesRegex(ValueError, "corridor evidence"):
        render_scene_package(make_paired_results(corridor_segments=[]), tmp_path)
```

Also decode GIFs, require equal panel limits, verify frame count equals frame-data rows, and require captions to contain paired key, denominator, units, sample size, missing data, interpretation, and limitations.

- [ ] **Step 2: Run renderer tests and confirm RED**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_showcase -v`

Expected: missing renderer API.

- [ ] **Step 3: Implement fixed visual grammar**

```python
GUIDE_STYLE = {"color": "#666666", "linestyle": "-.", "linewidth": 1.6}
LEGACY_STYLE = {"color": "#D55E00", "linestyle": "--", "linewidth": 2.4}
SAFE_STYLE = {"color": "#0072B2", "linestyle": "-", "linewidth": 3.0}

def draw_heading_arrow(ax, xy, yaw_rad, *, color, length_m, hollow=False):
    dx, dy = length_m * np.cos(yaw_rad), length_m * np.sin(yaw_rad)
    ax.arrow(xy[0], xy[1], dx, dy, color=color,
             fill=not hollow, length_includes_head=True,
             head_width=0.12 * length_m, zorder=8)
```

Draw safe corridors as real capsule unions from recorded segment endpoints/radii. Draw current yaw with a thick arrow, velocity with a thin arrow, goal yaw with a hollow purple arrow or explicit `N/A`. Use unwrapped yaw for time curves. Freeze terminal/failed frames long enough to read the status.

- [ ] **Step 4: Generate image/GIF fixtures and run visual validators**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_showcase experiments.tests.test_video_evidence -v`

Expected: all media decode, 300-DPI PNG metadata and vector PDFs exist, captions and receipts validate.

- [ ] **Step 5: Commit renderers**

```bash
git add experiments/visualizers/rolling_showcase.py experiments/tests/test_rolling_showcase.py
git commit -m "feat: render rolling paper showcase comparisons"
```

### Task 6: Scenario matrix, extreme-case selection, and成果索引

**Files:**
- Create: `experiments/rolling/showcase.py`
- Modify: `experiments/configs/rolling_showcase_v1.json`
- Test: `experiments/tests/test_rolling_showcase_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

```python
def test_showcase_runs_fair_three_way_matrix(tmp_path):
    manifest = run_rolling_showcase(CONFIG, tmp_path, planner=fixture_planner())
    for scene in manifest["scenes"]:
        self.assertEqual(scene["paired_key"], "scenario_uid+seed+initial_state")
        self.assertEqual(scene["methods"], ["guide_reference", "legacy", "safe_corridor_v1"])

def test_syn_dense_short_is_not_selected_as_corridor_effect_case(tmp_path):
    manifest = run_rolling_showcase(CONFIG, tmp_path, planner=fixture_planner())
    self.assertNotIn("syn_dense_short", manifest["extreme_corridor_case_uids"])

def test_chinese_index_links_every_required_category(tmp_path):
    run_rolling_showcase(CONFIG, tmp_path, planner=fixture_planner())
    text = (tmp_path / "README_成果索引.md").read_text()
    for heading in ("轨迹优化", "安全走廊", "不同初值", "动态障碍", "极端案例"):
        self.assertIn(heading, text)
```

- [ ] **Step 2: Run pipeline tests and confirm RED**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_showcase_pipeline -v`

Expected: missing showcase pipeline.

- [ ] **Step 3: Implement method pairing and locked selection**

Run guide-reference rendering without claiming planning metrics, execute Legacy and Safe with the same scenario instance and seed, then rank calibration cases using a versioned score composed of trajectory distortion, clearance deficit, failure state, and corridor improvement. Store all candidate scores in `extreme_case_selection.csv`; select before rendering the locked comparison package and report every locked case.

- [ ] **Step 4: Assemble the requested directory tree and aggregate figures**

Create `01_trajectory_optimization` through `06_aggregate_figures`; place scene packages in their semantic folder and write `README_成果索引.md` with relative links, “看什么/证明什么/限制是什么” for each figure. Generate aggregate success, clearance, full-route length, dynamics, latency, and failure-reason figures from recorded CSV only.

- [ ] **Step 5: Validate the full tree and run tests**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_rolling_showcase_pipeline experiments.tests.test_rolling_showcase -v`

Expected: complete exact inventory, no unreceipted files, no missing categories, no fabricated guide metrics.

- [ ] **Step 6: Commit pipeline**

```bash
git add experiments/rolling/showcase.py experiments/configs/rolling_showcase_v1.json experiments/tests/test_rolling_showcase_pipeline.py
git commit -m "feat: build complete rolling experiment showcase"
```

### Task 7: One-click workflow integration

**Files:**
- Modify: `experiments/orchestrators/research_workflow.py`
- Modify: `scripts/run_static_experiments.sh`
- Modify: `scripts/run_all_experiments.sh`
- Modify: `experiments/tests/test_research_workflow.py`
- Modify: `experiments/tests/test_shell_entrypoints.py`

- [ ] **Step 1: Write failing workflow tests**

```python
def test_static_workflow_generates_and_validates_paper_showcase(self):
    receipt = run_static_workflow(self.options)
    self.assertEqual(receipt["stages"]["rolling_showcase"]["status"], "COMPLETE")
    self.assertEqual(receipt["validation_errors"], [])
    self.assertTrue((self.output / "paper_showcase" / "README_成果索引.md").is_file())

def test_all_workflow_receipt_includes_showcase_hashes(self):
    receipt = json.loads((self.output / "experiment_receipt.json").read_text())
    self.assertIn("paper_showcase/showcase_manifest.json",
                  {row["path"] for row in receipt["artifacts"]})
```

- [ ] **Step 2: Run workflow tests and confirm RED**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_research_workflow experiments.tests.test_shell_entrypoints -v`

Expected: rolling-showcase stage absent.

- [ ] **Step 3: Add resumable fail-closed showcase stage**

Insert the stage after native build/static boundary work and before `generate_paper_report`. Its inputs include both static profiles, rolling config, calibration receipt, and native extension hash. Its outputs are the whole `paper_showcase` tree; resume must reject changed inputs or changed output hashes.

- [ ] **Step 4: Expose shell arguments without weakening real-run authorization**

Add `--rolling-showcase-config PATH` and `--skip-rolling-showcase`. Keep showcase enabled by default for local static runs. Real IsaacLab remains guarded by `--allow-real-simulation`; local deterministic dynamic showcase does not launch IsaacLab.

- [ ] **Step 5: Run workflow/shell tests and syntax checks**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest experiments.tests.test_research_workflow experiments.tests.test_shell_entrypoints -v`

Run: `bash -n scripts/run_static_experiments.sh scripts/run_simulation_experiments.sh scripts/run_all_experiments.sh`

Expected: all tests and syntax checks pass.

- [ ] **Step 6: Commit orchestration**

```bash
git add experiments/orchestrators/research_workflow.py scripts/run_static_experiments.sh scripts/run_all_experiments.sh experiments/tests/test_research_workflow.py experiments/tests/test_shell_entrypoints.py
git commit -m "feat: run rolling showcase in one-click workflow"
```

### Task 8: Documentation, clean regeneration, and final evidence audit

**Files:**
- Modify: `experiments/README.md`
- Generated: `results/<new_run_id>/paper_showcase/**`
- Generated: `results/<new_run_id>/experiment_receipt.json`

- [ ] **Step 1: Document exact commands and output entry points**

Document:

```bash
NAVDP_PYTHON=/home/alioth/miniforge3/envs/navdp/bin/python \
  bash scripts/run_static_experiments.sh --output results/<new_run_id>

NAVDP_PYTHON=/home/alioth/miniforge3/envs/navdp/bin/python \
ISAACLAB_PYTHON=/home/alioth/miniforge3/envs/isaaclab/bin/python \
  bash scripts/run_all_experiments.sh --output results/<new_run_id>
```

Point readers first to `paper_showcase/README_成果索引.md`, then to the aggregate figures and scene packages.

- [ ] **Step 2: Run focused and complete regression suites before deleting results**

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m unittest discover -s experiments/tests -v`

Run: `cmake --build minco_processor/build -j2 && ctest --test-dir minco_processor/build --output-on-failure`

Run: `/home/alioth/miniforge3/envs/navdp/bin/python -m compileall -q experiments`

Run: `git diff --check`

Expected: Python/native/compile/diff checks pass, with only explicitly documented skips.

- [ ] **Step 3: Resolve exact disposable iteration directories**

Use a read-only inventory and require every target to be an immediate child of `/home/alioth/NavDP/results` with a workflow receipt or generated artifact receipt. The authorized targets are currently:

```text
/home/alioth/NavDP/results/navdp_minco_paper_local_verification_20260812
/home/alioth/NavDP/results/navdp_minco_static_real
/home/alioth/NavDP/results/navdp_video_evidence_local_20260812
```

Abort cleanup if a resolved path escapes the results directory or contains source/config files outside the generated artifact inventory.

- [ ] **Step 4: Remove only the verified old iteration outputs**

After validation and immediately before regeneration, delete the three exact directories above. Do not use globs, environment-variable targets, the repository root, or the whole `results` directory. Report that this deletion is not recoverable from Git.

- [ ] **Step 5: Run a clean local one-click experiment**

Run:

```bash
NAVDP_PYTHON=/home/alioth/miniforge3/envs/navdp/bin/python \
ISAACLAB_PYTHON=/home/alioth/miniforge3/envs/isaaclab/bin/python \
bash scripts/run_all_experiments.sh \
  --output /home/alioth/NavDP/results/navdp_rolling_showcase_20260812
```

Do not pass `--allow-real-simulation`; the command must generate static and deterministic local-dynamic evidence while leaving IsaacLab real execution pending.

- [ ] **Step 6: Audit the regenerated evidence**

Verify `validation_errors=[]`, exact artifact receipt inventory, media decoding, 300-DPI PNGs, vector PDFs, monotonic frame/cycle timestamps, matching GIF frame/CSV row counts, nonempty actual corridor geometry in safe cases, visible heading-arrow metadata, and complete success/failure termination for every route.

Manually inspect at least: no-obstacle baseline, sparse obstacle, dense obstacle, narrow passage, malformed-detour extreme, initial-yaw reverse, dynamic crossing, dynamic head-on, and sudden-appearance three-way packages.

- [ ] **Step 7: Final regression against regenerated result**

Run the showcase validator directly and rerun workflow receipt validation. Expected status is `READY_FOR_REAL_RUN` with local dynamic evidence marked `SIMULATED` and IsaacLab-only evidence marked `PENDING_REAL_SIMULATION`.

- [ ] **Step 8: Commit documentation and source-only changes**

```bash
git add experiments/README.md
git commit -m "docs: index rolling paper showcase artifacts"
```

Do not commit generated `results/` unless repository policy explicitly changes.
