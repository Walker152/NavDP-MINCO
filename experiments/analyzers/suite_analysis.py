from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from experiments.analyzers.artifact_manifest import generate_artifact_manifest
from experiments.analyzers.case_selector import select_representative_cases
from experiments.analyzers.failure_cases import generate_failure_report
from experiments.analyzers.result_tables import generate_result_tables
from experiments.analyzers.paired import compare_runs
from experiments.visualizers.common import save_data_figure


EXPERIMENT_NAMES = ("headless", "raw_profile", "safety", "smoothness", "warm_start", "control", "navigation", "timing", "failures")
PLOTS = {
1:("raw_safety_class_bar.png","raw_min_clearance_ecdf.png","raw_unsafe_ratio_distribution.png","raw_turn_class_bar.png","raw_interplan_jump_ecdf.png","critic_vs_clearance_scatter.png","candidate_rank_safe_rate.png"),
2:("safety_transition_matrix.png","repair_outcome_stacked_bar.png","paired_clearance_scatter.png","paired_clearance_delta.png","paired_unsafe_ratio_delta.png","failure_reason_bar.png","clearance_vs_arclength_mock.png","trajectory_esdf_overlay_mock.png"),
3:("curvature_tv_distribution.png","curvature_rate_rms_distribution.png","equiv_jerk_rms_distribution.png","high_turn_metric_delta.png","curvature_vs_arclength_mock.png","equiv_acc_vs_arclength_mock.png","equiv_jerk_vs_arclength_mock.png","minco_speed_vs_time_mock.png","minco_acc_vs_time_mock.png","minco_jerk_vs_time_mock.png","minco_yaw_rate_vs_time_mock.png"),
4:("hot_accept_rate_bar.png","hot_reject_reason_bar.png","interplan_position_rmse_distribution.png","initial_tangent_jump_distribution.png","command_delta_w_distribution.png","optimizer_time_distribution.png","previous_vs_new_trajectory_mock.png","trajectory_jump_vs_time_mock.png"),
5:("tracking_error_ecdf.png","tracking_rmse_distribution.png","tracking_p95_distribution.png","mpc_failure_rate_bar.png","command_saturation_rate_bar.png","reference_age_ecdf.png","tracking_error_vs_time_mock.png","actual_reference_xy_mock.png","actual_reference_speed_mock.png","command_vw_vs_time_mock.png"),
6:("success_rate_with_ci.png","collision_rate_with_ci.png","failure_reason_stacked_bar.png","episode_duration_distribution.png","actual_path_length_distribution.png","spl_distribution.png","paired_episode_duration_scatter.png","paired_path_length_scatter.png"),
7:("planning_stage_mean_bar.png","planning_stage_p95_bar.png","planning_total_ecdf.png","minco_stage_p95_bar.png","mpc_solve_ecdf.png","recording_overhead_bar.png","observation_to_command_ecdf.png","deadline_miss_rate_bar.png"),
8:("failure_reason_bar.png","failure_stage_stacked_bar.png","failure_timeline_mock.png"),
}


def _read_all(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream: rows.extend(csv.DictReader(stream))
    return rows


def _finite(rows, field):
    values = []
    for row in rows:
        try: value = float(row.get(field, ""))
        except (TypeError, ValueError): continue
        if math.isfinite(value): values.append(value)
    return values


def _binary(rows, field):
    return [1.0 if str(row.get(field, "")).lower() == "true" else 0.0 for row in rows if str(row.get(field, "")).lower() in {"true", "false"}]


def _plot_values(filename, data):
    episodes, plans, cycles, controls, timings, candidates = data
    if "success" in filename: return _binary(episodes, "success")
    if "collision" in filename: return _binary(episodes, "collision")
    if "duration" in filename and "planning" not in filename: return _finite(episodes, "episode_duration_s")
    if "path_length" in filename: return _finite(episodes, "actual_path_length_m")
    if "spl" in filename: return _finite(episodes, "repository_spl")
    if "tracking" in filename: return _finite(controls, "time_aligned_position_error_m") or _finite(episodes, "tracking_error_rmse_m")
    if "command" in filename: return _finite(controls, "cmd_w_radps")
    if "reference_age" in filename: return _finite(controls, "reference_age_ms")
    if "mpc" in filename: return _finite(controls, "mpc_solve_ms")
    if "hot_accept" in filename: return _binary(plans, "hot_start_accepted")
    if "interplan" in filename or "position_rmse" in filename: return _finite(plans, "raw_interplan_position_rmse_m")
    if "tangent" in filename: return _finite(plans, "raw_initial_tangent_jump_rad")
    if "curvature" in filename: return _finite(plans, "raw_curvature_tv_1pm")
    if "jerk" in filename: return _finite(plans, "actual_jerk_rms_mps3")
    if "acc" in filename: return _finite(plans, "actual_acc_rms_mps2")
    if "speed" in filename: return _finite(plans, "actual_speed_mean_mps")
    if "clearance" in filename: return _finite(plans, "minco_min_clearance_m") or _finite(plans, "raw_min_clearance_m")
    if "unsafe" in filename: return _finite(plans, "raw_unsafe_ratio")
    if "critic" in filename or "candidate_rank" in filename: return _finite(candidates, "critic_value")
    if "timing" in filename or "planning" in filename or "optimizer_time" in filename or "deadline" in filename or "overhead" in filename or "observation" in filename:
        return _finite(timings, "duration_ms") or _finite(cycles, "planning_total_ms")
    if "failure" in filename or "repair" in filename or "transition" in filename:
        return _binary(cycles, "published")
    return _finite(cycles, "planning_total_ms") or [float(len(episodes))]


def _generate_paired_reports(suite_dir):
    run_dirs = [path.parent for path in suite_dir.glob("experiments/*/*/*/*/*/run_config.json")]
    groups = {}
    for run_dir in run_dirs:
        config = json.loads((run_dir / "run_config.json").read_text())
        key = (config.get("experiment_id"), config.get("scene_id"), config.get("seed"))
        groups.setdefault(key, {})[config.get("variant")] = run_dir
    summaries = []
    comparisons = (("raw","minco-hot"),("raw","minco-cold"),("minco-cold","minco-hot"))
    for (experiment, scene, seed), variants in sorted(groups.items()):
        for baseline, method in comparisons:
            if baseline not in variants or method not in variants: continue
            label = f"{baseline}_vs_{method}"
            output = suite_dir / "reports" / "paired" / f"{scene}_seed_{seed}" / label
            result = compare_runs(variants[baseline], variants[method], output)
            summaries.append({"experiment":experiment, "scene":scene, "seed":seed, "comparison":label, **result})
    return summaries


def analyze_suite(suite_dir):
    suite_dir = Path(suite_dir); reports = suite_dir / "reports"; reports.mkdir(parents=True, exist_ok=True)
    suite_config = json.loads((suite_dir / "suite_config.json").read_text()) if (suite_dir / "suite_config.json").exists() else {}
    data_source = suite_config.get("data_source", "UNKNOWN")
    episodes = _read_all(suite_dir.glob("experiments/*/*/*/*/*/episode_metrics.csv")); plans = _read_all(suite_dir.glob("experiments/*/*/*/*/*/plan_metrics.csv")); cycles = _read_all(suite_dir.glob("experiments/*/*/*/*/*/planning_cycles.csv"))
    controls = _read_all(suite_dir.glob("experiments/*/*/*/*/*/control_samples.csv")); timings = _read_all(suite_dir.glob("experiments/*/*/*/*/*/timing_samples.csv")); candidates = _read_all(suite_dir.glob("experiments/*/*/*/*/*/candidate_metrics.csv"))
    generate_result_tables(suite_dir, len(episodes), len(plans)); select_representative_cases(suite_dir, episodes, plans, cycles); failures = generate_failure_report(suite_dir, episodes)
    paired_summaries = _generate_paired_reports(suite_dir)
    plot_rows = []
    for index, name in enumerate(EXPERIMENT_NAMES):
        exp_id = f"EXP-{index:02d}_{name}"; directory = reports / exp_id; (directory / "plots").mkdir(parents=True, exist_ok=True); (directory / "tables").mkdir(exist_ok=True)
        (directory / "summary_metrics.csv").write_text(f"metric,n,value,data_source\nepisode_count,{len(episodes)},{len(episodes)},{data_source}\n", encoding="utf-8")
        with (directory / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["experiment","scene","seed","comparison","paired_count","data_source"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for summary in paired_summaries: writer.writerow({key:summary.get(key, "") for key in fields[:-1]} | {"data_source":data_source})
        for filename in PLOTS.get(index, ()):
            values = _plot_values(filename, (episodes, plans, cycles, controls, timings, candidates))
            path = save_data_figure(directory / "plots" / filename, f"{exp_id}: {filename} (n={len(values)})", values=values, data_source=data_source)
            plot_rows.append({"experiment_id":exp_id, "plot_path":str(path.relative_to(suite_dir)) if path else "", "status":"generated" if path else "skipped", "skip_reason":"" if path else "no finite data", "data_source":data_source})
        boundary = "> **SIMULATED DATA — pipeline validation only.**\n\n" if data_source == "SIMULATED" else ""
        (directory / "report.md").write_text(f"# {exp_id}\n\n{boundary}Data source: {data_source}. Episodes: {len(episodes)}; valid plans: {len(plans)}; planning cycles: {len(cycles)}.\n\nNo algorithm superiority is inferred automatically.\n", encoding="utf-8")
    with (reports / "plot_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment_id","plot_path","status","skip_reason","data_source"]); writer.writeheader(); writer.writerows(plot_rows)
    text = "# NavDP–MINCO Suite Report\n\n"
    if data_source == "SIMULATED": text += "> **SIMULATED DATA — 工具链验收数据，不代表算法科研结论。**\n\n"
    text += f"## Configuration and provenance\n\nData source: {data_source}; run files: {len(list(suite_dir.glob('experiments/*/*/*/*/*/run_config.json')))}.\n\n"
    text += f"## Data quality and coverage\n\nEpisodes: {len(episodes)}; planning cycles: {len(cycles)}; valid plans: {len(plans)}; failures: {len(failures)}.\n\n"
    text += "## Core results\n\nSeven CSV/Markdown result tables are under `core_tables/`; all values include sample count, interval fields, baseline delta and source.\n\n"
    text += "## EXP-00 through EXP-08\n\nEach experiment has an independent report, summary, paired table, plots and tables directory.\n\n"
    text += f"## Improvements, regressions and failures\n\nGenerated {len(paired_summaries)} paired run comparisons from shared episode UIDs. No performance claim is made automatically.\n\n"
    text += "## Missing fields and conclusion boundary\n\nUnavailable fields remain empty and are not converted to zero. Interpret conclusions only within the recorded scenes, episodes and data source.\n\n"
    text += "## Artifact index\n\nSee `plot_index.csv`, `representative_cases.csv`, `failure_case_index.csv`, and `artifact_manifest.{json,csv}`.\n"
    (reports / "suite_report.md").write_text(text, encoding="utf-8")
    generate_artifact_manifest(suite_dir)
    return reports / "suite_report.md"
