from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics

import numpy as np

from experiments.analyzers.statistics import bootstrap_ci, proportion_ci


TABLES = ("table_data_quality", "table_raw_profile", "table_safety_repair", "table_smoothness", "table_warm_start", "table_control_navigation", "table_timing")
FIELDS = ["metric", "n", "mean_or_rate", "median", "p95", "ci95_low", "ci95_high", "baseline_delta", "relative_change_percent", "data_source", "method"]


def _read(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream: rows.extend(csv.DictReader(stream))
    return rows


def _number(row, field):
    try:
        value = float(row.get(field, ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _boolean(row, field):
    value = str(row.get(field, "")).strip().lower()
    return 1.0 if value == "true" else 0.0 if value == "false" else None


def _metric_rows(rows, definitions, data_source):
    output = []
    methods = sorted({row.get("variant", "UNKNOWN") for row in rows})
    means = {}
    pending = []
    for method in methods:
        method_rows = [row for row in rows if row.get("variant", "UNKNOWN") == method]
        for metric, field, kind in definitions:
            values = [(_boolean(row, field) if kind == "binary" else _number(row, field)) for row in method_rows]
            values = np.asarray([value for value in values if value is not None], dtype=float)
            if not len(values): continue
            mean = float(np.mean(values)); means[(method, metric)] = mean
            if kind == "binary": interval = proportion_ci(int(np.sum(values)), len(values)); low, high = interval["low"], interval["high"]
            else: low, high = bootstrap_ci(values)
            pending.append((method, metric, values, mean, low, high))
    for method, metric, values, mean, low, high in pending:
        baseline = means.get(("raw", metric)); delta = mean - baseline if baseline is not None else float("nan")
        relative = delta / abs(baseline) * 100.0 if baseline not in (None, 0.0) else float("nan")
        output.append({
            "metric":metric, "n":len(values), "mean_or_rate":mean,
            "median":float(np.median(values)), "p95":float(np.quantile(values, .95)),
            "ci95_low":low, "ci95_high":high, "baseline_delta":delta,
            "relative_change_percent":relative, "data_source":data_source, "method":method,
        })
    return output


def generate_result_tables(suite_dir, episode_count=None, plan_count=None):
    suite_dir = Path(suite_dir); output = suite_dir / "reports" / "core_tables"; output.mkdir(parents=True, exist_ok=True)
    episodes = _read(suite_dir.glob("experiments/*/*/*/*/*/episode_metrics.csv"))
    plans = _read(suite_dir.glob("experiments/*/*/*/*/*/plan_metrics.csv"))
    cycles = _read(suite_dir.glob("experiments/*/*/*/*/*/planning_cycles.csv"))
    controls = _read(suite_dir.glob("experiments/*/*/*/*/*/control_samples.csv"))
    timings = _read(suite_dir.glob("experiments/*/*/*/*/*/timing_samples.csv"))
    sources = {row.get("data_source") for rows in (episodes, plans, cycles, controls, timings) for row in rows if row.get("data_source")}
    data_source = next(iter(sources)) if len(sources) == 1 else "MIXED" if sources else "UNKNOWN"
    definitions = {
        "table_data_quality": (cycles, [("plan_publish_rate","published","binary"), ("stale_rate","stale","binary")]),
        "table_raw_profile": (plans, [("raw_min_clearance_m","raw_min_clearance_m","continuous"), ("raw_unsafe_ratio","raw_unsafe_ratio","continuous"), ("raw_curvature_tv_1pm","raw_curvature_tv_1pm","continuous")]),
        "table_safety_repair": (cycles, [("plan_publish_rate","published","binary"), ("optimizer_success_rate","optimizer_success","binary"), ("validation_success_rate","python_validation_success","binary")]),
        "table_smoothness": (plans, [
            ("curvature_tv_1pm","raw_curvature_tv_1pm","continuous"),
            ("curvature_rate_rms_1pm2","raw_curvature_rate_rms_1pm2","continuous"),
            ("actual_speed_mean_mps","actual_speed_mean_mps","continuous"),
            ("actual_speed_p95_mps","actual_speed_p95_mps","continuous"),
            ("actual_acc_rms_mps2","actual_acc_rms_mps2","continuous"),
            ("actual_acc_p95_mps2","actual_acc_p95_mps2","continuous"),
            ("actual_jerk_rms_mps3","actual_jerk_rms_mps3","continuous"),
            ("actual_yaw_rate_rms_radps","actual_yaw_rate_rms_radps","continuous"),
        ]),
        "table_warm_start": (plans, [("hot_accept_rate","hot_start_accepted","binary"), ("interplan_position_rmse_m","raw_interplan_position_rmse_m","continuous"), ("initial_tangent_jump_rad","raw_initial_tangent_jump_rad","continuous")]),
        "table_control_navigation": (episodes, [("success_rate","success","binary"), ("collision_rate","collision","binary"), ("tracking_rmse_m","tracking_error_rmse_m","continuous"), ("duration_s","episode_duration_s","continuous"), ("path_length_m","actual_path_length_m","continuous"), ("spl","repository_spl","continuous")]),
        "table_timing": (timings, [("duration_ms","duration_ms","continuous")]),
    }
    for name in TABLES:
        rows, metrics = definitions[name]; result_rows = _metric_rows(rows, metrics, data_source)
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(result_rows)
        lines = ["| Metric | Method | n | Mean/rate | Median | P95 | 95% CI | Baseline Δ | Source |", "|---|---|---:|---:|---:|---:|---|---:|---|"]
        for row in result_rows:
            lines.append(f"| {row['metric']} | {row['method']} | {row['n']} | {row['mean_or_rate']:.6g} | {row['median']:.6g} | {row['p95']:.6g} | [{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] | {row['baseline_delta']:.6g} | {row['data_source']} |")
        (output / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
