#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def parse_candidate_failures(reason: str) -> list[str]:
    """Extract one normalized failure category for each attempted candidate."""
    categories = []
    for item in str(reason or "").split("; "):
        value = item.split(": ", 1)[1] if ": " in item else item
        value = value.strip()
        if not value:
            continue
        if value.startswith("PY_ESDF_UNSAFE"):
            value = "PY_ESDF_UNSAFE"
        else:
            value = re.split(r"\s+", value, maxsplit=1)[0]
        categories.append(value)
    return categories


def _xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    points = points[:, :2]
    return points[np.all(np.isfinite(points), axis=1)]


def path_metrics(points: np.ndarray) -> dict[str, float | bool]:
    path = _xy(points)
    if len(path) < 2:
        return {
            "length_m": 0.0,
            "displacement_m": 0.0,
            "directness": math.nan,
            "mean_segment_m": math.nan,
            "segment_cv": math.nan,
            "max_turn_deg": math.nan,
            "turn_tv_deg": math.nan,
            "has_reversal": False,
        }
    delta = np.diff(path, axis=0)
    segment = np.linalg.norm(delta, axis=1)
    length = float(segment.sum())
    displacement = float(np.linalg.norm(path[-1] - path[0]))
    nonzero = segment > 1e-9
    directions = delta[nonzero]
    turns = np.empty(0, dtype=np.float64)
    if len(directions) >= 2:
        dot = np.sum(directions[:-1] * directions[1:], axis=1)
        denom = np.linalg.norm(directions[:-1], axis=1) * np.linalg.norm(directions[1:], axis=1)
        turns = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
    mean_segment = float(segment.mean()) if len(segment) else math.nan
    return {
        "length_m": length,
        "displacement_m": displacement,
        "directness": length / displacement if displacement > 1e-9 else math.nan,
        "mean_segment_m": mean_segment,
        "segment_cv": float(segment.std() / mean_segment) if mean_segment > 1e-9 else math.nan,
        "max_turn_deg": float(turns.max()) if len(turns) else 0.0,
        "turn_tv_deg": float(turns.sum()) if len(turns) else 0.0,
        "has_reversal": bool(np.any(turns >= 150.0)),
    }


def _resample_arclength(points: np.ndarray, count: int = 101) -> np.ndarray:
    path = _xy(points)
    if len(path) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(path) == 1:
        return np.repeat(path, count, axis=0)
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    path = path[keep]
    cumulative = cumulative[keep]
    if cumulative[-1] <= 1e-9:
        return np.repeat(path[:1], count, axis=0)
    target = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([
        np.interp(target, cumulative, path[:, axis]) for axis in range(2)
    ])


def deformation_metrics(guide: np.ndarray, optimized: np.ndarray) -> dict[str, float]:
    guide_rs = _resample_arclength(guide)
    optimized_rs = _resample_arclength(optimized)
    if len(guide_rs) == 0 or len(optimized_rs) == 0:
        return {key: math.nan for key in (
            "corresponding_mean_m", "corresponding_p95_m", "corresponding_max_m",
            "hausdorff_m", "length_ratio", "endpoint_shift_m",
        )}
    corresponding = np.linalg.norm(guide_rs - optimized_rs, axis=1)
    pairwise = np.linalg.norm(guide_rs[:, None, :] - optimized_rs[None, :, :], axis=2)
    hausdorff = max(float(pairwise.min(axis=0).max()), float(pairwise.min(axis=1).max()))
    guide_length = float(np.linalg.norm(np.diff(guide_rs, axis=0), axis=1).sum())
    optimized_length = float(np.linalg.norm(np.diff(optimized_rs, axis=0), axis=1).sum())
    return {
        "corresponding_mean_m": float(corresponding.mean()),
        "corresponding_p95_m": float(np.percentile(corresponding, 95)),
        "corresponding_max_m": float(corresponding.max()),
        "hausdorff_m": hausdorff,
        "length_ratio": optimized_length / guide_length if guide_length > 1e-9 else math.nan,
        "endpoint_shift_m": float(np.linalg.norm(guide_rs[-1] - optimized_rs[-1])),
    }


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "results" / "navdp_minco_full_real"
DEFAULT_OUT = ROOT / "codex_ana" / "new_results_analysis"
VARIANTS = ["raw", "minco-cold", "minco-hot"]
SCENES = ["SPARSE", "DENSE"]
VARIANT_LABELS = {"raw": "RAW", "minco-cold": "MINCO-COLD", "minco-hot": "MINCO-HOT"}
VARIANT_COLORS = {"raw": "#355070", "minco-cold": "#2A9D8F", "minco-hot": "#E76F51"}
INK = "#263238"
GRID = "#D9E1E5"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_all(suite: Path, filename: str) -> list[dict[str, str]]:
    rows = []
    for path in sorted(suite.glob(f"experiments/**/{filename}")):
        for row in read_csv(path):
            row["_source_file"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def as_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def as_bool(value) -> bool | None:
    value = str(value).strip().lower()
    return True if value == "true" else False if value == "false" else None


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q: float) -> float:
    clean = [as_float(value) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    return float(np.quantile(clean, q)) if clean else math.nan


def mean(values) -> float:
    clean = [as_float(value) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    return float(np.mean(clean)) if clean else math.nan


def run_directories(suite: Path) -> list[Path]:
    return sorted(path.parent for path in suite.glob("experiments/**/run_config.json"))


def group_rows(rows, *keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def summarize_runs(episodes, plans) -> list[dict]:
    plan_groups = group_rows(plans, "scene_label", "variant")
    output = []
    for (scene, variant), rows in sorted(
        group_rows(episodes, "scene_label", "variant").items(),
        key=lambda item: (SCENES.index(item[0][0]), VARIANTS.index(item[0][1])),
    ):
        plan_rows = plan_groups[(scene, variant)]
        output.append({
            "scene": scene,
            "variant": variant,
            "episodes": len(rows),
            "successes": sum(as_bool(row.get("success")) is True for row in rows),
            "success_rate": sum(as_bool(row.get("success")) is True for row in rows) / len(rows),
            "mean_duration_s": mean(row.get("episode_duration_s") for row in rows),
            "mean_path_length_m": mean(row.get("actual_path_length_m") for row in rows),
            "mean_spl": mean(row.get("repository_spl") for row in rows),
            "planning_cycles": sum(int(as_float(row.get("planning_count"))) for row in rows),
            "minco_ok": sum(int(as_float(row.get("minco_ok_count"))) for row in rows),
            "hold_count": sum(int(as_float(row.get("hold_count"))) for row in rows),
            "stop_count": sum(int(as_float(row.get("stop_count"))) for row in rows),
            "cold_plan_rows": sum(row.get("planning_state") == "COLD_START" for row in plan_rows),
            "hot_plan_rows": sum(row.get("planning_state") == "HOT_START" for row in plan_rows),
        })
    return output


def summarize_planning(cycles) -> tuple[list[dict], list[dict], list[dict]]:
    outcomes = []
    failures = []
    cycle_flags = []
    aliases = {
        "VALIDATION_CLEARANCE": "Clearance validation",
        "PY_ESDF_UNSAFE": "Python ESDF recheck",
        "OPTIMIZER_FAILED": "Optimizer failed",
        "VALIDATION_ACCELERATION_LIMIT": "Acceleration limit",
        "PATH_REVERSAL": "Path reversal",
        "PATH_TOO_SHORT": "Path too short",
        "VALIDATION_VELOCITY_LIMIT": "Velocity limit",
        "PATH_START_GAP_TOO_LARGE": "Start gap",
        "CANDIDATE_TIME_BUDGET_EXHAUSTED": "Time budget",
    }
    for (scene, variant), rows in sorted(group_rows(cycles, "scene_label", "variant").items()):
        if variant == "raw":
            continue
        counts = Counter()
        for row in rows:
            if as_bool(row.get("stale")) is True:
                counts["stale"] += 1
            elif as_bool(row.get("published")) is True:
                counts["published"] += 1
            elif row.get("fallback_mode") == "HOLD_LAST":
                counts["hold"] += 1
            else:
                counts["stop"] += 1
        total = sum(counts.values())
        outcomes.append({
            "scene": scene,
            "variant": variant,
            "cycles": total,
            **{name: counts[name] for name in ("published", "hold", "stop", "stale")},
            **{f"{name}_rate": counts[name] / total if total else math.nan for name in ("published", "hold", "stop", "stale")},
        })

        reason_counts = Counter()
        failed_cycles = 0
        for row in rows:
            if as_bool(row.get("published")) is not False or as_bool(row.get("stale")) is True:
                continue
            failed_cycles += 1
            reasons = parse_candidate_failures(row.get("failure_reason", ""))
            reason_counts.update(reasons)
            safety = any(reason in {"VALIDATION_CLEARANCE", "PY_ESDF_UNSAFE"} for reason in reasons)
            optimizer = "OPTIMIZER_FAILED" in reasons
            dynamics = any(reason in {"VALIDATION_ACCELERATION_LIMIT", "VALIDATION_VELOCITY_LIMIT"} for reason in reasons)
            input_path = any(reason in {"PATH_REVERSAL", "PATH_TOO_SHORT", "START_DISCONNECTED"} for reason in reasons)
            cycle_flags.append({
                "scene": scene,
                "variant": variant,
                "episode_uid": row.get("episode_uid"),
                "planning_cycle_uid": row.get("planning_cycle_uid"),
                "fallback_mode": row.get("fallback_mode"),
                "candidate_attempts_parsed": len(reasons),
                "has_safety_failure": safety,
                "has_optimizer_failure": optimizer,
                "has_dynamics_failure": dynamics,
                "has_input_path_failure": input_path,
                "failure_reason": row.get("failure_reason"),
            })
        attempted = sum(reason_counts.values())
        for reason, count in reason_counts.most_common():
            failures.append({
                "scene": scene,
                "variant": variant,
                "failure_code": reason,
                "failure_category": aliases.get(reason, reason.replace("_", " ").title()),
                "candidate_attempts": count,
                "candidate_attempt_share": count / attempted if attempted else math.nan,
                "failed_cycles": failed_cycles,
                "all_attempts_in_failed_cycles": attempted,
            })
    return outcomes, failures, cycle_flags


def summarize_controls(controls) -> list[dict]:
    output = []
    for (scene, variant), rows in sorted(group_rows(controls, "scene_label", "variant").items()):
        active = [row for row in rows if row.get("control_state") == "CONTROL_ACTIVE"]
        observable = [
            row for row in active
            if math.isfinite(as_float(row.get("planned_v_mps"))) and math.isfinite(as_float(row.get("cmd_v_mps")))
        ]
        mismatch = [
            row for row in observable
            if as_float(row.get("planned_v_mps")) > 0.05 and abs(as_float(row.get("cmd_v_mps"))) <= 0.01
        ]
        states = Counter(row.get("control_state") or "<empty>" for row in rows)
        output.append({
            "scene": scene,
            "variant": variant,
            "samples": len(rows),
            "active_samples": len(active),
            "planned_velocity_observable_active": len(observable),
            "expected_motion_zero_samples": len(mismatch),
            "expected_motion_zero_rate": len(mismatch) / len(observable) if observable else math.nan,
            "detected_stall_samples": sum(row.get("zero_command_reason") == "EXPECTED_MOTION_ZERO_STALL" for row in rows),
            "minco_stop_samples": states["MINCO_STOP"],
            "minco_hold_samples": states["MINCO_HOLD_LAST"],
            "mpc_reference_rejected_samples": states["MPC_REFERENCE_REJECTED"],
            "state_counts": json.dumps(states, ensure_ascii=False, sort_keys=True),
        })
    return output


def trace_context(trace_path: Path) -> tuple[str, str]:
    parts = trace_path.parts
    exp_index = parts.index("EXP-ALL_data_collection")
    return parts[exp_index + 1], parts[exp_index + 2]


def trace_cycle_uid(trace_path: Path) -> str:
    name = trace_path.name
    prefix = "planning_trace_"
    suffix = ".npz"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return trace_path.stem


def trace_run_id(trace_path: Path) -> str:
    return trace_path.parent.parent.name


def collect_trace_metrics(suite: Path, valid_trace_keys: set[tuple[str, str]]) -> tuple[list[dict], list[dict], int]:
    paths = sorted(suite.glob("experiments/**/traces/*.npz"))
    geometry_rows = []
    deformation_rows = []
    stale_trace_count = 0
    for index, path in enumerate(paths, 1):
        if (trace_run_id(path), trace_cycle_uid(path)) not in valid_trace_keys:
            stale_trace_count += 1
            continue
        scene, variant = trace_context(path)
        try:
            with np.load(path, allow_pickle=False) as trace:
                raw_path = np.asarray(trace["raw_path_xy"], dtype=np.float64)
                selected = np.asarray(trace["selected_candidate_xy"], dtype=np.float64)
                robot = np.asarray(trace["robot_state"], dtype=np.float64)
                goal = np.asarray(trace["goal"], dtype=np.float64)
                topk = np.asarray(trace["topk_candidates_xy"], dtype=np.float64)
                for role, candidate in (("raw_top1", raw_path), ("selected", selected)):
                    metrics = path_metrics(candidate)
                    metrics.update({
                        "scene": scene,
                        "variant": variant,
                        "path_role": role,
                        "trace_file": str(path.relative_to(ROOT)),
                        "start_gap_m": float(np.linalg.norm(_xy(candidate)[0] - robot[:2])) if len(_xy(candidate)) else math.nan,
                        "endpoint_goal_distance_m": float(np.linalg.norm(_xy(candidate)[-1] - goal[:2])) if len(_xy(candidate)) else math.nan,
                    })
                    geometry_rows.append(metrics)
                for candidate in topk:
                    metrics = path_metrics(candidate)
                    metrics.update({
                        "scene": scene,
                        "variant": variant,
                        "path_role": "topk_all",
                        "trace_file": str(path.relative_to(ROOT)),
                        "start_gap_m": float(np.linalg.norm(_xy(candidate)[0] - robot[:2])) if len(_xy(candidate)) else math.nan,
                        "endpoint_goal_distance_m": float(np.linalg.norm(_xy(candidate)[-1] - goal[:2])) if len(_xy(candidate)) else math.nan,
                    })
                    geometry_rows.append(metrics)
                if "minco_samples" in trace.files:
                    samples = np.asarray(trace["minco_samples"], dtype=np.float64)
                    if samples.ndim == 2 and samples.shape[1] >= 3 and len(samples) >= 2:
                        metrics = deformation_metrics(selected, samples[:, 1:3])
                        metrics.update({
                            "scene": scene,
                            "variant": variant,
                            "trace_file": str(path.relative_to(ROOT)),
                            "guide_length_m": path_metrics(selected)["length_m"],
                            "optimized_length_m": path_metrics(samples[:, 1:3])["length_m"],
                        })
                        deformation_rows.append(metrics)
        except (OSError, ValueError, KeyError) as error:
            print(f"[WARN] skipped trace {path}: {error}")
        if index % 1000 == 0:
            print(f"[Trace] {index}/{len(paths)}")
    return geometry_rows, deformation_rows, stale_trace_count


def summarize_geometry(rows: list[dict]) -> list[dict]:
    output = []
    fields = ["length_m", "directness", "max_turn_deg", "turn_tv_deg", "segment_cv", "start_gap_m", "endpoint_goal_distance_m"]
    for (scene, variant, role), group in sorted(group_rows(rows, "scene", "variant", "path_role").items()):
        row = {"scene": scene, "variant": variant, "path_role": role, "paths": len(group)}
        for field in fields:
            values = [item[field] for item in group]
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_p50"] = quantile(values, 0.5)
            row[f"{field}_p90"] = quantile(values, 0.9)
        row["reversal_rate"] = sum(bool(item["has_reversal"]) for item in group) / len(group)
        row["too_short_rate"] = sum(as_float(item["length_m"]) < 0.2 for item in group) / len(group)
        row["start_gap_over_0_5_rate"] = sum(as_float(item["start_gap_m"]) > 0.5 for item in group) / len(group)
        output.append(row)
    return output


def summarize_deformation(rows: list[dict]) -> list[dict]:
    output = []
    fields = ["corresponding_mean_m", "corresponding_p95_m", "corresponding_max_m", "hausdorff_m", "length_ratio", "endpoint_shift_m"]
    for (scene, variant), group in sorted(group_rows(rows, "scene", "variant").items()):
        row = {"scene": scene, "variant": variant, "successful_traces": len(group)}
        for field in fields:
            values = [item[field] for item in group]
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_p50"] = quantile(values, 0.5)
            row[f"{field}_p90"] = quantile(values, 0.9)
        row["max_shift_le_0_05_rate"] = sum(item["corresponding_max_m"] <= 0.05 for item in group) / len(group)
        row["max_shift_le_0_10_rate"] = sum(item["corresponding_max_m"] <= 0.10 for item in group) / len(group)
        row["hausdorff_over_0_20_rate"] = sum(item["hausdorff_m"] > 0.20 for item in group) / len(group)
        output.append(row)
    return output


def data_quality_summary(episodes, cycles, plans, candidates, controls, trace_count, stale_trace_count) -> list[dict]:
    checks = []
    def coverage(dataset, rows, field, severity, impact):
        finite = sum(math.isfinite(as_float(row.get(field))) for row in rows)
        checks.append({
            "dataset": dataset, "field_or_check": field, "rows": len(rows), "valid_rows": finite,
            "coverage_rate": finite / len(rows) if rows else math.nan,
            "severity": severity, "analytical_impact": impact,
        })
    coverage("episode_metrics", episodes, "minimum_executed_clearance_m", "HIGH", "Cannot measure executed safety margin")
    for field in ("raw_min_clearance_m", "raw_path_length_m", "raw_curvature_tv_1pm", "minco_min_clearance_m", "minco_path_length_m"):
        coverage("plan_metrics", plans, field, "HIGH", "Cannot compare candidate and optimized geometry from CSV")
    for field in ("path_length_m", "min_clearance_m", "unsafe_ratio", "curvature_tv_1pm"):
        coverage("candidate_metrics", candidates, field, "HIGH", "Candidate ranking quality is not auditable")
    coverage("planning_cycles", cycles, "validation_ms", "MEDIUM", "Validation timing is unavailable")
    coverage("planning_cycles", cycles, "plan_age_when_applied_ms", "MEDIUM", "Plan freshness is unavailable")
    coverage("control_samples", controls, "planned_v_mps", "LOW", "Expected-motion zero diagnosis is partially covered")
    unknown = sum((row.get("done_reason") or "") == "UNKNOWN" for row in episodes)
    checks.append({
        "dataset": "episode_metrics", "field_or_check": "known done_reason", "rows": len(episodes),
        "valid_rows": len(episodes) - unknown, "coverage_rate": (len(episodes) - unknown) / len(episodes),
        "severity": "HIGH", "analytical_impact": "Failure outcomes cannot be separated into timeout/collision/stuck",
    })
    failed = [row for row in cycles if as_bool(row.get("published")) is False and as_bool(row.get("stale")) is not True and row.get("variant") != "raw"]
    checks.append({
        "dataset": "planning_traces", "field_or_check": "failed-cycle trace coverage", "rows": len(failed),
        "valid_rows": 0, "coverage_rate": 0.0, "severity": "CRITICAL",
        "analytical_impact": "Cannot reconstruct exact geometry/ESDF of the dashed-only rejection cycles",
    })
    checks.append({
        "dataset": "planning_traces", "field_or_check": "saved trace files", "rows": trace_count,
        "valid_rows": trace_count, "coverage_rate": 1.0, "severity": "INFO",
        "analytical_impact": "Published-path geometry can be reconstructed",
    })
    checks.append({
        "dataset": "planning_traces", "field_or_check": "stale files excluded by current plan UID",
        "rows": trace_count + stale_trace_count, "valid_rows": trace_count,
        "coverage_rate": trace_count / (trace_count + stale_trace_count), "severity": "HIGH",
        "analytical_impact": "Resume/retry leaves old trace files; unfiltered geometry analysis is biased",
    })
    return checks


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB7BE")
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close(fig)


def plot_success_rates(run_summary, plot_dir: Path):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(SCENES))
    width = 0.23
    for i, variant in enumerate(VARIANTS):
        values = [next(row["success_rate"] for row in run_summary if row["scene"] == scene and row["variant"] == variant) for scene in SCENES]
        bars = ax.bar(x + (i - 1) * width, values, width, color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant], edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.025, f"{int(round(value*10))}/10", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, SCENES)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Episode success rate")
    ax.set_title("Episode success by scene and controller", loc="left", weight="bold", color=INK, pad=28)
    ax.text(0, 1.01, "10 matched episodes per controller in each scene", transform=ax.transAxes, color="#607D8B", fontsize=10)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, plot_dir / "01_success_rates.png")


def plot_episode_matrix(episodes, plot_dir: Path):
    lookup = {(row["scene_label"], row["episode_uid"], row["variant"]): as_bool(row["success"]) is True for row in episodes}
    keys = []
    for scene in SCENES:
        scene_rows = sorted([row for row in episodes if row["scene_label"] == scene and row["variant"] == "raw"], key=lambda row: int(as_float(row["episode_index"])))
        keys.extend((scene, row["episode_uid"], int(as_float(row["episode_index"]))) for row in scene_rows)
    matrix = np.array([[1 if lookup.get((scene, uid, variant), False) else 0 for variant in VARIANTS] for scene, uid, _ in keys])
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(["#F1B6A8", "#7CC8B6"]), vmin=0, vmax=1)
    ax.set_xticks(range(3), [VARIANT_LABELS[v] for v in VARIANTS])
    labels = [f"{scene} · ep {idx+1:02d}" for scene, _, idx in keys]
    ax.set_yticks(range(len(keys)), labels, fontsize=9)
    for y in range(len(keys)):
        for x in range(3):
            ax.text(x, y, "PASS" if matrix[y, x] else "FAIL", ha="center", va="center", fontsize=8, color="#17342E" if matrix[y,x] else "#6E2D22", weight="bold")
    ax.axhline(9.5, color="white", linewidth=4)
    ax.set_title("Matched episode outcome matrix", loc="left", weight="bold", color=INK, pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, plot_dir / "02_episode_outcome_matrix.png")


def plot_planning_outcomes(outcomes, plot_dir: Path):
    rows = sorted(outcomes, key=lambda row: (SCENES.index(row["scene"]), VARIANTS.index(row["variant"])))
    labels = [f"{row['scene']} · {VARIANT_LABELS[row['variant']]}" for row in rows]
    categories = [("published", "Published", "#2A9D8F"), ("hold", "Hold last", "#E9C46A"), ("stop", "Stop", "#E76F51"), ("stale", "Stale", "#AAB7BE")]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    left = np.zeros(len(rows))
    y = np.arange(len(rows))
    for key, label, color in categories:
        values = np.array([row[f"{key}_rate"] for row in rows])
        ax.barh(y, values, left=left, color=color, edgecolor="white", label=label)
        for yi, lft, value in zip(y, left, values):
            if value >= 0.06:
                ax.text(lft + value/2, yi, f"{value*100:.0f}%", ha="center", va="center", fontsize=9, color=INK)
        left += values
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.invert_yaxis()
    ax.set_xlabel("Share of planning cycles")
    ax.set_title("MINCO planning-cycle outcomes", loc="left", weight="bold", color=INK, pad=28)
    ax.text(0, 1.02, "Failed cycles become HOLD_LAST while cache is usable, otherwise STOP", transform=ax.transAxes, color="#607D8B", fontsize=10)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    style_axis(ax)
    ax.grid(axis="x", color=GRID, alpha=.7); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, plot_dir / "03_planning_outcomes.png")


def plot_failure_mix(failures, plot_dir: Path):
    groups = sorted({(row["scene"], row["variant"]) for row in failures}, key=lambda item: (SCENES.index(item[0]), VARIANTS.index(item[1])))
    category_totals = Counter()
    for row in failures: category_totals[row["failure_category"]] += row["candidate_attempts"]
    categories = [name for name, _ in category_totals.most_common()]
    colors = ["#D1495B", "#355070", "#F4A261", "#7A5195", "#2A9D8F", "#E9C46A", "#6C8EAD", "#B0BEC5"]
    fig, ax = plt.subplots(figsize=(11, 6.0))
    y = np.arange(len(groups)); left = np.zeros(len(groups))
    for category, color in zip(categories, colors):
        values = []
        for scene, variant in groups:
            row = next((item for item in failures if item["scene"] == scene and item["variant"] == variant and item["failure_category"] == category), None)
            values.append(row["candidate_attempt_share"] if row else 0.0)
        values = np.array(values)
        ax.barh(y, values, left=left, color=color, edgecolor="white", label=category)
        left += values
    ax.set_yticks(y, [f"{scene} · {VARIANT_LABELS[variant]}" for scene, variant in groups])
    ax.set_xlim(0, 1); ax.invert_yaxis()
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(v*100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Share of candidate attempts in failed cycles")
    ax.set_title("Why all candidate attempts were rejected", loc="left", weight="bold", color=INK, pad=28)
    ax.text(0, 1.02, "Each failed cycle normally contributes four attempted candidates", transform=ax.transAxes, color="#607D8B", fontsize=10)
    ax.legend(ncol=2, frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    style_axis(ax); ax.grid(axis="x", color=GRID, alpha=.7); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, plot_dir / "04_candidate_failure_mix.png")


def plot_path_quality(geometry_rows, plot_dir: Path):
    selected = [row for row in geometry_rows if row["path_role"] == "selected"]
    groups = [(scene, variant) for scene in SCENES for variant in VARIANTS]
    labels = [f"{scene}\n{VARIANT_LABELS[variant]}" for scene, variant in groups]
    specs = [
        ("length_m", "Candidate path length (m)"),
        ("directness", "Detour ratio (length / displacement)"),
        ("max_turn_deg", "Maximum local turn (degrees)"),
        ("start_gap_m", "Robot-to-path start gap (m)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    for ax, (field, title) in zip(axes.flat, specs):
        data = [[row[field] for row in selected if row["scene"] == scene and row["variant"] == variant and math.isfinite(as_float(row[field]))] for scene, variant in groups]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.62, medianprops={"color": INK, "linewidth": 1.5})
        for patch, (_, variant) in zip(bp["boxes"], groups):
            patch.set_facecolor(VARIANT_COLORS[variant]); patch.set_alpha(.78); patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(labels)+1), labels, fontsize=8)
        ax.set_title(title, loc="left", fontsize=11, weight="bold", color=INK)
        style_axis(ax)
    fig.suptitle("Observed NavDP path geometry at published planning states", x=.07, ha="left", fontsize=15, weight="bold", color=INK)
    fig.text(.07, .935, "Boxplots suppress outliers; these are repeated planning states, not independent episodes", color="#607D8B", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    save_figure(fig, plot_dir / "05_navdp_path_quality.png")


def plot_deformation(deformation_rows, plot_dir: Path):
    groups = [(scene, variant) for scene in SCENES for variant in VARIANTS if variant != "raw"]
    labels = [f"{scene}\n{VARIANT_LABELS[variant]}" for scene, variant in groups]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    specs = [("corresponding_max_m", "Maximum corresponding displacement (m)"), ("hausdorff_m", "Geometric Hausdorff distance (m)")]
    for ax, (field, title) in zip(axes, specs):
        data = [[row[field] for row in deformation_rows if row["scene"] == scene and row["variant"] == variant] for scene, variant in groups]
        parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=.78)
        for body, (_, variant) in zip(parts["bodies"], groups):
            body.set_facecolor(VARIANT_COLORS[variant]); body.set_edgecolor("white"); body.set_alpha(.78)
        parts["cmedians"].set_color(INK); parts["cmedians"].set_linewidth(2)
        ax.axhline(.20, color="#7A5195", linestyle="--", linewidth=1.2, label="0.20 m reference")
        ax.set_xticks(range(1, len(labels)+1), labels, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, weight="bold", color=INK)
        style_axis(ax)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("How far successful MINCO trajectories move away from NavDP guides", x=.07, ha="left", fontsize=15, weight="bold", color=INK)
    fig.text(.07, .91, "Successful published traces only; failed dashed-only cycles were not saved", color="#607D8B", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save_figure(fig, plot_dir / "06_minco_deformation.png")


def plot_control(control_summary, plot_dir: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    groups = [(scene, variant) for scene in SCENES for variant in VARIANTS]
    values = [next(row["expected_motion_zero_rate"] for row in control_summary if row["scene"] == scene and row["variant"] == variant) for scene, variant in groups]
    bars = ax.bar(range(len(groups)), values, color=[VARIANT_COLORS[variant] for _, variant in groups], edgecolor="white")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.002, f"{value*100:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(groups)), [f"{scene}\n{VARIANT_LABELS[variant]}" for scene, variant in groups], fontsize=9)
    ax.set_ylabel("Expected-motion zero rate")
    ax.set_ylim(0.0, max(values + [0.01]) * 1.24)
    ax.set_title("MPC outputs zero while planned speed is positive", loc="left", weight="bold", color=INK, pad=28)
    ax.text(0, 1.02, "CONTROL_ACTIVE samples with planned_v > 0.05 m/s and |cmd_v| ≤ 0.01 m/s", transform=ax.transAxes, color="#607D8B", fontsize=10)
    style_axis(ax); fig.tight_layout()
    save_figure(fig, plot_dir / "07_mpc_expected_motion_zero.png")


def make_video_evidence(suite: Path, plot_dir: Path):
    try:
        import cv2
    except ImportError:
        return
    cases = [
        ("DENSE", "minco-cold", "ep_ef5f54620f7aa451", 15, "DENSE: dashed candidates, MINCO_STOP"),
        ("SPARSE", "minco-cold", "ep_57374a7ba676a4f0", 15, "SPARSE: dashed candidates, MINCO_STOP"),
        ("DENSE", "minco-cold", "ep_aee40f2146b3323a", 107, "DENSE: accepted MINCO trajectory"),
    ]
    frames = []
    for scene, variant, uid, frame_idx, title in cases:
        candidates = list(suite.glob(f"experiments/**/{scene}/{variant}/**/videos/{uid}.mp4"))
        if not candidates:
            continue
        capture = cv2.VideoCapture(str(candidates[0])); capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read(); capture.release()
        if ok:
            frames.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), title))
    if not frames:
        return
    fig, axes = plt.subplots(len(frames), 1, figsize=(14, 4.2 * len(frames)))
    if len(frames) == 1: axes = [axes]
    for ax, (frame, title) in zip(axes, frames):
        ax.imshow(frame); ax.set_title(title, loc="left", fontsize=12, weight="bold", color=INK); ax.axis("off")
    fig.suptitle("Representative video evidence", x=.02, ha="left", fontsize=16, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, .98))
    save_figure(fig, plot_dir / "08_video_evidence.png")


def run_analysis(suite: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plot_dir = out / "plots"; table_dir = out / "tables"
    plot_dir.mkdir(exist_ok=True); table_dir.mkdir(exist_ok=True)
    episodes = read_all(suite, "episode_metrics.csv")
    cycles = read_all(suite, "planning_cycles.csv")
    plans = read_all(suite, "plan_metrics.csv")
    candidates = read_all(suite, "candidate_metrics.csv")
    controls = read_all(suite, "control_samples.csv")
    print(f"[Rows] episodes={len(episodes)} cycles={len(cycles)} plans={len(plans)} candidates={len(candidates)} controls={len(controls)}")

    run_summary = summarize_runs(episodes, plans)
    planning_outcomes, failure_summary, cycle_flags = summarize_planning(cycles)
    control_summary = summarize_controls(controls)
    valid_trace_keys = {
        (row.get("run_id", ""), row.get("plan_uid", "").removesuffix("_plan"))
        for row in plans if row.get("run_id") and row.get("plan_uid")
    }
    geometry_rows, deformation_rows, stale_trace_count = collect_trace_metrics(suite, valid_trace_keys)
    geometry_summary = summarize_geometry(geometry_rows)
    deformation_summary = summarize_deformation(deformation_rows)
    available_trace_count = len(list(suite.glob("experiments/**/traces/*.npz")))
    trace_count = available_trace_count - stale_trace_count
    quality = data_quality_summary(episodes, cycles, plans, candidates, controls, trace_count, stale_trace_count)

    write_csv(table_dir / "run_summary.csv", run_summary)
    write_csv(table_dir / "planning_outcome_summary.csv", planning_outcomes)
    write_csv(table_dir / "candidate_failure_summary.csv", failure_summary)
    write_csv(table_dir / "failed_cycle_flags.csv", cycle_flags)
    write_csv(table_dir / "control_summary.csv", control_summary)
    write_csv(table_dir / "navdp_path_quality_summary.csv", geometry_summary)
    write_csv(table_dir / "minco_deformation_summary.csv", deformation_summary)
    write_csv(table_dir / "data_quality_summary.csv", quality)
    write_csv(table_dir / "episode_detail.csv", episodes, [field for field in episodes[0] if not field.startswith("_")])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "#FAFBFC", "axes.facecolor": "#FAFBFC"})
    plot_success_rates(run_summary, plot_dir)
    plot_episode_matrix(episodes, plot_dir)
    plot_planning_outcomes(planning_outcomes, plot_dir)
    plot_failure_mix(failure_summary, plot_dir)
    plot_path_quality(geometry_rows, plot_dir)
    plot_deformation(deformation_rows, plot_dir)
    plot_control(control_summary, plot_dir)
    make_video_evidence(suite, plot_dir)

    manifest = {
        "suite": str(suite.relative_to(ROOT)),
        "output": str(out.relative_to(ROOT)),
        "counts": {
            "episodes": len(episodes), "planning_cycles": len(cycles), "plan_metrics": len(plans),
            "candidate_metrics": len(candidates), "control_samples": len(controls),
            "trace_files_available": available_trace_count, "trace_files_current": trace_count,
            "trace_files_stale_excluded": stale_trace_count,
            "geometry_rows": len(geometry_rows), "deformation_rows": len(deformation_rows),
        },
        "definitions": {
            "expected_motion_zero": "CONTROL_ACTIVE and planned_v_mps > 0.05 and abs(cmd_v_mps) <= 0.01",
            "candidate_failure_share": "candidate attempts within non-stale, unpublished MINCO cycles",
            "deformation": "arc-length-normalized geometric comparison between selected NavDP candidate and published MINCO sample positions",
        },
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_analysis(args.suite.resolve(), args.output.resolve())
