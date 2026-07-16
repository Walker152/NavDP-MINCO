#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "results" / "navdp_minco_full_real"
OUT = ROOT / "codex_ana"


def read_all(name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(SUITE.glob(f"experiments/*/*/*/*/*/{name}")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["_source_file"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


def f(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def b(row: dict[str, str], key: str) -> bool | None:
    value = str(row.get(key, "")).strip().lower()
    return True if value == "true" else False if value == "false" else None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cycles = read_all("planning_cycles.csv")
    controls = read_all("control_samples.csv")
    episodes = read_all("episode_metrics.csv")
    plans = read_all("plan_metrics.csv")

    cycle_rows = []
    candidate_failure_rows = []
    for variant in sorted({r["variant"] for r in cycles}):
        subset = [r for r in cycles if r["variant"] == variant]
        reasons = Counter((r.get("failure_reason") or "<empty>") for r in subset if b(r, "published") is False)
        candidate_reasons = Counter()
        for row in subset:
            if b(row, "published") is not False or row.get("failure_reason") == "STALE_RESULT":
                continue
            for item in row.get("failure_reason", "").split("; "):
                reason = item.split(": ", 1)[1] if ": " in item else item
                if reason.startswith("PY_ESDF_UNSAFE"):
                    reason = "PY_ESDF_UNSAFE"
                candidate_reasons[reason or "<empty>"] += 1
        for reason, count in sorted(candidate_reasons.items()):
            candidate_failure_rows.append({
                "variant": variant,
                "failure_category": reason,
                "candidate_attempts": count,
            })
        cycle_rows.append({
            "variant": variant,
            "cycles": len(subset),
            "published": sum(b(r, "published") is True for r in subset),
            "not_published": sum(b(r, "published") is False for r in subset),
            "stale": sum(b(r, "stale") is True for r in subset),
            "optimizer_false": sum(b(r, "optimizer_success") is False for r in subset),
            "validation_false": sum(b(r, "python_validation_success") is False for r in subset),
            "stop": sum(r.get("fallback_mode") == "STOP" for r in subset),
            "hold_last": sum(r.get("fallback_mode") == "HOLD_LAST" for r in subset),
            "top_failure_reasons": json.dumps(reasons.most_common(), ensure_ascii=False),
        })
    write_csv(
        OUT / "planning_failure_summary.csv",
        cycle_rows,
        ["variant", "cycles", "published", "not_published", "stale", "optimizer_false",
         "validation_false", "stop", "hold_last", "top_failure_reasons"],
    )
    write_csv(
        OUT / "candidate_failure_categories.csv",
        candidate_failure_rows,
        ["variant", "failure_category", "candidate_attempts"],
    )

    control_rows = []
    zero_eps = defaultdict(set)
    for variant in sorted({r["variant"] for r in controls}):
        subset = [r for r in controls if r["variant"] == variant]
        states = Counter(r.get("control_state") or "<empty>" for r in subset)
        zeros = [r for r in subset if abs(f(r, "cmd_v_mps")) <= 1e-6 and abs(f(r, "cmd_w_radps")) <= 1e-6]
        active = [r for r in subset if r.get("control_state") == "CONTROL_ACTIVE"]
        active_zero = [r for r in active if abs(f(r, "cmd_v_mps")) <= 1e-6 and abs(f(r, "cmd_w_radps")) <= 1e-6]
        active_v_zero = [r for r in active if abs(f(r, "cmd_v_mps")) <= 1e-4]
        for row in active_zero:
            zero_eps[variant].add(row["episode_uid"])
        control_rows.append({
            "variant": variant,
            "samples": len(subset),
            "zero_cmd_samples": len(zeros),
            "zero_cmd_rate": len(zeros) / len(subset) if subset else math.nan,
            "active_samples": len(active),
            "active_zero_both": len(active_zero),
            "active_zero_both_rate": len(active_zero) / len(active) if active else math.nan,
            "active_zero_v": len(active_v_zero),
            "active_zero_v_rate": len(active_v_zero) / len(active) if active else math.nan,
            "control_states": json.dumps(states.most_common(), ensure_ascii=False),
        })
    write_csv(
        OUT / "control_zero_summary.csv",
        control_rows,
        ["variant", "samples", "zero_cmd_samples", "zero_cmd_rate", "active_samples",
         "active_zero_both", "active_zero_both_rate", "active_zero_v", "active_zero_v_rate",
         "control_states"],
    )

    expected_motion_rows = []
    for variant in sorted({r["variant"] for r in controls}):
        subset = [r for r in controls if r["variant"] == variant]
        observable = [
            r for r in subset
            if math.isfinite(f(r, "planned_v_mps")) and math.isfinite(f(r, "cmd_v_mps"))
        ]
        mismatches = [
            r for r in observable
            if f(r, "planned_v_mps") > 0.05 and abs(f(r, "cmd_v_mps")) <= 0.01
        ]
        stalls = [r for r in subset if r.get("zero_command_reason") == "EXPECTED_MOTION_ZERO_STALL"]
        expected_motion_rows.append({
            "variant": variant,
            "samples": len(subset),
            "planned_velocity_observable_samples": len(observable),
            "planned_velocity_coverage": len(observable) / len(subset) if subset else math.nan,
            "expected_motion_zero_samples": len(mismatches),
            "expected_motion_zero_rate": len(mismatches) / len(observable) if observable else math.nan,
            "detected_stall_samples": len(stalls),
            "episodes_with_expected_motion_zero": len({r["episode_uid"] for r in mismatches}),
            "evidence_status": "MEASURED" if observable else "LEGACY_DATA_MISSING_PLANNED_V",
        })
    write_csv(
        OUT / "expected_motion_zero_summary.csv",
        expected_motion_rows,
        ["variant", "samples", "planned_velocity_observable_samples", "planned_velocity_coverage",
         "expected_motion_zero_samples", "expected_motion_zero_rate", "detected_stall_samples",
         "episodes_with_expected_motion_zero", "evidence_status"],
    )

    active_zero_cases = []
    grouped = defaultdict(list)
    for row in controls:
        grouped[(row["variant"], row["episode_uid"])].append(row)
    for (variant, episode_uid), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: int(float(r["frame_idx"])))
        active = [r for r in rows if r.get("control_state") == "CONTROL_ACTIVE"]
        active_zero = [r for r in active if abs(f(r, "cmd_v_mps")) <= 1e-6 and abs(f(r, "cmd_w_radps")) <= 1e-6]
        if active_zero:
            active_zero_cases.append({
                "variant": variant,
                "episode_uid": episode_uid,
                "active_samples": len(active),
                "active_zero_samples": len(active_zero),
                "first_zero_frame": min(int(float(r["frame_idx"])) for r in active_zero),
                "last_zero_frame": max(int(float(r["frame_idx"])) for r in active_zero),
                "max_cross_track_error_m": max((f(r, "cross_track_error_m") for r in active_zero), default=math.nan),
            })
    write_csv(
        OUT / "active_zero_cases.csv",
        active_zero_cases,
        ["variant", "episode_uid", "active_samples", "active_zero_samples",
         "first_zero_frame", "last_zero_frame", "max_cross_track_error_m"],
    )

    episode_rows = []
    for variant in sorted({r["variant"] for r in episodes}):
        subset = [r for r in episodes if r["variant"] == variant]
        reasons = Counter(r.get("done_reason") or "<empty>" for r in subset)
        episode_rows.append({
            "variant": variant,
            "episodes": len(subset),
            "success": sum(b(r, "success") is True for r in subset),
            "collision": sum(b(r, "collision") is True for r in subset),
            "timeout": sum(b(r, "timeout") is True for r in subset),
            "episodes_with_active_zero": len(zero_eps[variant]),
            "nonpositive_path_length": sum(f(r, "actual_path_length_m") <= 0 for r in subset),
            "done_reasons": json.dumps(reasons.most_common(), ensure_ascii=False),
        })
    write_csv(
        OUT / "episode_outcome_summary.csv",
        episode_rows,
        ["variant", "episodes", "success", "collision", "timeout",
         "episodes_with_active_zero", "nonpositive_path_length", "done_reasons"],
    )

    missing_rows = []
    fields = {
        "plans": [
            "raw_min_clearance_m", "raw_unsafe_ratio", "raw_curvature_tv_1pm",
            "actual_speed_mean_mps", "actual_acc_rms_mps2", "actual_jerk_rms_mps3",
            "actual_yaw_rate_rms_radps", "minco_min_clearance_m",
        ],
        "controls": [
            "cross_track_error_m", "time_aligned_position_error_m",
            "mpc_solve_ms", "reference_age_ms",
        ],
    }
    for dataset, rows in (("plans", plans), ("controls", controls)):
        for field in fields[dataset]:
            valid = sum(math.isfinite(f(r, field)) for r in rows)
            missing_rows.append({
                "dataset": dataset,
                "field": field,
                "rows": len(rows),
                "finite": valid,
                "finite_rate": valid / len(rows) if rows else math.nan,
            })
    write_csv(OUT / "field_coverage.csv", missing_rows, ["dataset", "field", "rows", "finite", "finite_rate"])

    variants = [row["variant"] for row in control_rows]
    zero_rates = [row["zero_cmd_rate"] for row in control_rows]
    active_zero_rates = [row["active_zero_both_rate"] for row in control_rows]
    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.36
    ax.bar(x - width / 2, zero_rates, width, label="All control samples")
    ax.bar(x + width / 2, active_zero_rates, width, label="CONTROL_ACTIVE only")
    ax.set_xticks(x, variants)
    ax.set_ylabel("Zero-command rate")
    ax.set_ylim(0, max(zero_rates + active_zero_rates + [0.1]) * 1.15)
    ax.set_title("Zero velocity command rate by controller variant")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "zero_command_rates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    measured = [row for row in expected_motion_rows if math.isfinite(row["expected_motion_zero_rate"])]
    if measured:
        ax.bar(
            [row["variant"] for row in measured],
            [row["expected_motion_zero_rate"] for row in measured],
            color="#C55A11",
        )
        ax.set_ylabel("planned_v > 0.05 and |cmd_v| <= 0.01 rate")
        ax.set_title("Unexpected MPC zero-speed rate under positive planned velocity")
    else:
        ax.text(
            0.5, 0.5,
            "Legacy result files do not contain planned_v_mps;\n"
            "the requested mismatch cannot be reconstructed reliably.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title("Unexpected MPC zero-speed observability")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "expected_motion_zero_rates.png", dpi=160)
    plt.close(fig)

    fail_counts = [row["not_published"] for row in cycle_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([row["variant"] for row in cycle_rows], fail_counts, color="#4472C4")
    ax.set_ylabel("Planning cycles not published")
    ax.set_title("Planning publication failures by variant")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "planning_failures.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
