"""Comparison analysis for capability sweep — both profiles on every chart."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROFILES = ("legacy", "superplanner_sfc_v1")


def _read_sweep_rows(sweep_csv: Path) -> list[dict[str, str]]:
    with sweep_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: str | None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def generate_sweep_comparison(sweep_dir: Path | str) -> dict[str, Any]:
    """Generate comparison tables and charts for a capability sweep.

    Outputs (all under sweep_dir/comparison):
      - paired_deltas.csv  — per-case legacy vs safe metric deltas
      - profile_aggregates.csv — per-profile summary statistics
      - factor_margins_by_profile.png — factor curves with BOTH profiles
      - margin_distribution.png — margin distribution comparison
    """
    sweep_dir = Path(sweep_dir).resolve()
    rows = _read_sweep_rows(sweep_dir / "sweep_runs.csv")
    out_dir = sweep_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_key: dict[tuple[str, str], dict[str, str]] = {
        (row["case_uid"], row["profile"]): row for row in rows
    }
    case_uids = sorted({row["case_uid"] for row in rows})

    # ---- Paired deltas ----
    metric_names = [
        "min_clearance_m", "clearance_p05_m", "unsafe_ratio",
        "path_length_m", "path_length_ratio", "endpoint_error_m",
        "guide_deviation_mean_m", "guide_deviation_p95_m",
        "velocity_violation_ratio", "acceleration_violation_ratio",
        "jerk_violation_ratio", "yaw_rate_violation_ratio",
        "backtracking_ratio", "self_intersection_count",
    ]
    delta_rows: list[dict[str, Any]] = []
    for uid in case_uids:
        legacy = by_key.get((uid, "legacy"))
        safe = by_key.get((uid, "superplanner_sfc_v1"))
        if legacy is None or safe is None:
            continue
        row: dict[str, Any] = {
            "case_uid": uid,
            "expected_category": legacy.get("expected_category", ""),
            "legacy_status": legacy.get("status", ""),
            "safe_status": safe.get("status", ""),
            "legacy_failure_reason": legacy.get("failure_reason", ""),
            "safe_failure_reason": safe.get("failure_reason", ""),
        }
        for metric in metric_names:
            lv = _num(legacy.get(metric))
            sv = _num(safe.get(metric))
            if not math.isnan(lv) and not math.isnan(sv):
                row[f"{metric}__legacy"] = lv
                row[f"{metric}__safe"] = sv
                row[f"{metric}__delta"] = sv - lv
        delta_rows.append(row)

    fields = sorted({k for row in delta_rows for k in row})
    with (out_dir / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(delta_rows)

    # ---- Profile aggregates ----
    agg_metrics = [
        "min_clearance_m", "unsafe_ratio", "path_length_ratio",
        "guide_deviation_p95_m", "yaw_rate_violation_ratio",
    ]
    agg_rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        prof_rows = [row for row in rows if row["profile"] == profile]
        summary: dict[str, Any] = {"profile": profile, "n": len(prof_rows)}
        succeeded = [row for row in prof_rows if row["status"] == "SUCCEEDED"]
        summary["success_rate"] = len(succeeded) / max(1, len(prof_rows))
        for metric in agg_metrics:
            values = [_num(row.get(metric)) for row in succeeded]
            values = [v for v in values if not math.isnan(v)]
            if values:
                arr = np.asarray(values)
                summary[f"{metric}__mean"] = float(np.mean(arr))
                summary[f"{metric}__median"] = float(np.median(arr))
                summary[f"{metric}__p95"] = float(np.percentile(arr, 95))
                summary[f"{metric}__min"] = float(np.min(arr))
                summary[f"{metric}__max"] = float(np.max(arr))
        agg_rows.append(summary)

    agg_fields = sorted({k for row in agg_rows for k in row})
    with (out_dir / "profile_aggregates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg_rows)

    # ---- Charts ----
    try:
        _render_factor_charts(rows, out_dir)
    except Exception:
        pass

    # ---- Summary JSON ----
    summary = {
        "schema_version": 1,
        "paired_cases": len(delta_rows),
        "profiles": agg_rows,
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _render_factor_charts(rows: list[dict[str, str]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factor_rows = [
        row for row in rows
        if row.get("factor_name") and row.get("factor_name") not in {"", "geometry_category"}
    ]
    if not factor_rows:
        return

    factors = sorted({row["factor_name"] for row in factor_rows})
    # One figure per factor: both profiles on same axes
    for factor in factors:
        subset = [row for row in factor_rows if row["factor_name"] == factor]
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for profile, style in (("legacy", {"ls": "--", "marker": "o"}),
                               ("superplanner_sfc_v1", {"ls": "-", "marker": "s"})):
            prof_rows = [row for row in subset if row["profile"] == profile]
            if not prof_rows:
                continue
            prof_rows = sorted(prof_rows, key=lambda r: _num(r.get("factor_level")))
            levels = [_num(row.get("factor_level")) for row in prof_rows]
            margins = [
                _num(row.get("min_normalized_margin")) if row["status"] == "SUCCEEDED"
                else 0.0
                for row in prof_rows
            ]
            ax.plot(levels, margins, label=f"{profile} (n={len(prof_rows)})",
                    **style)
        ax.axhline(0.0, color="red", linewidth=0.8, linestyle=":")
        ax.set_xlabel(factor)
        ax.set_ylabel("min normalized margin (0 = fail)")
        ax.set_title(f"{factor}: legacy vs superplanner_sfc_v1")
        ax.legend()
        fig.savefig(out_dir / f"factor_{factor}_both_profiles.png", dpi=150,
                    facecolor="white")
        plt.close(fig)

    # Margin distribution comparison
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for profile, color in (("legacy", "#D55E00"), ("superplanner_sfc_v1", "#0072B2")):
        values = [
            _num(row.get("min_normalized_margin"))
            for row in rows
            if row["profile"] == profile and row["status"] == "SUCCEEDED"
        ]
        values = [v for v in values if not math.isnan(v)]
        ax.hist(values, bins=12, alpha=0.45, label=f"{profile} (n={len(values)})",
                color=color)
    ax.set_xlabel("min normalized margin")
    ax.set_ylabel("count")
    ax.set_title("Margin distribution by profile (successful cases)")
    ax.legend()
    fig.savefig(out_dir / "margin_distribution.png", dpi=150, facecolor="white")
    plt.close(fig)


__all__ = ["generate_sweep_comparison"]
