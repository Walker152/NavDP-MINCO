from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics


REASONS = ("median_case", "largest_improvement", "largest_regression", "raw_unsafe_repaired", "raw_unsafe_rejected", "safe_but_false_rejected", "hot_accepted_stable", "hot_rejected_jump", "collision_case", "timeout_case")


def _float(row, field):
    try:
        value = float(row.get(field, "")); return value if math.isfinite(value) else None
    except (TypeError, ValueError): return None


def _paired_delta_cases(episodes):
    grouped = {}
    for row in episodes: grouped.setdefault(row.get("episode_uid"), {})[row.get("variant")] = row
    cases = []
    for uid, variants in grouped.items():
        if "raw" not in variants: continue
        baseline = _float(variants["raw"], "episode_duration_s")
        for method in ("minco-hot", "minco-cold"):
            value = _float(variants.get(method, {}), "episode_duration_s")
            if baseline is not None and value is not None: cases.append((value - baseline, variants[method]))
    return cases


def select_representative_cases(suite_dir, episodes, plans=None, cycles=None):
    plans, cycles = plans or [], cycles or []
    output = Path(suite_dir) / "reports" / "representative_cases.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    selected = {}
    durations = [(value, row) for row in episodes if (value := _float(row, "episode_duration_s")) is not None]
    if durations:
        median = statistics.median(value for value, _ in durations); selected["median_case"] = min(durations, key=lambda item:abs(item[0]-median))[1]
    deltas = _paired_delta_cases(episodes)
    if deltas:
        selected["largest_improvement"] = min(deltas, key=lambda item:item[0])[1]
        selected["largest_regression"] = max(deltas, key=lambda item:item[0])[1]
    selected["collision_case"] = next((row for row in episodes if str(row.get("collision")).lower() == "true"), {})
    selected["timeout_case"] = next((row for row in episodes if str(row.get("timeout")).lower() == "true"), {})
    selected["raw_unsafe_repaired"] = next((row for row in plans if row.get("variant") != "raw" and _float(row,"raw_unsafe_ratio") not in (None,0.0) and (_float(row,"minco_unsafe_ratio") or 0.0) == 0.0), {})
    selected["hot_accepted_stable"] = next((row for row in plans if row.get("variant") == "minco-hot" and str(row.get("hot_start_accepted")).lower() == "true" and row.get("temporal_class") == "STABLE_INPUT"), {})
    selected["hot_rejected_jump"] = next((row for row in plans if row.get("variant") == "minco-hot" and str(row.get("hot_start_accepted")).lower() == "false" and row.get("temporal_class") == "JUMP_INPUT"), {})
    rejected = next((row for row in cycles if row.get("variant") != "raw" and str(row.get("published")).lower() == "false"), {})
    selected["raw_unsafe_rejected"] = rejected
    selected["safe_but_false_rejected"] = next((row for row in cycles if row.get("variant") != "raw" and str(row.get("published")).lower() == "false" and "FALSE_REJECT" in row.get("failure_reason", "")), {})
    rows = []
    for reason in REASONS:
        source = selected.get(reason, {})
        uid = source.get("episode_uid", "")
        videos = sorted(Path(suite_dir).glob(f"experiments/*/*/*/*/*/videos/{uid}.mp4")) if uid else []
        traces = sorted(Path(suite_dir).glob(f"experiments/*/*/*/*/*/traces/*{uid}*.npz")) if uid else []
        rows.append({"episode_uid":uid, "plan_uid":source.get("plan_uid", ""), "selection_reason":reason, "video_path":str(videos[0].relative_to(suite_dir)) if videos else "", "trace_path":str(traces[0].relative_to(suite_dir)) if traces else "", "plot_paths":"", "data_source":source.get("data_source", "")})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows
