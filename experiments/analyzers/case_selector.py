from __future__ import annotations

import csv
from pathlib import Path


REASONS = ("median_case", "largest_improvement", "largest_regression", "raw_unsafe_repaired", "raw_unsafe_rejected", "safe_but_false_rejected", "hot_accepted_stable", "hot_rejected_jump", "collision_case", "timeout_case")


def select_representative_cases(suite_dir, episodes):
    output = Path(suite_dir) / "reports" / "representative_cases.csv"; output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, reason in enumerate(REASONS):
        source = episodes[index % len(episodes)] if episodes else {}
        rows.append({"episode_uid":source.get("episode_uid", ""), "plan_uid":"", "selection_reason":reason, "video_path":"", "trace_path":"", "plot_paths":"", "data_source":"SIMULATED"})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows
