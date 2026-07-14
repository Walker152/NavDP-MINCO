from __future__ import annotations

import csv
from pathlib import Path


TABLES = ("table_data_quality", "table_raw_profile", "table_safety_repair", "table_smoothness", "table_warm_start", "table_control_navigation", "table_timing")
FIELDS = ["metric", "n", "mean_or_rate", "median", "p95", "ci95_low", "ci95_high", "baseline_delta", "relative_change_percent", "data_source", "method"]


def generate_result_tables(suite_dir, episode_count, plan_count):
    output = Path(suite_dir) / "reports" / "core_tables"; output.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        row = {"metric":name.removeprefix("table_"), "n": plan_count if name not in {"table_control_navigation"} else episode_count, "mean_or_rate":0.5, "median":0.5, "p95":0.9, "ci95_low":0.25, "ci95_high":0.75, "baseline_delta":0.0, "relative_change_percent":0.0, "data_source":"SIMULATED", "method":"descriptive_mock"}
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerow(row)
        (output / f"{name}.md").write_text("| Metric | n | Value | 95% CI | Source |\n|---|---:|---:|---|---|\n" + f"| {row['metric']} | {row['n']} | {row['mean_or_rate']} | [{row['ci95_low']}, {row['ci95_high']}] | SIMULATED |\n", encoding="utf-8")
    return output
