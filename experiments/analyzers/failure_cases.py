from __future__ import annotations

import csv
from pathlib import Path


def generate_failure_report(suite_dir, episodes):
    reports = Path(suite_dir) / "reports"; failures = [row for row in episodes if str(row.get("success", "")).lower() != "true"]
    fields = ["episode_uid", "primary_reason", "scene_label", "variant", "data_source"]
    with (reports / "failure_case_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in failures: writer.writerow({"episode_uid":row.get("episode_uid", ""), "primary_reason":row.get("done_reason", "UNKNOWN"), "scene_label":row.get("scene_label", ""), "variant":row.get("variant", ""), "data_source":row.get("data_source", "UNKNOWN")})
    sources = {row.get("data_source", "UNKNOWN") for row in episodes}
    source = next(iter(sources)) if len(sources) == 1 else "MIXED"
    warning = "> SIMULATED DATA\n\n" if source == "SIMULATED" else ""
    text = f"# Failure Case Report\n\n{warning}Data source: {source}.\n\n" + ("\n".join(f"- {row.get('episode_uid')}: {row.get('done_reason')}" for row in failures) or "No failures recorded.") + "\n"
    (reports / "failure_case_report.md").write_text(text, encoding="utf-8")
    return failures
