from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


MATCH_FIELDS = ("scene_id", "seed", "safe_dist", "manifest_id")
METRICS = ("episode_duration_s", "actual_path_length_m", "repository_spl")


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def compare_runs(baseline_dir, method_dir, output_dir):
    baseline_dir, method_dir, output_dir = Path(baseline_dir), Path(method_dir), Path(output_dir)
    baseline_config = json.loads((baseline_dir / "run_config.json").read_text()); method_config = json.loads((method_dir / "run_config.json").read_text())
    for field in MATCH_FIELDS:
        if baseline_config.get(field) != method_config.get(field): raise ValueError(f"configuration mismatch: {field}")
    baseline = {row["episode_uid"]: row for row in _read_rows(baseline_dir / "episode_metrics.csv")}; method = {row["episode_uid"]: row for row in _read_rows(method_dir / "episode_metrics.csv")}
    common = sorted(set(baseline) & set(method)); result = {"paired_count": len(common)}
    details = []
    for metric in METRICS:
        deltas = []
        for uid in common:
            try: delta = float(method[uid][metric]) - float(baseline[uid][metric])
            except (KeyError, ValueError): continue
            if math.isfinite(delta): deltas.append(delta); details.append({"episode_uid": uid, "metric": metric, "baseline": baseline[uid][metric], "method": method[uid][metric], "delta": delta})
        result[f"{metric}_delta_mean"] = statistics.fmean(deltas) if deltas else float("nan")
        result[f"{metric}_delta_median"] = statistics.median(deltas) if deltas else float("nan")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_uid", "metric", "baseline", "method", "delta"]); writer.writeheader(); writer.writerows(details)
    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result); writer.writeheader(); writer.writerow(result)
    (output_dir / "report.md").write_text(f"# Paired Run Comparison\n\nPaired episodes: {len(common)}\n\nPositive delta means the method value is larger. No superiority claim is inferred automatically.\n", encoding="utf-8")
    return result
