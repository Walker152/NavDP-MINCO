import csv
from pathlib import Path
import statistics


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def analyze_run(run_dir):
    run_dir = Path(run_dir); output = run_dir / "analysis"; output.mkdir(exist_ok=True)
    episodes = _rows(run_dir / "episode_metrics.csv"); plans = _rows(run_dir / "plan_metrics.csv")
    success = sum(row["success"].lower() == "true" for row in episodes)
    timing = [float(row["planning_total_ms"]) for row in plans if row["planning_total_ms"]]
    summary = {"episode_count": len(episodes), "success_rate": success / len(episodes) if episodes else float("nan"), "planning_total_mean_ms": statistics.fmean(timing) if timing else float("nan"), "data_source": episodes[0]["data_source"] if episodes else "UNKNOWN"}
    with (output / "run_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary); writer.writeheader(); writer.writerow(summary)
    (output / "report.md").write_text(f"# Run Report\n\nData source: **{summary['data_source']}**\n\nEpisodes: {summary['episode_count']}  \nSuccess rate: {summary['success_rate']:.3f}  \nMean planning time: {summary['planning_total_mean_ms']:.3f} ms\n", encoding="utf-8")
    return summary
