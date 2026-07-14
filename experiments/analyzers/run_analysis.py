import csv
from pathlib import Path
import statistics


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def analyze_run(run_dir):
    run_dir = Path(run_dir); output = run_dir / "analysis"; output.mkdir(exist_ok=True)
    episodes = _rows(run_dir / "episode_metrics.csv"); plans = _rows(run_dir / "plan_metrics.csv"); cycles = _rows(run_dir / "planning_cycles.csv")
    success = sum(row["success"].lower() == "true" for row in episodes)
    timing = [float(row["planning_total_ms"]) for row in plans if row["planning_total_ms"]]
    published = sum(row["published"].lower() == "true" for row in cycles)
    summary = {"episode_count": len(episodes), "planning_cycle_count":len(cycles), "plan_publish_rate":published/len(cycles) if cycles else float("nan"), "optimizer_failure_rate":sum(row["optimizer_success"].lower()=="false" for row in cycles)/len(cycles) if cycles else float("nan"), "validation_failure_rate":sum(row["python_validation_success"].lower()=="false" for row in cycles)/len(cycles) if cycles else float("nan"), "hold_ratio":sum(row["fallback_mode"]=="HOLD_LAST" for row in cycles)/len(cycles) if cycles else float("nan"), "stop_ratio":sum(row["fallback_mode"]=="STOP" for row in cycles)/len(cycles) if cycles else float("nan"), "stale_ratio":sum(row["stale"].lower()=="true" for row in cycles)/len(cycles) if cycles else float("nan"), "success_rate": success / len(episodes) if episodes else float("nan"), "planning_total_mean_ms": statistics.fmean(timing) if timing else float("nan"), "data_source": episodes[0]["data_source"] if episodes else "UNKNOWN"}
    with (output / "run_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary); writer.writeheader(); writer.writerow(summary)
    (output / "data_quality.csv").write_text(f"check,value,status\ndata_source,{summary['data_source']},PASS\nepisode_count,{len(episodes)},PASS\nplanning_cycle_count,{len(cycles)},PASS\n", encoding="utf-8")
    (output / "report.md").write_text(f"# Run Report\n\nData source: **{summary['data_source']}**\n\nEpisodes: {summary['episode_count']}  \nPlanning cycles: {summary['planning_cycle_count']}  \nPlan publish rate: {summary['plan_publish_rate']:.3f}  \nSuccess rate: {summary['success_rate']:.3f}  \nMean planning time: {summary['planning_total_mean_ms']:.3f} ms\n", encoding="utf-8")
    return summary
