from pathlib import Path


def analyze_suite(suite_dir):
    suite_dir = Path(suite_dir); reports = suite_dir / "reports"; reports.mkdir(parents=True, exist_ok=True)
    runs = sorted(suite_dir.glob("experiments/*/*/*/*/*/analysis/run_summary.csv"))
    text = "# NavDP–MINCO Suite Report\n\n> **SIMULATED — 工具链验收数据，不代表算法科研结论。**\n\n"
    text += f"Validated mock runs: {len(runs)}\n\n"
    text += "Results are grouped by experiment purpose, scene, variant, seed and run ID.\n"
    (reports / "suite_report.md").write_text(text, encoding="utf-8")
    (reports / "plot_index.csv").write_text("experiment_id,plot_path,status\n", encoding="utf-8")
    return reports / "suite_report.md"
