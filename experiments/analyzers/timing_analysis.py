from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


PALETTE = {
    "raw": "#4472C4",
    "minco-cold": "#D6A42A",
    "minco-hot": "#8064A2",
}
INK = "#263238"
GRID = "#D9E1E5"
STAGE_FIELDS = (
    "planning_total_ms",
    "navdp_step_ms",
    "candidate_screen_ms",
    "candidate_attempt_total_ms",
    "candidate_cpp_total_ms",
    "python_validation_total_ms",
    "adapter_overhead_ms",
)
MINCO_STAGE_FIELDS = (
    "candidate_screen_ms",
    "candidate_attempt_total_ms",
    "candidate_cpp_total_ms",
    "python_validation_total_ms",
    "adapter_overhead_ms",
    "extract_local_path_ms",
    "sparsify_path_ms",
    "allocate_time_ms",
    "optimizer_ms",
    "validate_ms",
    "yaw_ms",
    "sample_ms",
)
TIMING_SUMMARY_FIELDS = (
    "variant", "scene_label", "metric_name", "n",
    "mean_ms", "median_ms", "p95_ms", "data_source",
)
CANDIDATE_SUMMARY_FIELDS = (
    "variant", "scene_label", "outcome", "metric_name", "n",
    "mean_ms", "median_ms", "p95_ms", "data_source",
)
CHART_FILENAMES = (
    "planning_stage_mean_bar.png",
    "planning_stage_p95_bar.png",
    "minco_stage_p95_bar.png",
    "candidate_screen_time_distribution.png",
    "candidate_attempts_vs_minco_latency.png",
    "minco_latency_by_outcome.png",
)


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_true(value):
    return str(value).strip().lower() == "true"


def describe(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "n": int(len(values)),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def summarize_timing_stages(rows):
    groups = {}
    sources = {}
    for row in rows:
        value = finite_number(row.get("duration_ms"))
        if value is None:
            continue
        key = (
            row.get("variant", "UNKNOWN"),
            row.get("scene_label", "UNKNOWN"),
            row.get("metric_name", "UNKNOWN"),
        )
        groups.setdefault(key, []).append(value)
        sources.setdefault(key, set()).add(row.get("data_source", "UNKNOWN"))
    output = []
    for key, values in sorted(groups.items()):
        stats = describe(values)
        source_values = sources[key]
        output.append({
            "variant": key[0],
            "scene_label": key[1],
            "metric_name": key[2],
            **stats,
            "data_source": (
                next(iter(source_values))
                if len(source_values) == 1 else "MIXED"
            ),
        })
    return output


def summarize_candidate_timings(rows):
    metrics = {
        "candidate_call_ms": "candidate_call_ms",
        "cpp_pipeline_ms": "cpp_pipeline_ms",
        "optimizer_ms": "optimizer_ms",
        "cpp_validation_ms": "cpp_validation_ms",
        "python_validation_ms": "python_validation_ms",
    }
    groups = {}
    sources = {}
    for row in rows:
        if not _is_true(row.get("attempted")):
            continue
        outcome = "ATTEMPT_SUCCESS" if _is_true(row.get("selected")) else "ATTEMPT_FAILED"
        for metric_name, field in metrics.items():
            value = finite_number(row.get(field))
            if value is None:
                continue
            key = (
                row.get("variant", "UNKNOWN"),
                row.get("scene_label", "UNKNOWN"),
                outcome,
                metric_name,
            )
            groups.setdefault(key, []).append(value)
            sources.setdefault(key, set()).add(row.get("data_source", "UNKNOWN"))
    output = []
    for key, values in sorted(groups.items()):
        stats = describe(values)
        source_values = sources[key]
        output.append({
            "variant": key[0],
            "scene_label": key[1],
            "outcome": key[2],
            "metric_name": key[3],
            **stats,
            "data_source": (
                next(iter(source_values))
                if len(source_values) == 1 else "MIXED"
            ),
        })
    return output


def _write_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish_figure(figure, axis, path, subtitle, data_source):
    axis.set_facecolor("white")
    figure.patch.set_facecolor("white")
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=INK)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)
    title = axis.get_title(loc="left")
    if title:
        axis.set_title(title, loc="left", weight="bold", y=1.11, pad=0)
    axis.text(
        0.0, 1.035, subtitle,
        transform=axis.transAxes,
        color="#607D8B",
        fontsize=9,
        ha="left",
        va="bottom",
    )
    figure.subplots_adjust(top=0.80)
    if data_source == "SIMULATED":
        figure.text(
            0.5, 0.5, "SIMULATED DATA", ha="center", va="center",
            alpha=0.12, fontsize=24, rotation=25, color=INK,
        )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt = _matplotlib()
    plt.close(figure)
    return path


def _skip(path, reason):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reason_path = path.with_suffix(path.suffix + ".skip_reason.txt")
    reason_path.write_text(str(reason).strip() + "\n", encoding="utf-8")
    return {
        "status": "skipped",
        "path": "",
        "skip_reason": str(reason),
    }


def _generated(path):
    return {"status": "generated", "path": str(path), "skip_reason": ""}


def _values_by_variant(timings, allowed_metrics, variants=None):
    output = {}
    for row in timings:
        metric = row.get("metric_name")
        variant = row.get("variant", "UNKNOWN")
        if metric not in allowed_metrics or (variants is not None and variant not in variants):
            continue
        value = finite_number(row.get("duration_ms"))
        if value is not None:
            output.setdefault((variant, metric), []).append(value)
    return output


def _stage_bar(path, timings, statistic, title, metrics, data_source, minco_only=False):
    groups = _values_by_variant(
        timings,
        metrics,
        variants={"minco-cold", "minco-hot"} if minco_only else None,
    )
    variants = [
        variant for variant in ("raw", "minco-cold", "minco-hot")
        if any(key[0] == variant for key in groups)
    ]
    stages = [
        metric for metric in metrics
        if any(key[1] == metric for key in groups)
    ]
    if not variants or not stages:
        return _skip(path, "No finite timing-stage observations.")

    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(10, max(4.8, len(stages) * 0.55)))
    y = np.arange(len(stages), dtype=float)
    height = 0.75 / max(1, len(variants))
    for variant_index, variant in enumerate(variants):
        values = []
        for stage in stages:
            stats = describe(groups.get((variant, stage), []))
            values.append(
                stats[f"{statistic}_ms"] if stats is not None else np.nan
            )
        offset = (variant_index - (len(variants) - 1) / 2.0) * height
        axis.barh(
            y + offset, values, height=height * 0.9,
            color=PALETTE.get(variant, "#90A4AE"), label=variant,
        )
    axis.set_yticks(y, [stage.removesuffix("_ms") for stage in stages])
    axis.invert_yaxis()
    axis.set_xlabel("Duration (ms)")
    axis.set_title(title, loc="left", weight="bold", pad=28)
    axis.legend(frameon=False, loc="lower right")
    sample_count = sum(len(values) for values in groups.values())
    _finish_figure(
        figure, axis, path,
        f"{statistic.upper()} by recorded stage; n={sample_count} stage observations",
        data_source,
    )
    return _generated(path)


def _screen_ecdf(path, timings, data_source):
    groups = _values_by_variant(
        timings, {"candidate_screen_ms"}, {"minco-cold", "minco-hot"}
    )
    groups = {
        variant: values
        for (variant, _), values in groups.items()
        if len(values) >= 2
    }
    if not groups:
        return _skip(path, "Need at least two finite screening observations per plotted variant.")
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for variant, values in sorted(groups.items()):
        x = np.sort(np.asarray(values, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        axis.step(x, y, where="post", color=PALETTE[variant], linewidth=2, label=f"{variant} (n={len(x)})")
    axis.set_xlabel("Candidate screening duration (ms)")
    axis.set_ylabel("Cumulative fraction")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Candidate screening duration distribution", loc="left", weight="bold", pad=28)
    axis.legend(frameon=False, loc="lower right")
    _finish_figure(
        figure, axis, path,
        "Planning-cycle grain; finite MINCO screening observations",
        data_source,
    )
    return _generated(path)


def _attempts_scatter(path, cycles, data_source):
    rows = []
    for row in cycles:
        if row.get("variant") not in {"minco-cold", "minco-hot"}:
            continue
        attempts = finite_number(row.get("attempted_candidate_count"))
        latency = finite_number(row.get("minco_ms"))
        if attempts is not None and latency is not None:
            rows.append((row.get("variant"), attempts, latency))
    if len(rows) < 8:
        return _skip(path, "Need at least eight MINCO cycles with attempts and latency.")
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for variant in ("minco-cold", "minco-hot"):
        variant_rows = [row for row in rows if row[0] == variant]
        if not variant_rows:
            continue
        axis.scatter(
            [row[1] for row in variant_rows],
            [row[2] for row in variant_rows],
            s=35, alpha=0.7, color=PALETTE[variant],
            edgecolor="white", linewidth=0.5,
            label=f"{variant} (n={len(variant_rows)})",
        )
    axis.set_xlabel("Attempted candidate count")
    axis.set_ylabel("MINCO duration (ms)")
    axis.set_title("Candidate attempts and MINCO duration", loc="left", weight="bold", pad=28)
    axis.legend(frameon=False)
    _finish_figure(
        figure, axis, path,
        f"One point per planning cycle; n={len(rows)}",
        data_source,
    )
    return _generated(path)


def _outcome_boxplot(path, cycles, data_source):
    groups = {"PUBLISHED": [], "HOLD_LAST": [], "STOP": []}
    for row in cycles:
        if row.get("variant") not in {"minco-cold", "minco-hot"}:
            continue
        value = finite_number(row.get("minco_ms"))
        if value is None:
            continue
        outcome = (
            "PUBLISHED" if _is_true(row.get("published"))
            else "HOLD_LAST" if row.get("fallback_mode") == "HOLD_LAST"
            else "STOP"
        )
        groups[outcome].append(value)
    labels = [label for label, values in groups.items() if values]
    if len(labels) < 2 or sum(len(groups[label]) for label in labels) < 6:
        return _skip(path, "Need at least six cycles across two MINCO outcomes.")
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    boxes = axis.boxplot(
        [groups[label] for label in labels],
        labels=[f"{label}\n(n={len(groups[label])})" for label in labels],
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.5},
    )
    colors = ("#4472C4", "#D6A42A", "#8064A2")
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_ylabel("MINCO duration (ms)")
    axis.set_title("MINCO duration by planning outcome", loc="left", weight="bold", pad=28)
    _finish_figure(
        figure, axis, path,
        "Planning-cycle grain; published and fallback outcomes",
        data_source,
    )
    return _generated(path)


def generate_timing_analysis(output_dir, timings, cycles, candidates, data_source):
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    stage_summary = summarize_timing_stages(timings)
    candidate_summary = summarize_candidate_timings(candidates)
    _write_rows(
        table_dir / "timing_stage_summary.csv",
        TIMING_SUMMARY_FIELDS,
        stage_summary,
    )
    _write_rows(
        table_dir / "candidate_timing_summary.csv",
        CANDIDATE_SUMMARY_FIELDS,
        candidate_summary,
    )

    results = {}
    results["planning_stage_mean_bar.png"] = _stage_bar(
        plot_dir / "planning_stage_mean_bar.png",
        timings, "mean", "Mean planning-stage duration",
        STAGE_FIELDS, data_source,
    )
    results["planning_stage_p95_bar.png"] = _stage_bar(
        plot_dir / "planning_stage_p95_bar.png",
        timings, "p95", "P95 planning-stage duration",
        STAGE_FIELDS, data_source,
    )
    results["minco_stage_p95_bar.png"] = _stage_bar(
        plot_dir / "minco_stage_p95_bar.png",
        timings, "p95", "P95 MINCO-stage duration",
        MINCO_STAGE_FIELDS, data_source, minco_only=True,
    )
    results["candidate_screen_time_distribution.png"] = _screen_ecdf(
        plot_dir / "candidate_screen_time_distribution.png",
        timings, data_source,
    )
    results["candidate_attempts_vs_minco_latency.png"] = _attempts_scatter(
        plot_dir / "candidate_attempts_vs_minco_latency.png",
        cycles, data_source,
    )
    results["minco_latency_by_outcome.png"] = _outcome_boxplot(
        plot_dir / "minco_latency_by_outcome.png",
        cycles, data_source,
    )
    return results
