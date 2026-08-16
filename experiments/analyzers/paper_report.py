from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.analyzers.artifact_manifest import (
    validate_paper_artifact_manifest,
)
from experiments.analyzers.data_quality import summarize_csv_paths
from experiments.analyzers.statistics import (
    bootstrap_ci,
    paired_bootstrap_ci,
    wilson_interval,
)
from experiments.core.artifact_receipt import (
    file_receipt,
    inventory_receipts,
)
from experiments.visualizers.paper_style import (
    apply_paper_style,
    save_paper_figure,
    scientific_caption,
)


GENERATOR_VERSION = "paper_report_v1"
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_ITERATIONS = 5000
VARIANT_ORDER = ("raw", "minco-cold", "minco-hot")
VARIANT_COLORS = {
    "raw": "#4472C4",
    "minco-cold": "#E69F00",
    "minco-hot": "#009E73",
}
PAIRED_KEY = "experiment_id + scene_id + seed + episode_uid"


def _discover(root: Path, filename: str) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.rglob(filename)
        if path.is_file() and ".tmp" not in path.name
    )


def _read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def _finite(row: Mapping[str, object], field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _boolean(row: Mapping[str, object], field: str) -> bool | None:
    value = str(row.get(field, "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def _variant_order(variants: Iterable[str]) -> list[str]:
    values = set(variants)
    return [variant for variant in VARIANT_ORDER if variant in values] + sorted(
        values - set(VARIANT_ORDER)
    )


def _episode_key(row: Mapping[str, object]) -> str | None:
    values = [
        str(row.get(field, "")).strip()
        for field in ("experiment_id", "scene_id", "seed", "episode_uid")
    ]
    if any(not value for value in values):
        return None
    return "|".join(values)


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(field) for row in rows for field in row})
    if not fields:
        fields = ["no_observations"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_classification(root: Path, rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "STATIC_ONLY"
    row_sources = {
        str(row.get("data_source", "")).strip().upper()
        for row in rows
        if str(row.get("data_source", "")).strip()
    }
    declared = ""
    config_path = root / "suite_config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        declared = str(payload.get("data_source", "")).strip().upper()
    if declared and row_sources and row_sources != {declared}:
        raise ValueError(
            f"suite data_source {declared} conflicts with row sources {sorted(row_sources)}"
        )
    if declared:
        return declared
    if len(row_sources) == 1:
        return next(iter(row_sources))
    if len(row_sources) > 1:
        return "MIXED"
    return "STATIC_ONLY"


def _validate_real_inputs(episode_paths: Sequence[Path], data_source: str) -> None:
    if data_source != "REAL":
        return
    for path in episode_paths:
        report_path = path.parent / "validation" / "validation_report.json"
        if not report_path.is_file():
            raise ValueError(f"REAL paper input is not validated: {path.parent}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("valid") is not True:
            raise ValueError(f"REAL paper input failed validation: {path.parent}")


def _episode_summary(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    summary: dict[str, dict[str, object]] = {}
    backing = []
    for variant in _variant_order(str(row.get("variant", "UNKNOWN")) for row in rows):
        subset = [row for row in rows if str(row.get("variant", "UNKNOWN")) == variant]
        successes = sum(_boolean(row, "success") is True for row in subset)
        total = len(subset)
        low, high = wilson_interval(successes, total)
        success_missing = sum(_boolean(row, "success") is None for row in subset)
        collisions = sum(_boolean(row, "collision") is True for row in subset)
        values: dict[str, object] = {
            "episode_count": total,
            "success_count": successes,
            "success_rate": successes / total,
            "success_ci95_low": low,
            "success_ci95_high": high,
            "success_missing_count": success_missing,
            "collision_count": collisions,
        }
        for field in (
            "episode_duration_s",
            "actual_path_length_m",
            "repository_spl",
            "tracking_error_rmse_m",
            "minimum_executed_clearance_m",
            "hold_duration_s",
            "stop_duration_s",
            "wheel_saturation_count",
        ):
            finite = [value for row in subset if (value := _finite(row, field)) is not None]
            values[f"{field}_n"] = len(finite)
            values[f"{field}_mean"] = float(np.mean(finite)) if finite else None
            values[f"{field}_median"] = float(np.median(finite)) if finite else None
        summary[variant] = values
        backing.append({"variant": variant, **values})
    return summary, backing


def _paired_success(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        key = _episode_key(row)
        outcome = _boolean(row, "success")
        variant = str(row.get("variant", "UNKNOWN"))
        if key is None or outcome is None:
            continue
        if key in values.setdefault(variant, {}):
            raise ValueError(f"duplicate paired episode key for {variant}: {key}")
        values[variant][key] = float(outcome)
    baseline = values.get("raw", {})
    output = []
    for method in ("minco-cold", "minco-hot"):
        if baseline and values.get(method):
            statistics = paired_bootstrap_ci(
                baseline,
                values[method],
                seed=BOOTSTRAP_SEED,
                iterations=BOOTSTRAP_ITERATIONS,
            )
            # An empty exact join is missing paired evidence, not a numerical
            # zero or a JSON NaN.  Keep the denominator explicitly as zero.
            for field in ("estimate", "ci_low", "ci_high"):
                value = statistics.get(field)
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    statistics[field] = None
            output.append(
                {
                    "baseline": "raw",
                    "method": method,
                    **statistics,
                }
            )
    return output


def _caption(
    title: str,
    *,
    source: str,
    units: str,
    sample_count: int,
    denominator: str,
    limitations: str,
    interpretation: str,
) -> str:
    return scientific_caption(
        title,
        source=source,
        units=units,
        sample_count=sample_count,
        missing_failed=f"Denominator: {denominator}",
        interpretation=interpretation,
        profile="RAW, MINCO-COLD, and MINCO-HOT when present",
    ) + (
        f"- Paired key: {PAIRED_KEY}\n"
        f"- Denominator: {denominator}\n"
        f"- Limitations: {limitations}\n"
    )


def _save_bundle(
    figure: plt.Figure,
    *,
    stem: str,
    output_dir: Path,
    backing_rows: Sequence[Mapping[str, object]],
    caption: str,
    input_paths: Sequence[Path],
    data_source: str,
    units: str,
    sample_count: int,
) -> dict[str, object]:
    receipt = save_paper_figure(
        figure,
        stem=stem,
        output_dir=output_dir,
        backing_rows=backing_rows,
        caption=caption,
        input_paths=input_paths,
    )
    receipt.update(
        {
            "generator_version": GENERATOR_VERSION,
            "data_source": data_source,
            "units": units,
            "sample_count": sample_count,
            "paired_key": PAIRED_KEY,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
    )
    _write_json(output_dir / "receipts" / f"{stem}.json", receipt)
    return receipt


def _success_figure(
    backing: list[dict[str, object]],
    output: Path,
    inputs: Sequence[Path],
    data_source: str,
) -> dict[str, object]:
    variants = [str(row["variant"]) for row in backing]
    rates = np.asarray([float(row["success_rate"]) for row in backing])
    low = np.asarray([float(row["success_ci95_low"]) for row in backing])
    high = np.asarray([float(row["success_ci95_high"]) for row in backing])
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    positions = np.arange(len(variants))
    axis.bar(
        positions,
        rates,
        color=[VARIANT_COLORS.get(variant, "#777777") for variant in variants],
        width=0.62,
    )
    axis.errorbar(
        positions,
        rates,
        yerr=np.vstack((rates - low, high - rates)),
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axis.set_xticks(positions, [variant.upper() for variant in variants])
    axis.set_ylabel("Success rate (proportion)")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Episode-level navigation success with Wilson 95% intervals")
    return _save_bundle(
        figure,
        stem="episode_success_rate",
        output_dir=output,
        backing_rows=backing,
        caption=_caption(
            "Episode-level navigation success",
            source=f"episode_metrics.csv; data_source={data_source}",
            units="dimensionless proportion",
            sample_count=sum(int(row["episode_count"]) for row in backing),
            denominator="all recorded episode rows per variant; missing success is retained as non-success and counted",
            limitations="Wilson intervals are descriptive; SIMULATED data are pipeline evidence, not real-world performance evidence.",
            interpretation="Higher bars indicate more successful terminal episodes. Intervals use the episode denominator, never planning-cycle rows.",
        ),
        input_paths=inputs,
        data_source=data_source,
        units="proportion",
        sample_count=sum(int(row["episode_count"]) for row in backing),
    )


def _continuous_episode_figure(
    rows: Sequence[Mapping[str, object]],
    output: Path,
    inputs: Sequence[Path],
    data_source: str,
) -> dict[str, object] | None:
    metrics = (
        ("repository_spl", "SPL", "dimensionless"),
        ("tracking_error_rmse_m", "Tracking RMSE", "m"),
        ("episode_duration_s", "Duration", "s"),
        ("actual_path_length_m", "Path length", "m"),
    )
    backing = []
    variants = _variant_order(str(row.get("variant", "UNKNOWN")) for row in rows)
    for field, label, units in metrics:
        for variant in variants:
            values = [
                value
                for row in rows
                if str(row.get("variant", "UNKNOWN")) == variant
                and (value := _finite(row, field)) is not None
            ]
            if values:
                low, high = bootstrap_ci(
                    values,
                    iterations=BOOTSTRAP_ITERATIONS,
                    seed=BOOTSTRAP_SEED,
                )
                backing.append(
                    {
                        "variant": variant,
                        "metric": field,
                        "label": label,
                        "units": units,
                        "n_finite": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    if not backing:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    for axis, (field, label, units) in zip(axes.flat, metrics):
        subset = [row for row in backing if row["metric"] == field]
        positions = np.arange(len(subset))
        means = np.asarray([float(row["mean"]) for row in subset])
        low = np.asarray([float(row["ci95_low"]) for row in subset])
        high = np.asarray([float(row["ci95_high"]) for row in subset])
        axis.bar(
            positions,
            means,
            color=[VARIANT_COLORS.get(str(row["variant"]), "#777777") for row in subset],
        )
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack((means - low, high - means)),
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        axis.set_xticks(positions, [str(row["variant"]).upper() for row in subset], rotation=15)
        axis.set_ylabel(f"{label} ({units})")
        axis.set_title(label)
    return _save_bundle(
        figure,
        stem="episode_navigation_metrics",
        output_dir=output,
        backing_rows=backing,
        caption=_caption(
            "Episode-level continuous navigation metrics",
            source=f"episode_metrics.csv; data_source={data_source}",
            units="SPL (1), tracking error/path length (m), duration (s)",
            sample_count=len(rows),
            denominator="finite episode-level observations are reported separately for each metric and variant",
            limitations="Missing continuous outcomes remain missing; bootstrap intervals describe observed episodes only.",
            interpretation="Panels retain their physical units and therefore avoid a visually misleading mixed-unit composite score.",
        ),
        input_paths=inputs,
        data_source=data_source,
        units="mixed; declared per backing row",
        sample_count=len(rows),
    )


def _categorical_figure(
    rows: Sequence[Mapping[str, object]],
    output: Path,
    inputs: Sequence[Path],
    data_source: str,
) -> dict[str, object]:
    variants = _variant_order(str(row.get("variant", "UNKNOWN")) for row in rows)
    categories = sorted(
        {str(row.get("done_reason", "")).strip() or "MISSING" for row in rows}
    )
    backing = [
        {
            "variant": variant,
            "termination": category,
            "episode_count": sum(
                str(row.get("variant", "UNKNOWN")) == variant
                and (str(row.get("done_reason", "")).strip() or "MISSING") == category
                for row in rows
            ),
            "variant_denominator": sum(
                str(row.get("variant", "UNKNOWN")) == variant for row in rows
            ),
        }
        for variant in variants
        for category in categories
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    bottoms = np.zeros(len(variants))
    for category in categories:
        counts = np.asarray(
            [
                next(
                    int(row["episode_count"])
                    for row in backing
                    if row["variant"] == variant and row["termination"] == category
                )
                for variant in variants
            ]
        )
        axis.bar(variants, counts, bottom=bottoms, label=category)
        bottoms += counts
    axis.set_ylabel("Episode count")
    axis.set_title("Machine-recorded termination composition")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return _save_bundle(
        figure,
        stem="episode_termination_outcomes",
        output_dir=output,
        backing_rows=backing,
        caption=_caption(
            "Episode termination composition",
            source=f"episode_metrics.csv done_reason; data_source={data_source}",
            units="episode count",
            sample_count=len(rows),
            denominator="all episode rows per variant, including blank termination as MISSING",
            limitations="Collision object and impact force are not inferred when unavailable; categories reflect recorded machine terms only.",
            interpretation="Stacks expose the terminal-outcome composition without removing failed or missing episodes.",
        ),
        input_paths=inputs,
        data_source=data_source,
        units="episode count",
        sample_count=len(rows),
    )


def _episode_recovery_figure(
    rows: Sequence[Mapping[str, object]],
    output: Path,
    inputs: Sequence[Path],
    data_source: str,
) -> dict[str, object] | None:
    metrics = (
        ("hold_duration_s", "HOLD duration", "s"),
        ("stop_duration_s", "STOP duration", "s"),
        ("wheel_saturation_count", "Wheel saturation", "count/episode"),
    )


def _static_dynamic_figure(
    episode_rows: Sequence[Mapping[str, object]],
    static_rows: Sequence[Mapping[str, object]],
    output: Path,
    inputs: Sequence[Path],
    data_source: str,
) -> dict[str, object] | None:
    static_lookup: dict[str, Mapping[str, object]] = {}
    for row in static_rows:
        if str(row.get("profile", "")) != "superplanner_sfc_v1":
            continue
        uid = str(row.get("case_uid", "")).strip()
        if not uid:
            continue
        if uid in static_lookup:
            raise ValueError(f"duplicate SuperPlanner SFC static case for dynamic join: {uid}")
        static_lookup[uid] = row
    backing = []
    for row in episode_rows:
        uid = str(row.get("static_selected_case_uid", "")).strip()
        outcome = _boolean(row, "success")
        if uid not in static_lookup or outcome is None:
            continue
        static = static_lookup[uid]
        backing.append(
            {
                "episode_key": _episode_key(row) or "MISSING_KEY",
                "variant": str(row.get("variant", "UNKNOWN")),
                "case_uid": uid,
                "static_classification": str(
                    static.get("classification", "MISSING")
                ),
                "static_min_normalized_margin": _finite(
                    static, "min_normalized_margin"
                ),
                "dynamic_success": bool(outcome),
            }
        )
    if not backing:
        return None
    categories = sorted({str(row["static_classification"]) for row in backing})
    rates = []
    for category in categories:
        values = [
            bool(row["dynamic_success"])
            for row in backing
            if row["static_classification"] == category
        ]
        rates.append(sum(values) / len(values))
    figure, axis = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    axis.bar(categories, rates, color="#56B4E9")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Dynamic episode success rate")
    axis.set_xlabel("Static safe-profile classification")
    axis.set_title("Static prediction versus dynamic outcome")
    axis.tick_params(axis="x", rotation=20)
    return _save_bundle(
        figure,
        stem="static_dynamic_outcomes",
        output_dir=output,
        backing_rows=backing,
        caption=_caption(
            "Static prediction versus dynamic outcome",
            source=f"static_runs.csv joined to episode_metrics.csv by static_selected_case_uid; data_source={data_source}",
            units="dynamic episode success proportion; static margin dimensionless",
            sample_count=len(backing),
            denominator="dynamic episodes with an explicit selected case UID, binary success, and matching safe-profile static row",
            limitations="This descriptive join measures consistency, not calibration causality; unmatched cases remain unavailable and are not imputed.",
            interpretation="Bars reveal whether static feasibility classes agree with observed dynamic terminal success for frozen cases.",
        ),
        input_paths=inputs,
        data_source=data_source,
        units="proportion and dimensionless margin",
        sample_count=len(backing),
    )
    backing = []
    variants = _variant_order(str(row.get("variant", "UNKNOWN")) for row in rows)
    for field, label, units in metrics:
        for variant in variants:
            values = [
                value
                for row in rows
                if str(row.get("variant", "UNKNOWN")) == variant
                and (value := _finite(row, field)) is not None
            ]
            if values:
                backing.append(
                    {
                        "variant": variant,
                        "metric": field,
                        "label": label,
                        "units": units,
                        "n_finite": len(values),
                        "mean": float(np.mean(values)),
                    }
                )
    if not backing:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.7), constrained_layout=True)
    for axis, (field, label, units) in zip(axes, metrics):
        subset = [row for row in backing if row["metric"] == field]
        axis.bar(
            [str(row["variant"]).upper() for row in subset],
            [float(row["mean"]) for row in subset],
            color=[VARIANT_COLORS.get(str(row["variant"]), "#777777") for row in subset],
        )
        axis.set_title(label)
        axis.set_ylabel(units)
        axis.tick_params(axis="x", rotation=20)
    return _save_bundle(
        figure,
        stem="episode_recovery_outcomes",
        output_dir=output,
        backing_rows=backing,
        caption=_caption(
            "Episode recovery and saturation outcomes",
            source=f"episode_metrics.csv; data_source={data_source}",
            units="seconds and count per episode",
            sample_count=len(rows),
            denominator="finite episode-level recovery fields per metric and variant",
            limitations="A blank recovery field is unavailable, not zero, and is excluded only from that continuous metric denominator.",
            interpretation="Lower durations and saturation counts indicate less reliance on recovery behavior, without implying causal superiority.",
        ),
        input_paths=inputs,
        data_source=data_source,
        units="mixed; declared per backing row",
        sample_count=len(rows),
    )


def _cluster_rows_by_episode(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = _episode_key(row)
        if key is not None:
            grouped.setdefault((str(row.get("variant", "UNKNOWN")), key), []).append(row)
    collapsed = []
    for (variant, key), group in sorted(grouped.items()):
        output: dict[str, object] = {"variant": variant, "episode_key": key}
        for field, reduction in metrics:
            values = [value for row in group if (value := _finite(row, field)) is not None]
            if values:
                output[field] = float(min(values) if reduction == "min" else np.mean(values))
        collapsed.append(output)
    return collapsed


def _clustered_metric_figure(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[tuple[str, str, str]],
    stem: str,
    title: str,
    output: Path,
    inputs: Sequence[Path],
    data_source: str,
    limitations: str,
) -> dict[str, object] | None:
    collapsed = _cluster_rows_by_episode(
        rows, [(field, reduction) for field, _, reduction in metrics]
    )
    backing = []
    variants = _variant_order(str(row["variant"]) for row in collapsed)
    for field, units, _ in metrics:
        for variant in variants:
            values = [
                float(row[field])
                for row in collapsed
                if row["variant"] == variant and field in row
            ]
            if values:
                backing.append(
                    {
                        "variant": variant,
                        "metric": field,
                        "units": units,
                        "episode_count": len(values),
                        "episode_mean": float(np.mean(values)),
                        "episode_median": float(np.median(values)),
                    }
                )
    if not backing:
        return None
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(max(5.0, 3.6 * len(metrics)), 3.8),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (field, units, _) in zip(axes.flat, metrics):
        subset = [row for row in backing if row["metric"] == field]
        axis.bar(
            [str(row["variant"]).upper() for row in subset],
            [float(row["episode_mean"]) for row in subset],
            color=[VARIANT_COLORS.get(str(row["variant"]), "#777777") for row in subset],
        )
        axis.set_title(field.replace("_", " "))
        axis.set_ylabel(units)
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle(title)
    return _save_bundle(
        figure,
        stem=stem,
        output_dir=output,
        backing_rows=backing,
        caption=_caption(
            title,
            source=f"recorded samples collapsed by episode; data_source={data_source}",
            units="declared per backing row",
            sample_count=len(collapsed),
            denominator="episodes with at least one finite sample; cycle/control samples are never treated as independent trials",
            limitations=limitations,
            interpretation="Each bar summarizes episode-level clusters, preventing high-frequency samples from inflating the trial denominator.",
        ),
        input_paths=inputs,
        data_source=data_source,
        units="mixed; declared per backing row",
        sample_count=len(collapsed),
    )


def _selection_directory(root: Path) -> Path | None:
    candidates = []
    for path in root.rglob("selected_dynamic_cases.json"):
        if (path.parent / "static_runs.csv").is_file():
            candidates.append(path.parent.resolve())
    if not candidates:
        return None
    if len(set(candidates)) > 1:
        raise ValueError(f"multiple static selection inputs found: {candidates}")
    return candidates[0]


def generate_paper_report(
    input_root: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    """Generate immutable, data-driven paper tables, figures, and provenance."""

    input_root = Path(input_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"paper input root does not exist: {input_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable paper output already exists: {output_dir}")
    selection_dir = _selection_directory(input_root)
    episode_paths = _discover(input_root, "episode_metrics.csv")
    control_paths = _discover(input_root, "control_samples.csv")
    cycle_paths = _discover(input_root, "planning_cycles.csv")
    plan_paths = _discover(input_root, "plan_metrics.csv")
    static_run_paths = _discover(input_root, "static_runs.csv")
    auxiliary_paths = [
        path
        for filename in ("candidate_metrics.csv", "timing_samples.csv", "events.csv")
        for path in _discover(input_root, filename)
    ]
    all_csv_paths = sorted(
        set(
            episode_paths
            + control_paths
            + cycle_paths
            + plan_paths
            + static_run_paths
            + auxiliary_paths
        )
    )
    episode_rows = _read_rows(episode_paths)
    data_source = _source_classification(input_root, episode_rows)
    _validate_real_inputs(episode_paths, data_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    summary, summary_rows = _episode_summary(episode_rows) if episode_rows else ({}, [])
    _write_json(output_dir / "tables" / "episode_summary.json", summary)
    _write_csv(output_dir / "tables" / "episode_summary.csv", summary_rows)
    paired = _paired_success(episode_rows) if episode_rows else []
    _write_json(output_dir / "tables" / "paired_success_bootstrap.json", paired)
    _write_csv(output_dir / "tables" / "paired_success_bootstrap.csv", paired)
    quality = summarize_csv_paths(all_csv_paths, root=input_root)
    _write_csv(output_dir / "tables" / "data_quality.csv", quality)
    _write_json(output_dir / "tables" / "data_quality.json", quality)

    figure_receipts = []
    generated_panels = []
    skipped_panels = []
    if episode_rows:
        figure_receipts.append(
            _success_figure(summary_rows, output_dir, episode_paths, data_source)
        )
        generated_panels.append("episode_success_rate")
        continuous = _continuous_episode_figure(
            episode_rows, output_dir, episode_paths, data_source
        )
        if continuous:
            figure_receipts.append(continuous)
            generated_panels.append("episode_navigation_metrics")
        termination = _categorical_figure(
            episode_rows, output_dir, episode_paths, data_source
        )
        figure_receipts.append(termination)
        generated_panels.append("episode_termination_outcomes")
        recovery = _episode_recovery_figure(
            episode_rows, output_dir, episode_paths, data_source
        )
        if recovery:
            figure_receipts.append(recovery)
            generated_panels.append("episode_recovery_outcomes")
    else:
        skipped_panels.extend(
            [
                "episode_success_rate: no episode_metrics.csv",
                "episode_navigation_metrics: no episode_metrics.csv",
                "episode_termination_outcomes: no episode_metrics.csv",
                "episode_recovery_outcomes: no episode_metrics.csv",
            ]
        )

    static_rows = _read_rows(static_run_paths)
    static_dynamic = _static_dynamic_figure(
        episode_rows,
        static_rows,
        output_dir,
        episode_paths + static_run_paths,
        data_source,
    ) if episode_rows and static_rows else None
    if static_dynamic:
        figure_receipts.append(static_dynamic)
        generated_panels.append("static_dynamic_outcomes")
    else:
        skipped_panels.append(
            "static_dynamic_outcomes: no joinable static/dynamic case evidence"
        )

    plan_rows = _read_rows(plan_paths)
    plan_receipt = _clustered_metric_figure(
        plan_rows,
        metrics=(
            ("raw_min_clearance_m", "m", "min"),
            ("minco_min_clearance_m", "m", "min"),
            ("planning_total_ms", "ms", "mean"),
        ),
        stem="planning_safety_profile",
        title="Episode-clustered planning safety and runtime profile",
        output=output_dir,
        inputs=plan_paths,
        data_source=data_source,
        limitations="Only finite recorded planner metrics are shown; failed plans remain represented by episode outcome panels and are not assigned artificial clearance.",
    ) if plan_rows else None
    if plan_receipt:
        figure_receipts.append(plan_receipt)
        generated_panels.append("planning_safety_profile")
    else:
        skipped_panels.append("planning_safety_profile: no supported plan evidence")

    control_rows = _read_rows(control_paths)
    control_receipt = _clustered_metric_figure(
        control_rows,
        metrics=(
            ("time_aligned_position_error_m", "m", "mean"),
            ("executed_clearance_m", "m", "min"),
            ("reference_age_ms", "ms", "mean"),
        ),
        stem="control_tracking_safety",
        title="Executed tracking, clearance, and reference age",
        output=output_dir,
        inputs=control_paths,
        data_source=data_source,
        limitations="Unavailable control or clearance signals remain absent; static clearance is not substituted for executed clearance.",
    ) if control_rows else None
    if control_receipt:
        figure_receipts.append(control_receipt)
        generated_panels.append("control_tracking_safety")
    else:
        skipped_panels.append("control_tracking_safety: no supported control evidence")

    cycle_rows = _read_rows(cycle_paths)
    latency_receipt = _clustered_metric_figure(
        cycle_rows,
        metrics=(
            ("planning_total_ms", "ms", "mean"),
            ("plan_age_when_applied_ms", "ms", "mean"),
        ),
        stem="planning_latency_reference_age",
        title="Episode-clustered planning latency and applied-plan age",
        output=output_dir,
        inputs=cycle_paths,
        data_source=data_source,
        limitations="The plot is descriptive and does not count correlated planning cycles as independent experimental trials.",
    ) if cycle_rows else None
    if latency_receipt:
        figure_receipts.append(latency_receipt)
        generated_panels.append("planning_latency_reference_age")
    else:
        skipped_panels.append("planning_latency_reference_age: no supported cycle evidence")

    static_manifest = None
    if selection_dir is not None:
        from experiments.analyzers.static_comparison import (
            generate_static_paper_outputs,
        )

        static_manifest = generate_static_paper_outputs(
            selection_dir, output_dir / "static"
        )
        generated_panels.append(
            f"static_bundle:{static_manifest['figure_count']} figures"
        )
    else:
        skipped_panels.append("static_bundle: no static selection evidence")

    dynamic_evidence = (
        "UNAVAILABLE"
        if not episode_rows
        else "REAL_VALIDATED"
        if data_source == "REAL"
        else f"{data_source}_NON_REAL"
    )
    report_lines = [
        "# NavDP–MINCO Data-Driven Paper Report",
        "",
        "## Evidence boundary",
        "",
        f"- Data source: `{data_source}`",
        f"- Dynamic evidence: `{dynamic_evidence}`",
        f"- Episode rows: {len(episode_rows)}",
        f"- Paired key: `{PAIRED_KEY}`",
        f"- Bootstrap: seed={BOOTSTRAP_SEED}, iterations={BOOTSTRAP_ITERATIONS}",
        "",
    ]
    if data_source != "REAL":
        report_lines.extend(
            [
                "> Non-REAL evidence validates the experiment and analysis pipeline only. It does not establish real-simulation or deployment performance.",
                "",
            ]
        )
    report_lines.extend(["## Episode results", ""])
    if not summary:
        report_lines.append("Dynamic panels are **UNAVAILABLE** because no episode table was supplied.")
    else:
        for variant in _variant_order(summary):
            row = summary[variant]
            report_lines.append(
                f"- {variant}: {row['success_count']}/{row['episode_count']} successes "
                f"({100.0 * float(row['success_rate']):.1f}%; Wilson 95% CI "
                f"[{100.0 * float(row['success_ci95_low']):.1f}%, "
                f"{100.0 * float(row['success_ci95_high']):.1f}%])."
            )
    report_lines.extend(
        [
            "",
            "## Generated panels",
            "",
            *[f"- {panel}" for panel in generated_panels],
            "",
            "## Unavailable or unsupported panels",
            "",
            *[f"- {panel}" for panel in skipped_panels],
            "",
            "## Interpretation limits",
            "",
            "Failures remain in discrete outcome denominators. Continuous fields use their explicitly reported finite-observation denominator; missing values are never converted to zero. Planning and control samples are clustered within episode before method summaries.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    consumed_inputs = sorted(
        set(all_csv_paths)
        | ({input_root / "suite_config.json"} if (input_root / "suite_config.json").is_file() else set())
        | (
            {
                selection_dir / "static_runs.csv",
                selection_dir / "selected_dynamic_cases.json",
            }
            if selection_dir is not None
            else set()
        )
    )
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "data_source": data_source,
        "dynamic_evidence": dynamic_evidence,
        "episode_count": len(episode_rows),
        "figure_count": len(figure_receipts)
        + (int(static_manifest["figure_count"]) if static_manifest else 0),
        "generated_panels": generated_panels,
        "skipped_panels": skipped_panels,
        "paired_key": PAIRED_KEY,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "iterations": BOOTSTRAP_ITERATIONS,
            "cluster_unit": "episode_or_case_key",
        },
        "inputs": [file_receipt(path, input_root) for path in consumed_inputs],
        "claims": {
            "data_driven_only": True,
            "hardcoded_experiment_values": False,
            "real_performance_claim_allowed": data_source == "REAL",
        },
    }
    _write_json(output_dir / "paper_manifest.json", manifest)
    inventory_path = output_dir / "artifact_receipt.json"
    _write_json(
        inventory_path,
        {
            "schema_version": 1,
            "root": ".",
            "artifacts": inventory_receipts(
                output_dir, exclude=(inventory_path,)
            ),
        },
    )
    validation_errors = validate_paper_artifact_manifest(output_dir)
    if validation_errors:
        raise RuntimeError(
            "generated paper artifact validation failed: "
            + "; ".join(validation_errors)
        )
    return manifest
