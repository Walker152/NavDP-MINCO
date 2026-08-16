from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from matplotlib.patches import Rectangle

from experiments.core.artifact_receipt import (
    file_receipt,
    inventory_receipts,
    validate_file_receipt,
)
from experiments.visualizers.paper_style import (
    PAPER_COLORS,
    apply_paper_style,
    save_paper_figure,
    scientific_caption,
)


PROFILES = ("legacy", "superplanner_sfc_v1")


def validate_static_paper_outputs(output_dir: Path | str) -> list[str]:
    root = Path(output_dir).resolve()
    manifest_path = root / "static_paper_manifest.json"
    inventory_path = root / "artifact_receipt.json"
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable static paper manifest: {error}"]
    receipts = manifest.get("figure_receipts", [])
    if not isinstance(receipts, list) or not receipts:
        errors.append("static paper manifest has no figure receipts")
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            errors.append("malformed static paper figure receipt")
            continue
        for row in receipt.get("outputs", []):
            if isinstance(row, Mapping):
                errors.extend(validate_file_receipt(root, row))
            else:
                errors.append("malformed static paper output receipt")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for row in inventory.get("artifacts", []):
            if isinstance(row, Mapping):
                errors.extend(validate_file_receipt(root, row))
            else:
                errors.append("malformed static paper inventory receipt")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable static paper inventory: {error}")
    return errors


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _missing_summary(rows: Sequence[Mapping[str, object]], key: str) -> str:
    missing = sum(not math.isfinite(_number(row, key)) for row in rows)
    failed = sum(str(row.get("status")) != "SUCCEEDED" for row in rows)
    return (
        f"{missing} rows have no finite {key}; {failed}/{len(rows)} runs failed. "
        "Failed runs remain in the stated denominator."
    )


def _bundle(
    output_dir: Path,
    input_paths: Sequence[Path],
    stem: str,
    figure: plt.Figure,
    rows: Sequence[Mapping[str, object]],
    caption: str,
) -> dict[str, object]:
    return save_paper_figure(
        figure,
        stem=stem,
        output_dir=output_dir,
        backing_rows=rows,
        caption=caption,
        input_paths=input_paths,
    )


def _single_factor(
    rows: list[dict[str, str]], output: Path, inputs: Sequence[Path]
) -> dict[str, object]:
    values = [
        row
        for row in rows
        if row.get("profile") in PROFILES
        and row.get("factor_name") not in {"", "geometry_category"}
        and not row.get("factor_name_secondary")
        and math.isfinite(_number(row, "factor_level"))
    ]
    factors = sorted({row["factor_name"] for row in values})
    panel_count = max(1, len(factors))
    columns = 2
    row_count = math.ceil(panel_count / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(9.0, max(3.2, row_count * 2.7)),
        constrained_layout=True,
        squeeze=False,
    )
    labels = {
        "initial_speed_mps": "Initial longitudinal speed (m/s)",
        "initial_lateral_speed_mps": "Initial lateral speed (m/s)",
        "initial_acceleration_x_mps2": "Initial longitudinal acceleration (m/s²)",
        "initial_acceleration_y_mps2": "Initial lateral acceleration (m/s²)",
        "initial_lateral_offset_m": "Initial lateral offset (m)",
        "initial_yaw_rate_radps": "Initial yaw rate (rad/s)",
        "yaw_error_rad": "Initial yaw error (rad)",
    }
    for axis, factor in zip(axes.flat, factors):
        subset = [row for row in values if row["factor_name"] == factor]
        for profile in PROFILES:
            profile_rows = sorted(
                (row for row in subset if row["profile"] == profile),
                key=lambda row: _number(row, "factor_level"),
            )
            finite = [
                row for row in profile_rows
                if math.isfinite(_number(row, "min_normalized_margin"))
            ]
            axis.plot(
                [_number(row, "factor_level") for row in finite],
                [_number(row, "min_normalized_margin") for row in finite],
                marker="o" if profile == "legacy" else "s",
                linestyle="--" if profile == "legacy" else "-",
                linewidth=1.5,
                color=PAPER_COLORS[profile],
                label=f"{profile} (n={len(profile_rows)})",
            )
            failed = [row for row in profile_rows if row not in finite]
            if failed:
                axis.scatter(
                    [_number(row, "factor_level") for row in failed],
                    [0.04 if profile == "legacy" else 0.08] * len(failed),
                    transform=axis.get_xaxis_transform(),
                    marker="x",
                    color=PAPER_COLORS[profile],
                    zorder=4,
                )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel(labels.get(factor, factor))
        axis.set_ylabel("Normalized margin (1)")
        axis.set_title(f"{factor} · n={len(subset)}", loc="left", fontsize=9)
        axis.legend(loc="best", frameon=False, fontsize=7)
    if not values:
        axes.flat[0].text(
            0.5, 0.5, "No isolated single-factor rows", ha="center", va="center"
        )
    for axis in list(axes.flat)[len(factors) :]:
        axis.set_visible(False)
    fig.suptitle("Controlled single-factor initial-state scans")
    return _bundle(
        output,
        inputs,
        "single_factor_margin_curves",
        fig,
        values,
        scientific_caption(
            "Controlled single-factor initial-state scan",
            source="static_runs.csv; paired native legacy and SuperPlanner 2-D SFC recomputations",
            units="factor-specific SI units; normalized margin is dimensionless",
            sample_count=len(values),
            missing_failed=_missing_summary(values, "min_normalized_margin") if values else "No isolated rows; n=0 is explicitly reported.",
            interpretation="Each line varies one declared initial-state factor while retaining the configured source geometry.",
            profile="legacy and superplanner_sfc_v1",
        ),
    )


def _factor_heatmap(
    rows: list[dict[str, str]], output: Path, inputs: Sequence[Path]
) -> dict[str, object]:
    values = [
        row
        for row in rows
        if row.get("profile") in PROFILES
        and row.get("factor_name_secondary")
        and math.isfinite(_number(row, "factor_level"))
        and math.isfinite(_number(row, "factor_level_secondary"))
    ]
    groups = sorted({
        row.get("scan_group") or f"{row['factor_name']}__{row['factor_name_secondary']}"
        for row in values
    })
    fig, axes = plt.subplots(
        max(1, len(groups)), len(PROFILES),
        figsize=(10.5, max(4.0, 3.6 * max(1, len(groups)))),
        constrained_layout=True, squeeze=False,
    )
    finite_margins = [
        _number(row, "min_normalized_margin") for row in values
        if math.isfinite(_number(row, "min_normalized_margin"))
    ]
    span = max(0.001, max((abs(value) for value in finite_margins), default=0.0))
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    colour_map = plt.get_cmap("RdYlGn").copy()
    colour_map.set_bad("#D1D5DB")
    axis_labels = {
        "initial_speed_mps": "Initial longitudinal speed (m/s)",
        "initial_acceleration_x_mps2": "Initial longitudinal acceleration (m/s²)",
        "initial_yaw_rate_radps": "Initial yaw rate (rad/s)",
        "yaw_error_rad": "Initial yaw error (rad)",
    }
    last_image = None
    if not values:
        axes[0, 0].text(0.5, 0.5, "No two-factor measurements", ha="center", va="center")
        axes[0, 1].set_visible(False)
    for group_index, group in enumerate(groups):
        group_rows = [
            row for row in values
            if (row.get("scan_group") or f"{row['factor_name']}__{row['factor_name_secondary']}") == group
        ]
        for profile_index, profile in enumerate(PROFILES):
            ax = axes[group_index, profile_index]
            subset = [row for row in group_rows if row["profile"] == profile]
            first_name = group_rows[0]["factor_name"]
            second_name = group_rows[0]["factor_name_secondary"]
            x_values = sorted({_number(row, "factor_level") for row in subset})
            y_values = sorted({_number(row, "factor_level_secondary") for row in subset})
            matrix = np.full((len(y_values), len(x_values)), np.nan)
            classifications: dict[tuple[int, int], str] = {}
            for row in subset:
                y_index = y_values.index(_number(row, "factor_level_secondary"))
                x_index = x_values.index(_number(row, "factor_level"))
                matrix[y_index, x_index] = _number(row, "min_normalized_margin")
                classifications[(y_index, x_index)] = str(row.get("classification", ""))
            if matrix.size:
                last_image = ax.imshow(
                    matrix, origin="lower", aspect="auto", cmap=colour_map, norm=norm,
                )
                ax.set_xticks(range(len(x_values)), [f"{value:g}" for value in x_values])
                ax.set_yticks(range(len(y_values)), [f"{value:g}" for value in y_values])
                for y_index in range(len(y_values)):
                    for x_index in range(len(x_values)):
                        value = matrix[y_index, x_index]
                        classification = classifications.get((y_index, x_index), "MISSING")
                        label = f"{value:.2f}" if math.isfinite(value) else classification.replace("_", "\n")
                        ax.text(x_index, y_index, label, ha="center", va="center", fontsize=6)
            else:
                ax.text(0.5, 0.5, "No measurements", ha="center", va="center")
            ax.set_xlabel(axis_labels.get(first_name, first_name))
            ax.set_ylabel(axis_labels.get(second_name, second_name))
            ax.set_title(f"{group} · {profile} · n={len(subset)}", fontsize=8)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), label="minimum normalized margin (1)")
    fig.suptitle("Paired measured two-factor capability boundaries")
    first_name = values[0]["factor_name"] if values else "factor x"
    second_name = values[0]["factor_name_secondary"] if values else "factor y"
    return _bundle(
        output,
        inputs,
        "two_factor_margin_heatmap",
        fig,
        values,
        scientific_caption(
            "Paired measured two-factor capability boundaries",
            source="static_runs.csv; configured factor grids grouped by scan_group and paired by profile",
            units=f"{first_name} and {second_name} in declared SI units; cell value dimensionless",
            sample_count=len(values),
            missing_failed=_missing_summary(values, "min_normalized_margin") if values else "No two-factor rows; n=0 is explicitly reported.",
            interpretation="Every cell is an executed factor combination; scan groups and profiles use separate panels with one shared colour scale.",
            profile="legacy and superplanner_sfc_v1",
        ),
    )


def _transitions(
    rows: list[dict[str, str]], output: Path, inputs: Sequence[Path]
) -> dict[str, object]:
    lookup = {(row["case_uid"], row["profile"]): row for row in rows}
    paired = sorted(
        uid
        for uid in {row["case_uid"] for row in rows}
        if all((uid, profile) in lookup for profile in PROFILES)
    )
    classes = sorted(
        {
            lookup[(uid, profile)]["classification"]
            for uid in paired
            for profile in PROFILES
        }
    )
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    backing: list[dict[str, object]] = []
    for uid in paired:
        left = lookup[(uid, "legacy")]["classification"]
        right = lookup[(uid, "superplanner_sfc_v1")]["classification"]
        matrix[classes.index(left), classes.index(right)] += 1
        backing.append(
            {"case_uid": uid, "legacy": left, "superplanner_sfc_v1": right}
        )
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    if matrix.size:
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, label="paired case count")
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        for y, x in np.ndindex(matrix.shape):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center")
    ax.set_xlabel("SuperPlanner 2-D SFC classification")
    ax.set_ylabel("legacy classification")
    ax.set_title("Paired profile classification transitions")
    return _bundle(
        output,
        inputs,
        "classification_transition_matrix",
        fig,
        backing,
        scientific_caption(
            "Paired profile classification transitions",
            source="static_runs.csv joined exactly by case_uid",
            units="paired case count",
            sample_count=len(paired),
            missing_failed=f"{len({row['case_uid'] for row in rows}) - len(paired)} unpaired cases; failures retained as classifications.",
            interpretation="Diagonal cells preserve classification; off-diagonal cells show the direction of profile-induced changes.",
        ),
    )


def _failure_stack(
    rows: list[dict[str, str]], output: Path, inputs: Sequence[Path]
) -> dict[str, object]:
    reasons = sorted(
        {row.get("failure_reason") or "NONE" for row in rows}
    )
    backing = [
        {
            "profile": profile,
            "failure_reason": reason,
            "case_count": sum(
                row["profile"] == profile
                and (row.get("failure_reason") or "NONE") == reason
                for row in rows
            ),
        }
        for profile in PROFILES
        for reason in reasons
    ]
    fig, ax = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    bottoms = np.zeros(len(PROFILES))
    for reason in reasons:
        counts = np.asarray(
            [
                next(
                    int(row["case_count"])
                    for row in backing
                    if row["profile"] == profile
                    and row["failure_reason"] == reason
                )
                for profile in PROFILES
            ]
        )
        ax.bar(PROFILES, counts, bottom=bottoms, label=reason)
        bottoms += counts
    ax.set_ylabel("case count")
    ax.set_title("Failure-reason composition with fixed denominators")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return _bundle(
        output,
        inputs,
        "failure_reason_stack",
        fig,
        backing,
        scientific_caption(
            "Failure-reason composition with fixed denominators",
            source="static_runs.csv",
            units="case count",
            sample_count=len(rows),
            missing_failed="Blank reasons are encoded as NONE; all failed cases remain in profile denominators.",
            interpretation="The stack separates fail-closed causes from successful NONE outcomes without dropping invalid runs.",
        ),
    )


def _pareto(
    rows: list[dict[str, str]], output: Path, inputs: Sequence[Path]
) -> dict[str, object]:
    values = [
        row
        for row in rows
        if math.isfinite(_number(row, "runtime_ms"))
        and math.isfinite(_number(row, "guide_deviation_p95_m"))
    ]
    fig, ax = plt.subplots(figsize=(6.7, 4.6), constrained_layout=True)
    for profile in PROFILES:
        subset = [row for row in values if row["profile"] == profile]
        ax.scatter(
            [_number(row, "runtime_ms") for row in subset],
            [_number(row, "guide_deviation_p95_m") for row in subset],
            color=PAPER_COLORS[profile],
            marker="o" if profile == "legacy" else "s",
            alpha=0.78,
            label=f"{profile} (n={len(subset)})",
        )
    ax.set_xlabel("Native optimization runtime (ms)")
    ax.set_ylabel("Guide deviation p95 (m)")
    ax.set_title("Runtime–shape fidelity Pareto view")
    ax.legend(loc="best")
    return _bundle(
        output,
        inputs,
        "runtime_shape_pareto",
        fig,
        values,
        scientific_caption(
            "Runtime–shape fidelity Pareto view",
            source="static_runs.csv; native extension diagnostics and trajectory metrics",
            units="milliseconds and metres",
            sample_count=len(values),
            missing_failed=_missing_summary(rows, "runtime_ms"),
            interpretation="Lower-left points jointly reduce runtime and guide deviation; this descriptive scan does not imply population significance.",
        ),
    )


def _rankings(
    frozen: Mapping[str, Any], output: Path, inputs: Sequence[Path]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best = {str(row["case_uid"]): row for row in frozen["best_ranking"]}
    worst = {str(row["case_uid"]): row for row in frozen["worst_ranking"]}
    values = [
        {
            "case_uid": uid,
            "best_rank": best[uid]["rank"],
            "worst_rank": worst[uid]["rank"],
            "classification": best[uid].get("classification", ""),
            "failure_reason": best[uid].get("failure_reason", ""),
        }
        for uid in sorted(set(best) & set(worst))
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    ax.scatter(
        [int(row["best_rank"]) for row in values],
        [int(row["worst_rank"]) for row in values],
        color="#CC79A7",
        alpha=0.8,
    )
    ax.set_xlabel("Best-oriented rank (1 is best)")
    ax.set_ylabel("Worst-oriented rank (1 is worst)")
    ax.set_title("Complete deterministic ranking audit")
    receipt = _bundle(
        output,
        inputs,
        "complete_ranking_overview",
        fig,
        values,
        scientific_caption(
            "Complete deterministic ranking audit",
            source="selected_dynamic_cases.json frozen rankings",
            units="ordinal rank",
            sample_count=len(values),
            missing_failed="Every eligible case is retained in both rankings; no rank is imputed.",
            interpretation="Opposed axes make Best2/Worst2 selection and backup ordering directly auditable.",
            profile="SuperPlanner 2-D SFC eligibility ranking",
        ),
    )
    return receipt, values


def _case_cards(
    rows: list[dict[str, str]],
    frozen: Mapping[str, Any],
    output: Path,
    inputs: Sequence[Path],
) -> dict[str, object]:
    selected = list(frozen.get("best2", [])) + list(frozen.get("worst2", []))
    lookup = {(row["case_uid"], row["profile"]): row for row in rows}
    backing = [
        {
            "case_uid": uid,
            "selection_group": "BEST" if uid in frozen.get("best2", []) else "WORST",
            "profile": profile,
            "min_normalized_margin": _number(lookup[(uid, profile)], "min_normalized_margin"),
            "guide_deviation_p95_m": _number(lookup[(uid, profile)], "guide_deviation_p95_m"),
            "runtime_ms": _number(lookup[(uid, profile)], "runtime_ms"),
            "classification": lookup[(uid, profile)]["classification"],
            "status": lookup[(uid, profile)]["status"],
        }
        for uid in selected
        for profile in PROFILES
        if (uid, profile) in lookup
    ]
    fig, axes = plt.subplots(
        max(1, len(selected)),
        1,
        figsize=(8.2, max(3.0, 1.65 * len(selected))),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, uid in zip(axes_array, selected):
        subset = [row for row in backing if row["case_uid"] == uid]
        axis.axis("off")
        cell_text = [
            [
                str(row["profile"]),
                str(row["classification"]),
                (
                    f"{float(row['min_normalized_margin']):.3e}"
                    if math.isfinite(float(row["min_normalized_margin"]))
                    else "NA (failed)"
                ),
                (
                    f"{float(row['guide_deviation_p95_m']):.3f}"
                    if math.isfinite(float(row["guide_deviation_p95_m"]))
                    else "NA"
                ),
                (
                    f"{float(row['runtime_ms']):.2f}"
                    if math.isfinite(float(row["runtime_ms"]))
                    else "NA"
                ),
            ]
            for row in subset
        ]
        table = axis.table(
            cellText=cell_text,
            colLabels=(
                "Profile",
                "Classification",
                "Min. margin (1)",
                "Guide dev. p95 (m)",
                "Runtime (ms)",
            ),
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=(0.16, 0.28, 0.18, 0.20, 0.16),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.3)
        for column in range(5):
            table[(0, column)].set_facecolor("#E5E7EB")
            table[(0, column)].set_text_props(weight="bold")
        for row_index, row in enumerate(subset, 1):
            colour = (
                "#E5E7EB"
                if row["profile"] == "legacy"
                else "#DBEAFE"
            )
            for column in range(5):
                table[(row_index, column)].set_facecolor(colour)
        group = "BEST" if uid in frozen.get("best2", []) else "WORST"
        group_values = (
            list(frozen.get("best2", []))
            if group == "BEST"
            else list(frozen.get("worst2", []))
        )
        rank = group_values.index(uid) + 1
        axis.set_title(f"{group}{rank} · {uid}", loc="left", fontsize=9, weight="bold")
    if not selected:
        axes_array[0].text(0.5, 0.5, "No frozen selected cases", ha="center")
    fig.suptitle("Frozen Best2/Worst2 paired case cards")
    return _bundle(
        output,
        inputs,
        "best_worst_case_cards",
        fig,
        backing,
        scientific_caption(
            "Frozen Best2/Worst2 paired case cards",
            source="static_runs.csv joined to selected_dynamic_cases.json",
            units="normalized margin is dimensionless; backing table also reports metres and milliseconds",
            sample_count=len(backing),
            missing_failed="Unavailable profile/case pairs are omitted and counted by the difference from 2×selected cases; failed values remain NaN.",
            interpretation="Paired bars expose how each frozen dynamic pilot case changes between legacy and native SuperPlanner 2-D SFC.",
        ),
    )


def _write_complete_rankings(output: Path, values: list[dict[str, object]]) -> None:
    path = output / "tables" / "complete_case_rankings.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["case_uid", "best_rank", "worst_rank", "classification", "failure_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _trajectory_cards(
    selection_dir: Path,
    rows: list[dict[str, str]],
    frozen: Mapping[str, Any],
    output: Path,
    base_inputs: Sequence[Path],
) -> tuple[list[dict[str, object]], list[str]]:
    lookup = {(row["case_uid"], row["profile"]): row for row in rows}
    selected = list(frozen.get("best2", [])) + list(frozen.get("worst2", []))
    receipts: list[dict[str, object]] = []
    rendered: list[str] = []
    for uid in selected:
        samples_path = (
            selection_dir / "selected_artifacts" / uid / "trajectory_samples.csv"
        )
        if not samples_path.is_file():
            continue
        samples = _read_csv(samples_path)
        fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
        safe_row = lookup.get((str(uid), "superplanner_sfc_v1"), {})
        try:
            obstacles = json.loads(str(safe_row.get("obstacle_layout", "[]")))
        except json.JSONDecodeError:
            obstacles = []
        obstacle_label_used = False
        for obstacle in obstacles:
            if not isinstance(obstacle, list) or len(obstacle) != 4:
                continue
            x0, y0, x1, y1 = map(float, obstacle)
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="#4B5563",
                    edgecolor="black",
                    alpha=0.6,
                    label="obstacle" if not obstacle_label_used else None,
                )
            )
            obstacle_label_used = True
        styles = {
            "guide": {"color": "#111827", "linestyle": "--", "linewidth": 1.3},
            "legacy": {"color": PAPER_COLORS["legacy"], "linestyle": ":", "linewidth": 2.0},
            "superplanner_sfc_v1": {"color": PAPER_COLORS["superplanner_sfc_v1"], "linestyle": "-", "linewidth": 2.0},
        }
        for profile in ("guide", "legacy", "superplanner_sfc_v1"):
            subset = sorted(
                (row for row in samples if row.get("profile") == profile),
                key=lambda row: int(row.get("sample_index", 0)),
            )
            if not subset:
                continue
            ax.plot(
                [_number(row, "x_m") for row in subset],
                [_number(row, "y_m") for row in subset],
                label=profile,
                **styles[profile],
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World y (m)")
        group = "Best2" if uid in frozen.get("best2", []) else "Worst2"
        ax.set_title(f"{group} paired trajectory · {uid}")
        ax.legend(loc="best", frameon=False)
        stem = f"trajectory_case_{uid}"
        receipts.append(
            _bundle(
                output,
                tuple(base_inputs) + (samples_path,),
                stem,
                fig,
                samples,
                scientific_caption(
                    f"{group} paired trajectory · {uid}",
                    source="native static trajectory_samples.csv with static_runs.csv obstacle metadata",
                    units="world-frame metres",
                    sample_count=len(samples),
                    missing_failed="Missing profile trajectories are not interpolated; failed profiles may contain no trajectory samples.",
                    interpretation="The shared axes directly compare the same guide, legacy output and native SuperPlanner 2-D SFC output for one frozen case.",
                ),
            )
        )
        rendered.append(str(uid))
    return receipts, rendered


def generate_static_paper_outputs(
    selection_dir: Path | str, output_dir: Path | str
) -> dict[str, object]:
    selection_dir = Path(selection_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        errors = validate_static_paper_outputs(output_dir)
        if not errors:
            return json.loads((output_dir / "static_paper_manifest.json").read_text(encoding="utf-8"))
        raise FileExistsError(
            "immutable paper output already exists but is incomplete: "
            + "; ".join(errors)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = selection_dir / "static_runs.csv"
    selection_path = selection_dir / "selected_dynamic_cases.json"
    if not runs_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("static paper outputs require runs CSV and selection JSON")
    rows = _read_csv(runs_path)
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("static_runs.csv contains no observations")
    if frozen.get("hot_start_evidence") != "PENDING_DYNAMIC_VALIDATION":
        raise ValueError("static selection must not claim dynamic hot-start evidence")
    apply_paper_style()
    inputs = (runs_path, selection_path)
    receipts = [
        _single_factor(rows, output_dir, inputs),
        _factor_heatmap(rows, output_dir, inputs),
        _transitions(rows, output_dir, inputs),
        _failure_stack(rows, output_dir, inputs),
        _pareto(rows, output_dir, inputs),
    ]
    ranking_receipt, ranking_rows = _rankings(frozen, output_dir, inputs)
    receipts.append(ranking_receipt)
    receipts.append(_case_cards(rows, frozen, output_dir, inputs))
    trajectory_receipts, trajectory_uids = _trajectory_cards(
        selection_dir, rows, frozen, output_dir, inputs
    )
    receipts.extend(trajectory_receipts)
    _write_complete_rankings(output_dir, ranking_rows)
    manifest = {
        "schema_version": 1,
        "figure_count": len(receipts),
        "selected_trajectory_case_uids": trajectory_uids,
        "source_inputs": [file_receipt(path, selection_dir) for path in inputs],
        "figure_receipts": receipts,
        "claims": {
            "data_driven_only": True,
            "hardcoded_experiment_values": False,
            "hot_start_evidence": "PENDING_DYNAMIC_VALIDATION",
        },
    }
    (output_dir / "static_paper_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory_path = output_dir / "artifact_receipt.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": ".",
                "artifacts": inventory_receipts(
                    output_dir, exclude=(inventory_path,)
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
