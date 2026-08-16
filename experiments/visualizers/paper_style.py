from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.core.artifact_receipt import file_receipt, sha256_file


PAPER_COLORS = {
    "legacy": "#6B7280",
    "superplanner_sfc_v1": "#0072B2",
    "success": "#009E73",
    "failure": "#D55E00",
    "degraded": "#E69F00",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def scientific_caption(
    title: str,
    *,
    source: str,
    units: str,
    sample_count: int,
    missing_failed: str,
    interpretation: str,
    profile: str = "legacy paired with SuperPlanner 2-D SFC",
    paired_key: str = (
        "case_uid; profile is the within-case condition where applicable"
    ),
    denominator: str | None = None,
    limitations: str = (
        "Deterministic controlled static cases provide descriptive boundary "
        "evidence, not a population-level significance claim."
    ),
) -> str:
    denominator_text = denominator or (
        f"All n={sample_count} backing-data rows, with exclusions stated explicitly."
    )
    return (
        f"# {title}\n\n"
        f"- Source: {source}\n"
        f"- Profiles: {profile}\n"
        f"- Units: {units}\n"
        f"- Sample size: n={sample_count}\n"
        f"- Paired key: {paired_key}\n"
        f"- Denominator: {denominator_text}\n"
        f"- Missing/failed: {missing_failed}\n"
        f"- Limitations: {limitations}\n"
        f"- Interpretation: {interpretation}\n"
    )


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["no_observations"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _external_receipt(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def save_paper_figure(
    figure: plt.Figure,
    *,
    stem: str,
    output_dir: Path | str,
    backing_rows: Sequence[Mapping[str, object]],
    caption: str,
    input_paths: Iterable[Path | str],
) -> dict[str, object]:
    """Atomically define one traceable bitmap/vector/data/caption bundle."""
    output_dir = Path(output_dir).resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    caption_dir = output_dir / "captions"
    receipt_dir = output_dir / "receipts"
    for directory in (figure_dir, table_dir, caption_dir, receipt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    pdf = figure_dir / f"{stem}.pdf"
    table = table_dir / f"{stem}.csv"
    caption_path = caption_dir / f"{stem}.md"
    receipt_path = receipt_dir / f"{stem}.json"
    required_caption_fields = (
        "Source:",
        "Units:",
        "n=",
        "Paired key:",
        "Denominator:",
        "Missing/failed:",
        "Limitations:",
        "Interpretation:",
    )
    if any(field not in caption for field in required_caption_fields):
        raise ValueError("scientific caption is missing required provenance fields")
    inputs = [Path(path).resolve() for path in input_paths]
    if not inputs or any(not path.is_file() for path in inputs):
        raise ValueError("paper figure inputs must be existing files")
    try:
        figure.savefig(
            png,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "NavDP research workflow"},
        )
        figure.savefig(
            pdf,
            bbox_inches="tight",
            metadata={
                "Creator": "NavDP research workflow",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        _write_rows(table, backing_rows)
        caption_path.write_text(caption, encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "figure_stem": stem,
            "inputs": [_external_receipt(path) for path in inputs],
            "outputs": [
                file_receipt(path, output_dir)
                for path in (png, pdf, table, caption_path)
            ],
            "rendering": {
                "bitmap_dpi": 300,
                "vector_format": "PDF",
                "backing_data_format": "CSV",
            },
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt
    finally:
        plt.close(figure)
