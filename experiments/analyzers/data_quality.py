from __future__ import annotations

import csv
import math
from pathlib import Path


NONFINITE_TEXT = {
    "nan", "+nan", "-nan",
    "inf", "+inf", "-inf",
    "infinity", "+infinity", "-infinity",
}


def _populated(value) -> bool:
    text = str(value).strip()
    if not text or text.lower() in NONFINITE_TEXT:
        return False
    try:
        number = float(text)
    except ValueError:
        return True
    return math.isfinite(number)


def summarize_csv_paths(
    paths,
    *,
    root: Path | str | None = None,
) -> list[dict]:
    """Summarize every declared field in explicit CSV evidence files."""

    root_path = Path(root).resolve() if root is not None else None
    summaries = []
    for path in sorted(Path(item).resolve() for item in paths):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            table_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        if root_path is None:
            source_file = str(path)
        else:
            try:
                source_file = path.relative_to(root_path).as_posix()
            except ValueError:
                source_file = str(path)
        count = len(table_rows)
        for field in fieldnames:
            populated = sum(_populated(row.get(field, "")) for row in table_rows)
            summaries.append(
                {
                    "source_file": source_file,
                    "table": path.stem,
                    "field": field,
                    "row_count": count,
                    "populated_count": populated,
                    "missing_count": count - populated,
                    "coverage_rate": populated / count if count else 0.0,
                }
            )
    return summaries


def summarize_field_coverage(run_dir: Path | str, write_output: bool = True) -> list[dict]:
    run_dir = Path(run_dir)
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            table_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        count = len(table_rows)
        for field in fieldnames:
            populated = sum(_populated(row.get(field, "")) for row in table_rows)
            rows.append({
                "table": path.stem,
                "field": field,
                "row_count": count,
                "populated_count": populated,
                "missing_count": count - populated,
                "coverage_rate": populated / count if count else 0.0,
            })
    if write_output:
        output_dir = run_dir / "data_quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "field_coverage.csv"
        fields = [
            "table", "field", "row_count", "populated_count",
            "missing_count", "coverage_rate",
        ]
        temporary = output_path.with_name(output_path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_path)
    return rows


def summarize_suite_field_coverage(
    suite_dir: Path | str,
    write_output: bool = True,
) -> list[dict]:
    """Aggregate field coverage across every run CSV in one experiment suite.

    A field absent from an older run schema is counted as missing for every row
    in that file. This makes schema drift visible instead of silently computing
    coverage only from newer files that happen to contain the field.
    """
    suite_dir = Path(suite_dir)
    table_paths: dict[str, list[Path]] = {}
    for path in sorted((suite_dir / "experiments").glob("**/run_*/*.csv")):
        table_paths.setdefault(path.stem, []).append(path)

    summaries = []
    for table, paths in sorted(table_paths.items()):
        files = []
        fields = set()
        for path in paths:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                table_rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            files.append((fieldnames, table_rows))
            fields.update(fieldnames)

        row_count = sum(len(table_rows) for _, table_rows in files)
        for field in sorted(fields):
            populated = sum(
                _populated(row.get(field, ""))
                for _, table_rows in files
                for row in table_rows
            )
            summaries.append({
                "table": table,
                "field": field,
                "file_count": len(paths),
                "row_count": row_count,
                "populated_count": populated,
                "missing_count": row_count - populated,
                "coverage_rate": populated / row_count if row_count else 0.0,
            })

    if write_output:
        output_dir = suite_dir / "reports" / "data_quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "field_coverage.csv"
        fields = [
            "table", "field", "file_count", "row_count",
            "populated_count", "missing_count", "coverage_rate",
        ]
        temporary = output_path.with_name(output_path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
        temporary.replace(output_path)
    return summaries
