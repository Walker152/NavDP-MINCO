#!/usr/bin/env python3
"""Compatibility entry for the data-driven NavDP paper report generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyzers.paper_report import generate_paper_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate traceable NavDP paper figures from experiment tables"
    )
    parser.add_argument("--input", required=True, help="experiment or workflow root")
    parser.add_argument("--output", required=True, help="new immutable paper directory")
    args = parser.parse_args(argv)
    result = generate_paper_report(args.input, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
