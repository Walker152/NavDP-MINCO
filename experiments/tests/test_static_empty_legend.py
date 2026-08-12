from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.static.runner import StaticRunResult
from experiments.static.synthetic import generate_catalogue
from experiments.visualizers.static_benchmark import render_static_case


class StaticEmptyLegendTests(unittest.TestCase):
    def test_failed_case_without_clearance_samples_emits_no_legend_warning(self):
        config = Path(__file__).parents[1] / "configs" / "static_legacy_suite.json"
        case = generate_catalogue(config)[0]
        result = StaticRunResult(
            case_uid=case.case_uid,
            case_hash=case.case_hash,
            mode="recompute",
            status="FAILED",
            engine="test",
            native_extension_path="",
            native_extension_sha256="",
            diagnostics={"success": False},
            samples=np.empty((0, 15), dtype=np.float64),
            waypoints=np.empty((0, 3), dtype=np.float64),
        )
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("matplotlib.legend")
        handler = Capture()
        logger.addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                render_static_case(
                    case,
                    result,
                    {"safe_dist_m": 0.15},
                    {},
                    Path(temporary),
                )
        finally:
            logger.removeHandler(handler)

        messages = [record.getMessage() for record in records]
        self.assertFalse(
            any("No artists with labels found" in message for message in messages),
            messages,
        )


if __name__ == "__main__":
    unittest.main()
