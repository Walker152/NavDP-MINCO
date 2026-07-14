import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.core.schemas import SCHEMAS
from experiments.core.trace_schema import TRACE_SCHEMA_VERSION, validate_trace
from experiments.recorders.trace_writer import PlanningTraceWriter


class PipelineContractTests(unittest.TestCase):
    def test_planning_cycle_schema_has_failure_denominator_fields(self):
        required = {
            "planning_cycle_uid", "episode_generation", "raw_available", "candidate_count",
            "attempted_candidate_count", "optimizer_success", "cpp_validation_success",
            "python_validation_success", "stale", "published", "fallback_mode", "failure_reason",
            "navdp_ms", "minco_ms", "validation_ms", "planning_total_ms", "plan_age_when_applied_ms",
        }
        self.assertTrue(required.issubset(SCHEMAS["planning_cycles"]))

    def test_trace_writer_emits_versioned_npz_and_metadata(self):
        root = Path(tempfile.mkdtemp())
        writer = PlanningTraceWriter(root)
        arrays = {
            "raw_path_xy": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "robot_state": np.zeros(5),
            "goal": np.array([1.0, 0.0]),
        }
        npz_path, metadata_path = writer.write("cycle-1", arrays)
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata["schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(metadata["arrays"]["raw_path_xy"]["shape"], [2, 2])
        self.assertEqual(validate_trace(npz_path, metadata_path), [])
        self.assertFalse(list(root.rglob("*.tmp")))
