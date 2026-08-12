from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.tests.test_rolling_models import make_cycle


def make_result(cycle_count: int = 3):
    from experiments.rolling.models import RolloutResult

    cycles = []
    x = 0.0
    for index in range(cycle_count):
        cycle = make_cycle(index=index, input_x=x, executed_end_x=x + 0.25)
        cycles.append(cycle)
        x += 0.25
    executed = np.concatenate(
        [cycles[0].executed_samples, *[cycle.executed_samples[1:] for cycle in cycles[1:]]]
    )
    return RolloutResult(
        scenario_uid="scenario-a",
        method="safe_corridor_v1",
        status="MAX_CYCLES",
        cycles=tuple(cycles),
        executed_samples=executed,
        final_goal_xyz=[3.0, 0.0, 0.0],
        metrics={"final_error_m": 3.0 - x, "minimum_clearance_m": None},
        goal_tolerance_m=0.1,
    )


class RollingSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()) / "rollout"

    def test_rollout_round_trip_preserves_every_cycle_and_canonical_artifacts(self):
        from experiments.rolling.serialization import (
            load_rollout_result,
            validate_rollout_result,
            write_rollout_result,
        )

        receipt = write_rollout_result(make_result(3), self.root)
        loaded = load_rollout_result(receipt.manifest_path)
        self.assertEqual(len(loaded.cycles), 3)
        self.assertEqual(validate_rollout_result(self.root), [])
        self.assertEqual(
            {path.name for path in self.root.iterdir()},
            {
                "run_manifest.json",
                "cycle_metrics.csv",
                "executed_trajectory.csv",
                "candidate_trajectories.npz",
                "corridor_segments.csv",
                "obstacle_states.csv",
                "metrics.json",
                "artifact_receipt.json",
            },
        )
        with (self.root / "cycle_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(row["scenario_uid"] == "scenario-a" for row in rows))
        self.assertTrue(all(row["method"] == "safe_corridor_v1" for row in rows))
        self.assertIsNone(json.loads((self.root / "metrics.json").read_text())["metrics"]["minimum_clearance_m"])

    def test_changed_csv_or_npz_fails_closed(self):
        from experiments.rolling.serialization import validate_rollout_result, write_rollout_result

        write_rollout_result(make_result(), self.root)
        with (self.root / "cycle_metrics.csv").open("a", encoding="utf-8") as stream:
            stream.write("tamper")
        self.assertTrue(any("hash" in error for error in validate_rollout_result(self.root)))

        other = self.root.parent / "npz_tamper"
        write_rollout_result(make_result(), other)
        with (other / "candidate_trajectories.npz").open("ab") as stream:
            stream.write(b"tamper")
        self.assertTrue(any("hash" in error for error in validate_rollout_result(other)))

    def test_writer_rejects_invalid_result_and_nonempty_output(self):
        from experiments.rolling.models import RolloutResult
        from experiments.rolling.serialization import write_rollout_result

        invalid = RolloutResult(
            scenario_uid="scenario-a",
            method="legacy",
            status="GOAL_REACHED",
            cycles=(make_cycle(executed_end_x=0.25),),
            executed_samples=make_cycle(executed_end_x=0.25).executed_samples,
            final_goal_xyz=[9, 0, 0],
            metrics={},
            goal_tolerance_m=0.1,
        )
        with self.assertRaisesRegex(ValueError, "goal tolerance"):
            write_rollout_result(invalid, self.root)
        self.root.mkdir(parents=True)
        (self.root / "owned.txt").write_text("user", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "nonempty"):
            write_rollout_result(make_result(), self.root)


if __name__ == "__main__":
    unittest.main()
