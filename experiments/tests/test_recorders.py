import csv
import json
import tempfile
import unittest

import numpy as np
from experiments.recorders.async_writer import AsyncRecordWriter
from experiments.recorders.run_recorder import RunLifecycle


class RecorderTests(unittest.TestCase):
 def test_async_writer_flushes_rows_and_npz(self):
    tmp_path = __import__('pathlib').Path(tempfile.mkdtemp())
    writer = AsyncRecordWriter(tmp_path, {"items": ["id", "value"]}, flush_rows=2)
    for i in range(5):
        writer.submit_csv("items", {"id": i, "value": i * 2})
    writer.submit_npz("traces/x.npz", {"x": np.arange(3)})
    writer.close()
    with (tmp_path / "items.csv").open() as stream:
        self.assertEqual(len(list(csv.DictReader(stream))), 5)
    self.assertEqual(np.load(tmp_path / "traces/x.npz")["x"].tolist(), [0, 1, 2])
    self.assertFalse(list(tmp_path.rglob("*.tmp")))


 def test_async_writer_rejects_extra_fields(self):
    tmp_path = __import__('pathlib').Path(tempfile.mkdtemp())
    writer = AsyncRecordWriter(tmp_path, {"items": ["id"]})
    with self.assertRaises(ValueError):
        writer.submit_csv("items", {"id": 1, "extra": 2})
    writer.close()


 def test_lifecycle_writes_atomic_status_and_rejects_invalid_transition(self):
    tmp_path = __import__('pathlib').Path(tempfile.mkdtemp())
    lifecycle = RunLifecycle(tmp_path)
    lifecycle.transition("RUNNING")
    with self.assertRaises(ValueError):
        lifecycle.transition("COMPLETE")
    self.assertEqual(json.loads((tmp_path / "run_status.json").read_text())["status"], "RUNNING")
