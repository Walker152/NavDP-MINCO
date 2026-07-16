from __future__ import annotations

import csv
from pathlib import Path
import queue
import threading
import numpy as np


class AsyncRecordWriter:
    def __init__(
        self,
        output_dir: Path,
        schemas: dict[str, list[str]],
        flush_rows: int = 256,
        queue_size: int = 8192,
        immediate_flush_tables=("episode_metrics",),
    ):
        self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schemas = schemas; self.flush_rows = flush_rows; self._queue = queue.Queue(queue_size)
        self.immediate_flush_tables = frozenset(immediate_flush_tables)
        self._error = None; self._closed = False; self._handles = {}; self._writers = {}; self._counts = {}
        self._thread = threading.Thread(target=self._run, name="experiment-writer", daemon=False); self._thread.start()

    def _check(self):
        if self._error: raise RuntimeError("writer thread failed") from self._error
        if self._closed: raise RuntimeError("writer is closed")

    def submit_csv(self, table_name, row):
        self._check()
        if table_name not in self.schemas: raise KeyError(table_name)
        extras = set(row) - set(self.schemas[table_name])
        if extras: raise ValueError(f"unexpected fields for {table_name}: {sorted(extras)}")
        self._queue.put(("csv", table_name, dict(row)))

    def submit_npz(self, relative_path, arrays):
        self._check(); self._queue.put(("npz", relative_path, dict(arrays)))

    def _csv(self, table):
        if table not in self._writers:
            path = self.output_dir / f"{table}.csv"; path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=self.schemas[table]); writer.writeheader()
            self._handles[table] = handle; self._writers[table] = writer; self._counts[table] = 0
        return self._writers[table]

    def _run(self):
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None: return
                    kind, name, payload = item
                    if kind == "csv":
                        writer = self._csv(name); writer.writerow({field: payload.get(field, "") for field in self.schemas[name]}); self._counts[name] += 1
                        if (
                            name in self.immediate_flush_tables
                            or self._counts[name] % self.flush_rows == 0
                        ):
                            self._handles[name].flush()
                    else:
                        path = self.output_dir / name; path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_name(path.name + ".tmp")
                        with temporary.open("wb") as stream: np.savez_compressed(stream, **payload)
                        temporary.replace(path)
                finally: self._queue.task_done()
        except BaseException as error:
            self._error = error

    def flush(self):
        self._check(); self._queue.join()
        for handle in self._handles.values(): handle.flush()

    def wait_pending(self):
        self._check()
        self._queue.join()

    def close(self):
        if self._closed: return
        if self._error: self._closed = True; raise RuntimeError("writer thread failed") from self._error
        self._queue.put(None); self._thread.join()
        for handle in self._handles.values(): handle.close()
        self._closed = True
        if self._error: raise RuntimeError("writer thread failed") from self._error
