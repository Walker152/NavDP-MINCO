from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


TRANSITIONS = {
    "CREATED": {"RUNNING", "FAILED"}, "RUNNING": {"SIMULATION_COMPLETE", "FAILED", "INTERRUPTED"},
    "SIMULATION_COMPLETE": {"VALIDATING", "FAILED"}, "VALIDATING": {"COMPLETE", "FAILED"},
    "COMPLETE": set(), "FAILED": set(), "INTERRUPTED": {"RUNNING", "FAILED"},
}


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(path)


class RunLifecycle:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir); self.run_dir.mkdir(parents=True, exist_ok=True); self.status = "CREATED"
        status_path = self.run_dir / "run_status.json"
        if status_path.exists(): self.status = json.loads(status_path.read_text())["status"]
        else: self._write()

    def _write(self, **extra):
        atomic_json(self.run_dir / "run_status.json", {"status": self.status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra})

    def transition(self, status: str, **extra):
        if status not in TRANSITIONS.get(self.status, set()): raise ValueError(f"invalid run transition {self.status} -> {status}")
        self.status = status; self._write(**extra)
