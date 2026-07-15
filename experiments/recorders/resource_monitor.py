from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time

import psutil

from experiments.recorders.run_recorder import atomic_json


FIELDS = (
    "timestamp_utc", "elapsed_s", "owned_process_count", "owned_cpu_percent",
    "owned_rss_bytes", "system_cpu_percent", "system_memory_used_bytes",
    "gpu_memory_mib", "gpu_utilization_percent", "gpu_status",
)


class ResourceMonitor:
    def __init__(self, run_dir, process_supplier, interval_s=1.0, command_runner=subprocess.run):
        self.run_dir = Path(run_dir); self.process_supplier = process_supplier
        self.interval_s = float(interval_s); self.command_runner = command_runner
        self.samples_path = self.run_dir / 'resource_samples.csv'
        self.summary_path = self.run_dir / 'resource_summary.json'
        self._stop = threading.Event(); self._thread = None; self._stream = None; self._writer = None
        self._start_monotonic = None; self._start_utc = None; self._peaks = {
            "peak_owned_cpu_percent":0.0, "peak_owned_rss_bytes":0,
            "peak_system_memory_used_bytes":0, "peak_gpu_memory_mib":None,
            "peak_gpu_utilization_percent":None,
        }

    def start(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._stream = self.samples_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=FIELDS); self._writer.writeheader(); self._stream.flush()
        self._start_monotonic = time.monotonic(); self._start_utc = datetime.now(timezone.utc)
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="experiment-resource-monitor", daemon=False)
        self._thread.start(); return self

    def _owned_processes(self):
        owned = {}
        for process in list(self.process_supplier()):
            pid = getattr(process, "pid", None)
            if not pid: continue
            try:
                root = psutil.Process(pid); owned[root.pid] = root
                for child in root.children(recursive=True): owned[child.pid] = child
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        return owned

    def _gpu_sample(self, owned_pids):
        try:
            apps = self.command_runner(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=3,
            )
            gpu = self.command_runner(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=3,
            )
            if apps.returncode != 0 or gpu.returncode != 0: return None, None, "unavailable"
            memory = 0.0
            for line in apps.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and int(parts[0]) in owned_pids: memory += float(parts[1])
            utilization = max((float(line.strip()) for line in gpu.stdout.splitlines() if line.strip()), default=0.0)
            return memory, utilization, "available"
        except (OSError, ValueError, subprocess.SubprocessError):
            return None, None, "unavailable"

    def _sample(self):
        processes = self._owned_processes(); rss = 0; cpu = 0.0
        for process in processes.values():
            try:
                rss += int(process.memory_info().rss); cpu += float(process.cpu_percent(interval=None))
            except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        system_memory = int(psutil.virtual_memory().used)
        gpu_memory, gpu_utilization, gpu_status = self._gpu_sample(set(processes))
        row = {
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "elapsed_s":time.monotonic() - self._start_monotonic,
            "owned_process_count":len(processes), "owned_cpu_percent":cpu,
            "owned_rss_bytes":rss, "system_cpu_percent":psutil.cpu_percent(interval=None),
            "system_memory_used_bytes":system_memory,
            "gpu_memory_mib":"" if gpu_memory is None else gpu_memory,
            "gpu_utilization_percent":"" if gpu_utilization is None else gpu_utilization,
            "gpu_status":gpu_status,
        }
        self._writer.writerow(row); self._stream.flush()
        self._peaks["peak_owned_cpu_percent"] = max(self._peaks["peak_owned_cpu_percent"], cpu)
        self._peaks["peak_owned_rss_bytes"] = max(self._peaks["peak_owned_rss_bytes"], rss)
        self._peaks["peak_system_memory_used_bytes"] = max(self._peaks["peak_system_memory_used_bytes"], system_memory)
        if gpu_memory is not None: self._peaks["peak_gpu_memory_mib"] = max(self._peaks["peak_gpu_memory_mib"] or 0.0, gpu_memory)
        if gpu_utilization is not None: self._peaks["peak_gpu_utilization_percent"] = max(self._peaks["peak_gpu_utilization_percent"] or 0.0, gpu_utilization)

    def _loop(self):
        while not self._stop.wait(self.interval_s): self._sample()

    def stop(self, status="finished", error=None):
        if self._thread is None: return None
        self._stop.set(); self._thread.join(timeout=max(5.0, self.interval_s * 2.0))
        self._sample(); end_utc = datetime.now(timezone.utc)
        self._stream.close(); self._stream = None; self._thread = None
        summary = {
            "schema_version":1, "status":status, "error":None if error is None else repr(error),
            "started_at_utc":self._start_utc.isoformat(), "ended_at_utc":end_utc.isoformat(),
            "duration_s":time.monotonic() - self._start_monotonic,
            "sample_interval_s":self.interval_s, **self._peaks,
            "gpu_status":"available" if self._peaks["peak_gpu_memory_mib"] is not None else "unavailable",
        }
        atomic_json(self.summary_path, summary); return summary
