from __future__ import annotations

import sys
import time


class SuiteProgressReporter:
    def __init__(self, total_episodes, stream=None, heartbeat_interval_s=30.0):
        self.total_episodes = max(1, int(total_episodes)); self.stream = stream or sys.stdout
        self.completed_episodes = 0; self.current_expected = 0; self.current_local = -1
        self.current_label = ""; self.current_run = 0; self.total_runs = 0
        self.heartbeat_interval_s = max(1.0, float(heartbeat_interval_s))
        self.run_started_s = 0.0; self.last_report_s = 0.0

    def _write(self, message):
        print(message, file=self.stream, flush=True)

    def advance_completed(self, count):
        self.completed_episodes = min(self.total_episodes, self.completed_episodes + max(0, int(count)))

    def start_run(self, run_index, total_runs, variant, scene_id, expected_episodes, now=None):
        now = time.monotonic() if now is None else float(now)
        self.current_run = int(run_index); self.total_runs = int(total_runs)
        self.current_expected = int(expected_episodes); self.current_local = -1
        self.current_label = f"{str(variant).upper()} | {scene_id}"
        self.run_started_s = now; self.last_report_s = now
        self._write(f"[Progress] run {self.current_run}/{self.total_runs} START | {self.current_label} | {self.current_expected} episodes")

    def update_run(self, local_completed, now=None, process_id=None, log_age_s=None):
        now = time.monotonic() if now is None else float(now)
        local_completed = min(self.current_expected, max(0, int(local_completed)))
        if local_completed == self.current_local:
            if now - self.last_report_s < self.heartbeat_interval_s:
                return
            details = [f"elapsed={int(max(0.0, now - self.run_started_s))}s"]
            if process_id is not None:
                details.append(f"pid={int(process_id)}")
            if log_age_s is not None:
                details.append(f"log_age={max(0.0, float(log_age_s)):.1f}s")
            self._write(
                f"[Progress] HEARTBEAT | run {self.current_run}/{self.total_runs} | "
                f"{self.current_label} | episode {local_completed}/{self.current_expected} | "
                + " ".join(details)
            )
            self.last_report_s = now
            return
        self.current_local = local_completed
        self.last_report_s = now
        global_completed = min(self.total_episodes, self.completed_episodes + local_completed)
        percent = 100.0 * global_completed / self.total_episodes
        self._write(
            f"[Progress] {global_completed}/{self.total_episodes} episodes | {percent:.1f}% | "
            f"run {self.current_run}/{self.total_runs} | {self.current_label} | "
            f"episode {local_completed}/{self.current_expected}"
        )

    def finish_run(self):
        self.update_run(self.current_expected); self.advance_completed(self.current_expected)
        self._write(f"[Progress] run {self.current_run}/{self.total_runs} COMPLETE | {self.current_label}")

    def skip_completed_run(self):
        self.update_run(self.current_expected); self.advance_completed(self.current_expected)
        self._write(f"[Progress] run {self.current_run}/{self.total_runs} SKIP_COMPLETE | {self.current_label}")

    def fail_run(self, error):
        self._write(f"[Progress] run {self.current_run}/{self.total_runs} FAILED | {self.current_label} | {error!r}")
