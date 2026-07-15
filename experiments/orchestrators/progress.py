from __future__ import annotations

import sys


class SuiteProgressReporter:
    def __init__(self, total_episodes, stream=None):
        self.total_episodes = max(1, int(total_episodes)); self.stream = stream or sys.stdout
        self.completed_episodes = 0; self.current_expected = 0; self.current_local = -1
        self.current_label = ""; self.current_run = 0; self.total_runs = 0

    def _write(self, message):
        print(message, file=self.stream, flush=True)

    def advance_completed(self, count):
        self.completed_episodes = min(self.total_episodes, self.completed_episodes + max(0, int(count)))

    def start_run(self, run_index, total_runs, variant, scene_id, expected_episodes):
        self.current_run = int(run_index); self.total_runs = int(total_runs)
        self.current_expected = int(expected_episodes); self.current_local = -1
        self.current_label = f"{str(variant).upper()} | {scene_id}"
        self._write(f"[Progress] run {self.current_run}/{self.total_runs} START | {self.current_label} | {self.current_expected} episodes")

    def update_run(self, local_completed):
        local_completed = min(self.current_expected, max(0, int(local_completed)))
        if local_completed == self.current_local: return
        self.current_local = local_completed
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
