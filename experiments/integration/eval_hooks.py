from __future__ import annotations

import time


class ExperimentHookBridge:
    """Fail-small monitor bridge; it observes events and never changes commands."""
    def __init__(self, sink, identity):
        self.sink = sink; self.identity = dict(identity); self.episode_uid = None; self.generation = -1; self._cycle = 0; self._frame = 0

    def _base(self): return {**self.identity, "episode_uid":self.episode_uid, "data_source":"REAL"}

    def start_episode(self, episode_uid, generation):
        self.episode_uid = str(episode_uid); self.generation = int(generation); self._cycle = 0; self._frame = 0
        self.sink.submit_csv("events", {**self._base(), "timestamp_monotonic_s":time.monotonic(), "frame_idx":0, "plan_uid":"", "event_type":"EPISODE_START", "severity":"INFO", "primary_reason":"", "secondary_reason":"", "message":f"generation={generation}"})

    def record_planning_cycle(self, cycle_uid, published, stale, fallback_mode, failure_reason="", **fields):
        row = {**self._base(), "episode_generation":self.generation, "planning_cycle_uid":cycle_uid, "trigger_timestamp_s":time.monotonic(), "raw_available":fields.get("raw_available", True), "candidate_count":fields.get("candidate_count", 0), "attempted_candidate_count":fields.get("attempted_candidate_count", 0), "selected_candidate_index":fields.get("selected_candidate_index", ""), "optimizer_success":fields.get("optimizer_success", ""), "cpp_validation_success":fields.get("cpp_validation_success", ""), "python_validation_success":fields.get("python_validation_success", ""), "stale":bool(stale), "published":bool(published), "fallback_mode":fallback_mode, "failure_reason":failure_reason, "navdp_ms":fields.get("navdp_ms", ""), "minco_ms":fields.get("minco_ms", ""), "validation_ms":fields.get("validation_ms", ""), "planning_total_ms":fields.get("planning_total_ms", ""), "plan_age_when_applied_ms":fields.get("plan_age_when_applied_ms", "")}
        self.sink.submit_csv("planning_cycles", row); self._cycle += 1

    def record_plan(self, plan_uid, published, stale, **fields):
        if not published or stale: return False
        self.sink.submit_csv("plan_metrics", {**self._base(), "plan_uid":plan_uid, "timestamp_monotonic_s":time.monotonic(), "plan_status":fields.get("plan_status", "PUBLISHED"), "fallback_mode":"NONE"})
        return True

    def record_control_step(self, frame_idx, plan_uid, cmd_v, cmd_w, **fields):
        self.sink.submit_csv("control_samples", {**self._base(), "frame_idx":frame_idx, "plan_uid":plan_uid, "timestamp_monotonic_s":time.monotonic(), "control_state":fields.get("control_state", "TRACK"), "cmd_v_mps":cmd_v, "cmd_w_radps":cmd_w}); self._frame += 1

    def end_episode(self, success=False, **fields):
        self.sink.submit_csv("episode_metrics", {**self._base(), "episode_index":fields.get("episode_index", 0), "success":bool(success), "collision":fields.get("collision", False), "timeout":fields.get("timeout", False), "done_reason":fields.get("done_reason", "GOAL_REACHED" if success else "UNKNOWN"), "planning_count":self._cycle})

    def reset(self, generation):
        self.episode_uid = None; self.generation = int(generation); self._cycle = 0; self._frame = 0
