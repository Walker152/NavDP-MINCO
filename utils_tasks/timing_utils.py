import time
from contextlib import contextmanager

import cv2
import numpy as np


class StageTimer:
    def __init__(self, clock=None):
        self.clock = clock or time.perf_counter
        self.records = {}

    def now(self):
        return self.clock()

    def elapsed_ms(self, start):
        return (self.clock() - start) * 1000.0

    @contextmanager
    def section(self, name):
        start = self.clock()
        try:
            yield
        finally:
            self.records[name] = self.elapsed_ms(start)

    def snapshot(self):
        return dict(self.records)


def mean_timing(records):
    records = list(records)
    if not records:
        return {}
    keys = set()
    for record in records:
        keys.update(record.keys())
    return {
        key: float(np.mean([record.get(key, 0.0) for record in records]))
        for key in keys
    }


def format_planning_summary(timing):
    return (
        "[Timing][Planning] "
        f"total={timing.get('planning_total_ms', 0.0):.2f}ms "
        f"navdp={timing.get('navdp_step_ms', 0.0):.2f}ms "
        f"transform_raw={timing.get('raw_transform_ms', 0.0):.2f}ms "
        f"transform_candidates={timing.get('candidate_transform_ms', 0.0):.2f}ms "
        f"state={timing.get('state_build_ms', 0.0):.2f}ms "
        f"minco={timing.get('minco_total_ms', 0.0):.2f}ms "
        f"mpc_construct={timing.get('mpc_construct_ms', 0.0):.2f}ms"
    )


def format_minco_summary(env_i, info):
    return (
        "[Timing][MINCO] "
        f"env={env_i} success={info.get('success', False)} fallback={info.get('fallback', False)} "
        f"adapter={info.get('adapter_total_ms', np.nan):.2f}ms "
        f"cpp={info.get('selected_cpp_optimize_time_ms', np.nan):.2f}ms "
        f"selected_idx={info.get('selected_index', -1)}"
    )


def format_control_summary(timing):
    return (
        "[Timing][Control] "
        f"vis={timing.get('visualize_ms', 0.0):.2f}ms "
        f"mpc_solve={timing.get('mpc_solve_ms', 0.0):.2f}ms "
        f"speed_plot={timing.get('speed_plot_ms', 0.0):.2f}ms "
        f"text={timing.get('text_overlay_ms', 0.0):.2f}ms "
        f"video={timing.get('video_write_ms', 0.0):.2f}ms "
        f"env_step={timing.get('env_step_ms', 0.0):.2f}ms"
    )


def draw_timing_overlay(image, planning_timing, control_timing):
    if planning_timing is None and control_timing is None:
        return image
    y = 20
    lines = []
    if planning_timing is not None:
        lines.append("PLAN total %.1f navdp %.1f minco %.1f" % (
            planning_timing.get("planning_total_ms", 0.0),
            planning_timing.get("navdp_step_ms", 0.0),
            planning_timing.get("minco_total_ms", 0.0),
        ))
        lines.append("TF raw %.1f cand %.1f mpc_build %.1f" % (
            planning_timing.get("raw_transform_ms", 0.0),
            planning_timing.get("candidate_transform_ms", 0.0),
            planning_timing.get("mpc_construct_ms", 0.0),
        ))
    if control_timing is not None:
        lines.append("CTRL vis %.1f mpc %.1f video %.1f env %.1f" % (
            control_timing.get("visualize_ms", 0.0),
            control_timing.get("mpc_solve_ms", 0.0),
            control_timing.get("video_write_ms", 0.0),
            control_timing.get("env_step_ms", 0.0),
        ))
    for line in lines:
        cv2.putText(image, line, (450, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return image
