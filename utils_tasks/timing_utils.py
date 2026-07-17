import time
from contextlib import contextmanager

import cv2
import numpy as np

COLOR_PLANNING = (80, 200, 255)
COLOR_MINCO = (255, 220, 80)
COLOR_CONTROL = (100, 255, 140)
COLOR_IO = (255, 140, 180)
COLOR_TOTAL = (255, 255, 255)
COLOR_MUTED = (170, 170, 170)
COLOR_PANEL_BG = (18, 18, 18)


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
    timing_ms = info.get("timing_ms", {}) or {}
    adapter_timing = info.get("adapter_timing_ms", {}) or {}
    return (
        "[Timing][MINCO] "
        f"env={env_i} status={info.get('status', 'MINCO_OK' if info.get('success', False) else 'MINCO_STOP')} "
        f"adapter={info.get('adapter_total_ms', np.nan):.2f}ms "
        f"cpp={info.get('selected_cpp_optimize_time_ms', np.nan):.2f}ms "
        f"screen={adapter_timing.get('candidate_screen_ms', np.nan):.2f}ms "
        f"attempts={info.get('attempted_candidate_count', 0)}/{info.get('configured_top_k', 0)} "
        f"cpp_total={adapter_timing.get('candidate_cpp_total_ms', np.nan):.2f}ms "
        f"py_validate={adapter_timing.get('python_validation_total_ms', np.nan):.2f}ms "
        f"optimizer={timing_ms.get('optimizer_ms', np.nan):.2f}ms "
        f"validate={timing_ms.get('validate_ms', np.nan):.2f}ms "
        f"sparse_n={info.get('sparse_waypoint_size', 0)} "
        f"mandatory_corners={info.get('mandatory_corner_count', 0)} "
        f"iter={info.get('optimizer_iteration_count', 0)} "
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


def _fmt_ms(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{value:.1f}"


def _fmt_count(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "--"
    return str(value)


def _sum_optional(*values):
    total = 0.0
    seen = False
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            total += value
            seen = True
    return total if seen else float("nan")


def _minco_info_for_env(planning_timing, env_index):
    if not isinstance(planning_timing, dict):
        return {}
    results = planning_timing.get("minco_results")
    if isinstance(results, (list, tuple)) and 0 <= env_index < len(results):
        info = results[env_index]
        return info if isinstance(info, dict) else {}
    return {}


def append_timing_panel(image, planning_timing, control_timing, env_index=0):
    if planning_timing is None and control_timing is None:
        return image

    planning_timing = planning_timing or {}
    control_timing = control_timing or {}
    minco_info = _minco_info_for_env(planning_timing, env_index)
    adapter_timing = minco_info.get("adapter_timing_ms", {}) or {}
    transform_ms = _sum_optional(
        planning_timing.get("raw_transform_ms"),
        planning_timing.get("candidate_transform_ms"),
    )
    loop_total = _sum_optional(
        control_timing.get("mpc_solve_ms"),
        control_timing.get("visualize_ms"),
        control_timing.get("speed_plot_ms"),
        control_timing.get("text_overlay_ms"),
        control_timing.get("video_write_ms"),
        control_timing.get("env_step_ms"),
    )

    rows = [
        (
            "Planning",
            (
                f"total {_fmt_ms(planning_timing.get('planning_total_ms'))} ms | "
                f"NavDP {_fmt_ms(planning_timing.get('navdp_step_ms'))} | "
                f"transform {_fmt_ms(transform_ms)} | "
                f"input copy {_fmt_ms(planning_timing.get('input_copy_ms'))}"
            ),
            COLOR_PLANNING,
        ),
        (
            "MINCO",
            (
                f"total {_fmt_ms(planning_timing.get('minco_total_ms'))} ms | "
                f"screen {_fmt_ms(adapter_timing.get('candidate_screen_ms'))} | "
                f"attempts {_fmt_count(minco_info.get('attempted_candidate_count'))}/"
                f"{_fmt_count(minco_info.get('configured_top_k'))} | "
                f"calls {_fmt_ms(adapter_timing.get('candidate_attempt_total_ms'))}"
            ),
            COLOR_MINCO,
        ),
        (
            "MINCO detail",
            (
                f"C++ total {_fmt_ms(adapter_timing.get('candidate_cpp_total_ms'))} ms | "
                f"optimizer {_fmt_ms((minco_info.get('timing_ms') or {}).get('optimizer_ms'))} | "
                f"Py validate {_fmt_ms(adapter_timing.get('python_validation_total_ms'))} | "
                f"iter {_fmt_count(minco_info.get('optimizer_iteration_count'))}"
            ),
            COLOR_MINCO,
        ),
        (
            "Control",
            (
                f"MPC {_fmt_ms(control_timing.get('mpc_solve_ms'))} ms | "
                f"visualize {_fmt_ms(control_timing.get('visualize_ms'))} | "
                f"speed plot {_fmt_ms(control_timing.get('speed_plot_ms'))} | "
                f"text {_fmt_ms(control_timing.get('text_overlay_ms'))}"
            ),
            COLOR_CONTROL,
        ),
        (
            "Runtime",
            (
                f"video {_fmt_ms(control_timing.get('video_write_ms'))} ms | "
                f"env step {_fmt_ms(control_timing.get('env_step_ms'))} | "
                f"loop/control total {_fmt_ms(loop_total)}"
            ),
            COLOR_IO,
        ),
    ]

    panel_height = max(104, 18 + len(rows) * 24)
    panel = np.zeros((panel_height, image.shape[1], 3), dtype=np.uint8)
    panel[:] = COLOR_PANEL_BG

    x_label = 16
    x_text = 118
    y = 24
    for label, text, color in rows:
        cv2.putText(panel, label, (x_label, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(panel, text, (x_text, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TOTAL, 1, cv2.LINE_AA)
        y += 24

    return np.concatenate([image, panel], axis=0)


def draw_timing_overlay(image, planning_timing, control_timing):
    return append_timing_panel(image, planning_timing, control_timing)
