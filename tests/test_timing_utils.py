import unittest

import numpy as np

from utils_tasks.timing_utils import (
    StageTimer,
    format_control_summary,
    format_minco_summary,
    format_planning_summary,
    mean_timing,
)


class FakeClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


class TimingUtilsTest(unittest.TestCase):
    def test_stage_timer_records_section_ms_with_injected_clock(self):
        timer = StageTimer(clock=FakeClock([1.0, 1.025]))

        with timer.section("navdp_step_ms"):
            pass

        self.assertAlmostEqual(timer.records["navdp_step_ms"], 25.0)
        self.assertAlmostEqual(timer.snapshot()["navdp_step_ms"], 25.0)

    def test_mean_timing_averages_missing_keys_as_zero(self):
        averaged = mean_timing([
            {"video_write_ms": 2.0, "mpc_solve_ms": 4.0},
            {"video_write_ms": 4.0},
        ])

        self.assertEqual(averaged["video_write_ms"], 3.0)
        self.assertEqual(averaged["mpc_solve_ms"], 2.0)

    def test_summary_formatters_keep_eval_prints_consistent(self):
        planning = {
            "planning_total_ms": 10.0,
            "navdp_step_ms": 2.0,
            "raw_transform_ms": 1.0,
            "candidate_transform_ms": 3.0,
            "state_build_ms": 0.5,
            "minco_total_ms": 4.0,
            "mpc_construct_ms": 1.5,
        }
        control = {
            "visualize_ms": 1.0,
            "mpc_solve_ms": 2.0,
            "speed_plot_ms": 3.0,
            "text_overlay_ms": 4.0,
            "video_write_ms": 5.0,
            "env_step_ms": 6.0,
        }
        minco = {
            "success": True,
            "fallback": False,
            "adapter_total_ms": 7.0,
            "selected_cpp_optimize_time_ms": np.nan,
            "selected_index": 1,
        }

        self.assertIn("[Timing][Planning]", format_planning_summary(planning))
        self.assertIn("video=5.00ms", format_control_summary(control))
        self.assertIn("selected_idx=1", format_minco_summary(0, minco))


if __name__ == "__main__":
    unittest.main()
