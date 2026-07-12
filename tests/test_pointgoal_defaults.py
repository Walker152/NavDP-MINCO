import unittest
from pathlib import Path


class PointGoalDefaultsTests(unittest.TestCase):
    def test_defaults_match_launch_profile(self):
        source = Path("eval_pointgoal_wheeled.py").read_text(encoding="utf-8")
        expected_fragments = [
            'default="/home/alioth/NavDP/assets/scenes/cluttered_easy"',
            '"--scene_index", type=int, default=0',
            '"--num_episodes", type=int, default=10',
            '"--minco_top_k", type=int, default=1',
            '"--minco_safe_dist", type=float, default=0.60',
            '"--minco_max_vel", type=float, default=1.0',
            '"--minco_max_acc", type=float, default=2.0',
            '"--timing_log_interval", type=int, default=1',
            '"--mpc_max_wheel_speed",',
            'default=0.0,',
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, source)
        self.assertGreaterEqual(source.count("argparse.BooleanOptionalAction"), 2)
        self.assertIn('"--enable_minco", default=True', source)
        self.assertIn('"--show_timing_overlay", default=True', source)


if __name__ == "__main__":
    unittest.main()
