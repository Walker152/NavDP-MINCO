import unittest

from experiments.core.layout import ResultLayout
from experiments.core.models import EpisodeSpec, RunSpec


class CoreTests(unittest.TestCase):
    def test_episode_uid_is_independent_of_variant(self):
        episode = EpisodeSpec("sparse_0", "SPARSE", "SCN-01", 0, 7, (0, 0, 0), (3, 1, 0))
        self.assertEqual(episode.episode_uid, EpisodeSpec(**{**episode.as_dict(), "episode_uid": episode.episode_uid}).episode_uid)


    def test_result_layout_is_grouped_by_experiment_scene_variant_seed(self):
        from pathlib import Path
        run = RunSpec("suite", "EXP-01_raw_profile", "raw", "cold", "SPARSE", "sparse_0", 7, "run-0")
        path = ResultLayout(Path("/tmp/result-root")).run_dir(run)
        self.assertEqual(path.relative_to("/tmp/result-root").parts, (
            "suite", "experiments", "EXP-01_raw_profile", "SPARSE", "raw", "7", "run-0"))
