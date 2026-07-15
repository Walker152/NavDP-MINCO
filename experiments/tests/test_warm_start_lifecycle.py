from pathlib import Path
import unittest

from experiments.integration.warm_start import ProposalLifecycle, decide_warm_start


class WarmStartLifecycleTests(unittest.TestCase):
    def test_previews_share_snapshot_and_do_not_mutate_history(self):
        store = ProposalLifecycle(); store.reset_history(env_id=0, episode_generation=3)
        first = store.optimize_preview(0, 3, {"samples":[1]}, safe=True)
        second = store.optimize_preview(0, 3, {"samples":[2]}, safe=True)
        self.assertEqual(first["history_snapshot_id"], second["history_snapshot_id"])
        self.assertIsNone(store.committed_history(0))

    def test_commit_only_accepts_selected_safe_published_nonstale_proposal(self):
        store = ProposalLifecycle(); store.reset_history(0, 1)
        unsafe = store.optimize_preview(0, 1, {}, safe=False)
        with self.assertRaises(ValueError): store.commit_history(0, unsafe["proposal_id"], selected=True, published=True, stale=False)
        valid = store.optimize_preview(0, 1, {"samples":[1]}, safe=True)
        with self.assertRaises(ValueError): store.commit_history(0, valid["proposal_id"], selected=False, published=True, stale=False)
        store.commit_history(0, valid["proposal_id"], selected=True, published=True, stale=False)
        self.assertEqual(store.committed_history(0)["proposal_id"], valid["proposal_id"])
        store.reset_history(0, 2); self.assertIsNone(store.committed_history(0))

    def test_velocity_error_rejects_hot_start(self):
        decision = decide_warm_start(has_history=True, episode_matches=True, history_safe=True, remaining_duration=2., age=.1, position_error=.1, velocity_error=2., direction_dot=.9, shifted_seed_valid=True)
        self.assertEqual(decision, ("COLD_START", "HOT_REJECT_VELOCITY"))

    def test_cpp_binding_declares_preview_commit_discard_reset(self):
        source = Path("minco_processor/bindings/minco_pybind.cpp").read_text()
        for name in ("optimize_preview", "commit_history", "discard_proposal", "reset_history"):
            self.assertIn(name, source)
        cpp = Path("minco_processor/src/minco_processor/minco_pipeline.cpp").read_text()
        self.assertNotIn("if ((current.velocity - pred_vel).norm() > 1.0) {\n    return PlanningState::kHotStart;", cpp)
        self.assertNotIn('out["history_age_s"] = std::numeric_limits<double>::quiet_NaN()', source)
        self.assertNotIn('out["position_error"] = std::numeric_limits<double>::quiet_NaN()', source)
        self.assertIn('out["history_age_s"] = result.history_age_s', source)
        self.assertIn("committed_history_uid_", source)
