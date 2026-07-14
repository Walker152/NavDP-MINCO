import ast
from pathlib import Path
import unittest


class NavDPStaticInterfaceTests(unittest.TestCase):
    def test_server_has_health_seed_and_optional_video_contract(self):
        source = Path("baselines/navdp/navdp_server.py").read_text()
        ast.parse(source)
        for token in ("/health", "model_loaded", "--save-video", "--output-dir", "--run-id", "--seed", "--log-interval"):
            self.assertIn(token, source)
        self.assertIn("default=False", source)
        self.assertIn("seed=", source)

    def test_policy_uses_explicit_generator_for_every_diffusion_noise(self):
        source = Path("baselines/navdp/policy_network.py").read_text(); tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "randn"]
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(any(keyword.arg == "generator" for keyword in call.keywords) for call in calls))
        self.assertIn("torch.Generator", source); self.assertIn("manual_seed", source)

    def test_agent_reset_accepts_seed_without_changing_default_callers(self):
        source = Path("baselines/navdp/policy_agent.py").read_text(); tree = ast.parse(source)
        reset = next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reset")
        self.assertIn("seed", [argument.arg for argument in reset.args.args])
        self.assertIsNotNone(reset.args.defaults[-1])
