from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from experiments.core.models import Manifest, RunSpec, stable_id


@dataclass(frozen=True)
class SuiteConfig:
    suite_id: str
    output_root: Path
    manifest_path: Path
    runs: tuple[dict, ...]


def load_suite(path: Path | str) -> SuiteConfig:
    path = Path(path); data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("suite_id") or not data.get("runs"): raise ValueError("suite_id and runs are required")
    for run in data["runs"]:
        if run["variant"] == "minco-hot" and run.get("warm_start_mode") != "gated": raise ValueError("minco-hot requires gated warm_start_mode")
        if run["variant"] in {"raw", "minco-cold"} and run.get("warm_start_mode") != "cold": raise ValueError("raw/minco-cold require cold warm_start_mode")
    manifest = Path(data["manifest"]); manifest = manifest if manifest.is_absolute() else (path.parent / manifest).resolve()
    root = Path(data.get("output_root", "results")); root = root if root.is_absolute() else (path.parent / root).resolve()
    return SuiteConfig(data["suite_id"], root, manifest, tuple(data["runs"]))


def expand_runs(config: SuiteConfig, manifest: Manifest):
    for template in config.runs:
        labels = set(template.get("scene_labels", []))
        for scene in manifest.scenes:
            if labels and scene.scene_label not in labels: continue
            seeds = sorted({episode.seed for episode in scene.episodes})
            for seed in seeds:
                payload = {"experiment": template["experiment_id"], "variant": template["variant"], "scene": scene.scene_id, "seed": seed}
                yield RunSpec(config.suite_id, template["experiment_id"], template["variant"], template["warm_start_mode"], scene.scene_label, scene.scene_id, seed, stable_id("run", payload, 12))
