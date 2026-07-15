from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


DEFAULT_SCENES = (
    {"scene_id":"cluttered_easy_0", "scene_label":"SPARSE", "scene_path":"assets/scenes/cluttered_easy/easy_0", "scenario_ids":("REAL-REGULAR","REAL-YAW"), "navdp_seed_base":1000},
    {"scene_id":"cluttered_hard_0", "scene_label":"DENSE", "scene_path":"assets/scenes/cluttered_hard/hard_0", "scenario_ids":("REAL-DENSE","REAL-LONG"), "navdp_seed_base":2000},
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def import_pointgoal_scene(repo_root, definition, source_indices=(0, 1)):
    repo_root = Path(repo_root).resolve(); relative = Path(definition["scene_path"]); scene_dir = repo_root / relative
    usd_files = sorted(scene_dir.glob("*.usd"))
    if len(usd_files) != 1: raise ValueError(f"expected exactly one USD in {scene_dir}")
    rows = np.load(scene_dir / "pointgoal_start_goal_pairs.npy", allow_pickle=False)
    episodes = []
    for episode_index, source_index in enumerate(source_indices):
        if not 0 <= source_index < len(rows): raise IndexError(source_index)
        sx, sy, gx, gy, yaw = (float(value) for value in rows[source_index])
        episodes.append({
            "scenario_id":definition["scenario_ids"][episode_index], "episode_index":episode_index,
            "source_episode_index":source_index, "seed":0,
            "navdp_seed":definition["navdp_seed_base"] + episode_index,
            "start_pose":[sx, sy, yaw], "goal_pose":[gx, gy, 0.0],
            "selection_reason":f"deterministic source row {source_index}",
        })
    return {
        "scene_id":definition["scene_id"], "scene_label":definition["scene_label"],
        "scene_path":relative.as_posix(), "asset_hash":_sha256(usd_files[0]), "episodes":episodes,
    }


def build_default_real_manifest(repo_root):
    return {
        "manifest_version":1, "manifest_id":"navdp_minco_real_pointgoal_v1", "seed":0,
        "scenes":[import_pointgoal_scene(repo_root, definition) for definition in DEFAULT_SCENES],
    }
