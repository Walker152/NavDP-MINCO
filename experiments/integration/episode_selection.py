from __future__ import annotations

from pathlib import Path
import numpy as np

from experiments.designers.manifest import load_manifest


def select_episodes(manifest_path, scene_id, episode_uids):
    manifest = load_manifest(manifest_path)
    scenes = [scene for scene in manifest.scenes if scene.scene_id == scene_id]
    if len(scenes) != 1:
        raise ValueError(f"scene_id not found exactly once: {scene_id}")
    by_uid = {episode.episode_uid: episode for episode in scenes[0].episodes}
    missing = [uid for uid in episode_uids if uid not in by_uid]
    if missing:
        raise ValueError(f"episode_uids not in scene {scene_id}: {missing}")
    return [by_uid[uid] for uid in episode_uids]


def materialize_episode_init(manifest_path, scene_id, episode_uids, output_path):
    episodes = select_episodes(manifest_path, scene_id, episode_uids)
    rows = np.asarray([
        [episode.start_pose[0], episode.start_pose[1], episode.goal_pose[0], episode.goal_pose[1], episode.start_pose[2]]
        for episode in episodes
    ], dtype=np.float64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, rows, allow_pickle=False)
    return rows
