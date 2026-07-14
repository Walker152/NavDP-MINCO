import json
from pathlib import Path

from experiments.core.models import EpisodeSpec, Manifest, SceneSpec


def load_manifest(path: Path | str) -> Manifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_version") != 1: raise ValueError("manifest_version must be 1")
    scenes = [] ; seen = set()
    for raw_scene in sorted(data.get("scenes", []), key=lambda item: item["scene_id"]):
        episodes = []
        for raw in sorted(raw_scene.get("episodes", []), key=lambda item: (item["episode_index"], item["seed"])):
            episode = EpisodeSpec(scene_id=raw_scene["scene_id"], scene_label=raw_scene["scene_label"], scenario_id=raw["scenario_id"], episode_index=int(raw["episode_index"]), seed=int(raw["seed"]), start_pose=tuple(raw["start_pose"]), goal_pose=tuple(raw["goal_pose"]), episode_uid=raw.get("episode_uid", ""), navdp_seed=raw.get("navdp_seed"))
            if episode.episode_uid in seen: raise ValueError(f"duplicate episode_uid: {episode.episode_uid}")
            seen.add(episode.episode_uid); episodes.append(episode)
        if not episodes: raise ValueError(f"scene {raw_scene['scene_id']} has no episodes")
        scenes.append(SceneSpec(raw_scene["scene_id"], raw_scene["scene_label"], raw_scene["scene_path"], tuple(episodes)))
    if not scenes: raise ValueError("manifest has no scenes")
    return Manifest(1, data["manifest_id"], int(data.get("seed", 0)), tuple(scenes))
