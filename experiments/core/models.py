from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def stable_id(prefix: str, payload: dict[str, Any], length: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:length]}"


@dataclass(frozen=True)
class EpisodeSpec:
    scene_id: str
    scene_label: str
    scenario_id: str
    episode_index: int
    seed: int
    start_pose: tuple[float, ...]
    goal_pose: tuple[float, ...]
    episode_uid: str = ""
    navdp_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_pose", tuple(self.start_pose))
        object.__setattr__(self, "goal_pose", tuple(self.goal_pose))
        if not self.episode_uid:
            payload = {k: v for k, v in asdict(self).items() if k not in {"episode_uid", "navdp_seed"}}
            object.__setattr__(self, "episode_uid", stable_id("ep", payload))
        if self.navdp_seed is None:
            object.__setattr__(self, "navdp_seed", self.seed + 100000)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    scene_label: str
    scene_path: str
    episodes: tuple[EpisodeSpec, ...]


@dataclass(frozen=True)
class Manifest:
    manifest_version: int
    manifest_id: str
    seed: int
    scenes: tuple[SceneSpec, ...]


@dataclass(frozen=True)
class RunSpec:
    suite_id: str
    experiment_id: str
    variant: str
    warm_start_mode: str
    scene_label: str
    scene_id: str
    seed: int
    run_id: str


@dataclass(frozen=True)
class SuiteResult:
    completed: int = 0
    skipped: int = 0
    failed: int = 0
