from __future__ import annotations

import hashlib
import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from experiments.core.artifact_receipt import inventory_receipts
from experiments.rolling.models import RolloutConfig
from experiments.rolling.scenarios import (
    initial_state_sweeps,
    load_scenarios,
    load_showcase_config,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHOWCASE_DIRECTORIES = (
    "01_trajectory_optimization",
    "02_safe_corridor",
    "03_initial_state",
    "04_dynamic_obstacles",
    "05_extreme_cases",
    "06_aggregate_figures",
)
REQUIRED_METHODS = (
    "guide_reference",
    "legacy",
    "superplanner_sfc_v1",
)
PAIRING_KEY = ("scenario_uid", "seed", "initial_state_hash")
INELIGIBLE_CORRIDOR_EFFECT_FAMILIES = frozenset(
    {"free_space", "sampling_boundary"}
)


def _scene_package_is_valid(path: Path) -> bool:
    """Return true only for a self-validated immutable scene package."""
    if not (path / "scene_manifest.json").is_file():
        return False
    from experiments.visualizers.rolling_showcase import validate_scene_package

    return validate_scene_package(path) == []


def _resumable_root(root: Path) -> bool:
    """Recognise only a workflow-created, unfinished showcase directory."""
    if (root / "showcase_checkpoint.json").is_file():
        return True
    expected = {root / name for name in SHOWCASE_DIRECTORIES}
    return (
        expected.issubset(set(root.iterdir()))
        and not (root / "showcase_manifest.json").exists()
        and any(root.rglob("scene_manifest.json"))
    )


def _checkpoint(root: Path, config_path: Path, *, status: str) -> None:
    (root / "showcase_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _result_from_evidence_or_run(
    *, scenario: object, method: str, profile: Mapping[str, object],
    rollout_config: RolloutConfig, root: Path, rollout_runner: Callable[..., object],
    rollout_serializer: Callable[[object, Path], object],
) -> object:
    """Reuse hash-validated evidence; recompute only an absent or bad method."""
    from experiments.rolling.serialization import (
        load_rollout_result,
        validate_rollout_result,
    )

    evidence = root / "rollout_evidence" / str(getattr(scenario, "scenario_uid")) / method
    manifest = evidence / "run_manifest.json"
    if manifest.is_file() and validate_rollout_result(evidence) == []:
        return load_rollout_result(manifest)
    if evidence.exists():
        shutil.rmtree(evidence)
    result = rollout_runner(
        scenario, method=method, profile=profile, config=rollout_config,
    )
    rollout_serializer(result, evidence)
    return result


def _finite_number(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _extreme_score(row: Mapping[str, object]) -> float:
    return (
        _finite_number(row, "distortion_improvement")
        + 5.0 * _finite_number(row, "clearance_improvement_m")
        + 10.0 * _finite_number(row, "failure_improvement")
    )


def _result_metric(result: object, key: str) -> float:
    metrics = getattr(result, "metrics", {})
    if not isinstance(metrics, Mapping):
        return 0.0
    return _finite_number(metrics, key)


def _extreme_evidence_row(
    scenario_uid: str,
    scene_family: str,
    legacy: object,
    safe: object,
) -> dict[str, object]:
    legacy_distortion = _result_metric(legacy, "guide_deviation_max_m")
    safe_distortion = _result_metric(safe, "guide_deviation_max_m")
    legacy_clearance = _result_metric(legacy, "min_clearance_m")
    safe_clearance = _result_metric(safe, "min_clearance_m")
    row: dict[str, object] = {
        "scenario_uid": scenario_uid,
        "scene_family": scene_family,
        "legacy_distortion_m": legacy_distortion,
        "safe_distortion_m": safe_distortion,
        "legacy_clearance_m": legacy_clearance,
        "safe_clearance_m": safe_clearance,
        "distortion_improvement": legacy_distortion - safe_distortion,
        "clearance_improvement_m": safe_clearance - legacy_clearance,
        "failure_improvement": int(
            getattr(legacy, "status", "") != "GOAL_REACHED"
            and getattr(safe, "status", "") == "GOAL_REACHED"
        ),
    }
    row["score"] = _extreme_score(row)
    return row


def select_extreme_cases(
    rows: Iterable[Mapping[str, object]], *, count: int = 4
) -> tuple[str, ...]:
    """Select eligible corridor-effect cases with a deterministic locked score."""
    if count < 1:
        raise ValueError("count must be positive")
    eligible = [
        row
        for row in rows
        if str(row.get("scene_family", ""))
        not in INELIGIBLE_CORRIDOR_EFFECT_FAMILIES
        and str(row.get("scenario_uid", ""))
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (-_extreme_score(row), str(row["scenario_uid"])),
    )
    return tuple(str(row["scenario_uid"]) for row in ordered[:count])


def validate_showcase_manifest(manifest: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("showcase manifest schema_version must be 1")
    if tuple(manifest.get("paired_key", ())) != PAIRING_KEY:
        errors.append("paired_key must be scenario_uid+seed+initial_state_hash")
    scenes = manifest.get("scenes", [])
    if not isinstance(scenes, list):
        return [*errors, "scenes must be a list"]
    seen: set[str] = set()
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            errors.append(f"scene {index} must be an object")
            continue
        uid = str(scene.get("scenario_uid", ""))
        if not uid or uid in seen:
            errors.append(f"scene {index} scenario_uid must be nonempty and unique")
        seen.add(uid)
        methods = tuple(scene.get("methods", ()))
        if methods != REQUIRED_METHODS:
            errors.append(f"{uid or index}: methods must be {REQUIRED_METHODS}")
        receipts = scene.get("method_pair_receipts", {})
        if not isinstance(receipts, Mapping):
            errors.append(f"{uid or index}: method_pair_receipts must be an object")
            continue
        comparison = [
            receipts.get(method)
            for method in ("legacy", "superplanner_sfc_v1")
        ]
        if any(not isinstance(row, Mapping) for row in comparison):
            errors.append(f"{uid or index}: paired method receipts are incomplete")
            continue
        for key in ("seed", "initial_state_hash"):
            values = {str(row.get(key, "")) for row in comparison}
            if len(values) != 1 or "" in values:
                errors.append(f"{uid or index}: unfair paired {key}")
    return errors


def write_showcase_index(
    manifest: Mapping[str, object], output_dir: Path | str
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "README_成果索引.md"
    lines = [
        "# NavDP 论文直观成果索引",
        "",
        "本目录只索引由实验数据直接生成并通过收据校验的论文展示产物。",
        "",
        "## 轨迹优化",
        "",
        "完整路线比较位于 `01_trajectory_optimization/`。",
        "",
        "## 安全走廊",
        "",
        "稀疏、密集、窄通道和畸形绕行比较位于 `02_safe_corridor/`。",
        "",
        "## 不同初值",
        "",
        "位置、速度、加速度、yaw 与 yaw rate 单因素结果位于 `03_initial_state/`。",
        "",
        "## 动态障碍",
        "",
        "横穿、迎面和突然出现障碍位于 `04_dynamic_obstacles/`。",
        "",
        "## 极端案例",
        "",
        "由锁定评分选出的案例位于 `05_extreme_cases/`。",
        "",
        "## 直接查看",
        "",
        "| 场景 | 效果图 | 看什么 | 证明什么 | 限制 |",
        "|---|---|---|---|---|",
    ]
    for scene in manifest.get("scenes", []):
        if not isinstance(scene, Mapping):
            continue
        package = str(scene.get("package_path", "")).strip("/")
        figure = str(scene.get("headline_figure", "three_panel.png"))
        link = f"{package}/{figure}" if package else figure
        uid = str(scene.get("scenario_uid", "未命名"))
        conclusion = str(scene.get("conclusion_zh", "见配套 caption。"))
        limitation = str(scene.get("limitation_zh", "见配套 caption。"))
        lines.append(
            f"| `{uid}` | [{figure}]({link}) | 输入 guide、Legacy 与安全走廊完整路线 | "
            f"{conclusion} | {limitation} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def initial_state_hash(state: Mapping[str, object]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_rolling_showcase(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    rollout_runner: Callable[..., object] | None = None,
    scene_renderer: Callable[[object, Path], object] | None = None,
    rollout_serializer: Callable[[object, Path], object] | None = None,
    write_receipts: bool = True,
) -> dict[str, object]:
    """Execute both rollout profiles and render the requested showcase matrix."""
    config = load_showcase_config(config_path)
    scenarios = load_scenarios(config_path)
    if rollout_runner is None:
        from experiments.rolling.engine import run_rollout

        rollout_runner = run_rollout
    if scene_renderer is None:
        from experiments.visualizers.rolling_showcase import render_scene_package

        scene_renderer = render_scene_package
    if rollout_serializer is None:
        from experiments.rolling.serialization import write_rollout_result

        rollout_serializer = write_rollout_result
    config_path = Path(config_path).resolve()
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()) and not _resumable_root(root):
        raise FileExistsError(f"immutable rolling showcase exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for name in SHOWCASE_DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    _checkpoint(root, config_path, status="RUNNING")
    rollout_config = RolloutConfig(**config["rollout"])
    profiles = _load_profiles(Path(config_path).resolve().parents[2])
    scene_rows: list[dict[str, object]] = []
    for scenario in scenarios.values():
        results: dict[str, object] = {}
        for method in ("legacy", "superplanner_sfc_v1"):
            results[method] = _result_from_evidence_or_run(
                scenario=scenario, method=method, profile=profiles[method],
                rollout_config=rollout_config, root=root,
                rollout_runner=rollout_runner, rollout_serializer=rollout_serializer,
            )
        package_path = _scene_package_path(scenario.family, scenario.scenario_uid)
        paired = {
            "scenario_uid": scenario.scenario_uid,
            "guide_path_xyz": scenario.guide_path_xyz,
            "final_goal_xyz": scenario.final_goal_xyz,
            "goal_yaw_rad": None,
            "initial_state": scenario.initial_state,
            "static_rectangles": scenario.static_rectangles,
            **results,
        }
        package_root = root / package_path
        if _scene_package_is_valid(package_root):
            package = None
        else:
            if package_root.exists():
                shutil.rmtree(package_root)
            package = scene_renderer(paired, package_root)
        state_receipt = {
            "seed": scenario.seed,
            "initial_state_hash": initial_state_hash(
                {
                    "position_xyz": scenario.initial_state.position_xyz.tolist(),
                    "velocity_xyz_mps": scenario.initial_state.velocity_xyz_mps.tolist(),
                    "acceleration_xyz_mps2": scenario.initial_state.acceleration_xyz_mps2.tolist(),
                    "yaw_rad": scenario.initial_state.yaw_rad,
                    "yaw_rate_radps": scenario.initial_state.yaw_rate_radps,
                }
            ),
        }
        scene_rows.append(
            {
                "scenario_uid": scenario.scenario_uid,
                "scene_family": scenario.family,
                "paired_key": "scenario_uid+seed+initial_state_hash",
                "methods": list(REQUIRED_METHODS),
                "method_pair_receipts": {
                    "guide_reference": state_receipt,
                    "legacy": state_receipt,
                    "superplanner_sfc_v1": state_receipt,
                },
                "package_path": package_path.as_posix(),
                "headline_figure": "three_panel.png",
                "conclusion_zh": "比较相同输入与初值下的完整滚动路线。",
                "limitation_zh": "本地确定性规划证据，不替代 IsaacLab 真实闭环验证。",
                "statuses": {
                    method: str(getattr(result, "status", "UNKNOWN"))
                    for method, result in results.items()
                },
                "extreme_evidence": _extreme_evidence_row(
                    scenario.scenario_uid,
                    scenario.family,
                    results["legacy"],
                    results["superplanner_sfc_v1"],
                ),
            }
        )
    baseline = scenarios["unobstructed"]
    for sweep in initial_state_sweeps(config):
        scenario = replace(
            baseline,
            scenario_uid=sweep.sweep_uid,
            initial_state=sweep.variant,
        )
        results = {
            method: _result_from_evidence_or_run(
                scenario=scenario, method=method, profile=profiles[method],
                rollout_config=rollout_config, root=root,
                rollout_runner=rollout_runner, rollout_serializer=rollout_serializer,
            )
            for method in ("legacy", "superplanner_sfc_v1")
        }
        package_path = (
            Path("03_initial_state") / sweep.factor / scenario.scenario_uid
        )
        paired = {
            "scenario_uid": scenario.scenario_uid,
            "guide_path_xyz": scenario.guide_path_xyz,
            "final_goal_xyz": scenario.final_goal_xyz,
            "goal_yaw_rad": None,
            "initial_state": scenario.initial_state,
            "static_rectangles": scenario.static_rectangles,
            **results,
        }
        package_root = root / package_path
        if not _scene_package_is_valid(package_root):
            if package_root.exists():
                shutil.rmtree(package_root)
            scene_renderer(paired, package_root)
        state_receipt = {
            "seed": scenario.seed,
            "initial_state_hash": initial_state_hash(
                {
                    "position_xyz": scenario.initial_state.position_xyz.tolist(),
                    "velocity_xyz_mps": scenario.initial_state.velocity_xyz_mps.tolist(),
                    "acceleration_xyz_mps2": scenario.initial_state.acceleration_xyz_mps2.tolist(),
                    "yaw_rad": scenario.initial_state.yaw_rad,
                    "yaw_rate_radps": scenario.initial_state.yaw_rate_radps,
                }
            ),
        }
        scene_rows.append(
            {
                "scenario_uid": scenario.scenario_uid,
                "scene_family": scenario.family,
                "experiment_group": "initial_state",
                "initial_factor": sweep.factor,
                "paired_key": "scenario_uid+seed+initial_state_hash",
                "methods": list(REQUIRED_METHODS),
                "method_pair_receipts": {
                    method: state_receipt for method in REQUIRED_METHODS
                },
                "package_path": package_path.as_posix(),
                "headline_figure": "three_panel.png",
                "conclusion_zh": f"展示仅改变初值 {sweep.factor} 时的完整路线。",
                "limitation_zh": "单因素本地确定性规划证据。",
                "statuses": {
                    method: str(getattr(result, "status", "UNKNOWN"))
                    for method, result in results.items()
                },
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "paired_key": list(PAIRING_KEY),
        "config_path": str(Path(config_path).resolve()),
        "config_schema_version": config.get("schema_version"),
        "scenes": scene_rows,
        "data_source": "SIMULATED",
        "real_simulation_status": "PENDING_REAL_SIMULATION",
    }
    _write_aggregate_outputs(root, scene_rows)
    _write_extreme_selection(root, scene_rows)
    (root / "showcase_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_showcase_index(manifest, root)
    _checkpoint(root, config_path, status="COMPLETE")
    if write_receipts:
        receipts = inventory_receipts(
            root,
            exclude={root / "artifact_receipt.json"},
        )
        (root / "artifact_receipt.json").write_text(
            json.dumps(
                {"schema_version": 1, "root": ".", "artifacts": receipts},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return manifest


def _load_profiles(repo_root: Path) -> dict[str, Mapping[str, object]]:
    from experiments.static.runner import load_legacy_profile

    config = repo_root / "experiments" / "configs"
    return {
        "legacy": load_legacy_profile(config / "static_legacy_suite.json"),
        "superplanner_sfc_v1": load_legacy_profile(
            config / "static_superplanner_sfc_suite.json"
        ),
    }


def _scene_package_path(family: str, scenario_uid: str) -> Path:
    if family == "unobstructed":
        return Path("01_trajectory_optimization") / scenario_uid
    if family in {"static_sparse", "static_dense", "narrow_passage", "malformed_detour"}:
        leaf = {
            "static_sparse": "sparse_obstacles",
            "static_dense": "dense_obstacles",
            "narrow_passage": "narrow_passage",
            "malformed_detour": "malformed_detour",
        }[family]
        return Path("02_safe_corridor") / leaf / scenario_uid
    return Path("04_dynamic_obstacles") / family.removeprefix("dynamic_") / scenario_uid


def _write_aggregate_outputs(
    root: Path, scenes: Iterable[Mapping[str, object]]
) -> None:
    output = root / "06_aggregate_figures"
    rows = []
    for scene in scenes:
        statuses = scene.get("statuses", {})
        if not isinstance(statuses, Mapping):
            statuses = {}
        rows.append(
            {
                "scenario_uid": scene.get("scenario_uid", ""),
                "scene_family": scene.get("scene_family", ""),
                "experiment_group": scene.get("experiment_group", "scenario"),
                "legacy_status": statuses.get("legacy", "UNKNOWN"),
                "superplanner_sfc_v1_status": statuses.get(
                    "superplanner_sfc_v1", statuses.get("safe_corridor_v1", "UNKNOWN")
                ),
            }
        )
    csv_path = output / "all_scenarios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0]) if rows else ["scenario_uid"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    methods = ("legacy", "superplanner_sfc_v1")
    reached = [
        sum(row[f"{method}_status"] == "GOAL_REACHED" for row in rows)
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    axis.bar(methods, reached, color=("#D55E00", "#0072B2"))
    axis.set_ylabel("Goal-reached scenarios")
    axis.set_title("Full-route rolling status comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "status_comparison.png", dpi=300, facecolor="white")
    figure.savefig(
        output / "status_comparison.pdf",
        metadata={"Creator": "NavDP rolling showcase", "CreationDate": None},
    )
    plt.close(figure)


def _write_extreme_selection(
    root: Path, scenes: Iterable[Mapping[str, object]]
) -> None:
    candidates = [
        dict(scene.get("extreme_evidence", {}))
        for scene in scenes
        if scene.get("experiment_group") != "initial_state"
    ]
    selected = set(select_extreme_cases(candidates, count=min(4, len(candidates))))
    output = root / "05_extreme_cases" / "selection.csv"
    fields = [
        "scenario_uid",
        "scene_family",
        "distortion_improvement",
        "clearance_improvement_m",
        "failure_improvement",
        "legacy_distortion_m",
        "safe_distortion_m",
        "legacy_clearance_m",
        "safe_clearance_m",
        "score",
        "selected",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in candidates:
            writer.writerow(
                {
                    **row,
                    "score": _extreme_score(row),
                    "selected": str(row["scenario_uid"] in selected).lower(),
                }
            )


def validate_showcase(output_dir: Path | str) -> list[str]:
    root = Path(output_dir)
    errors = [
        f"missing showcase directory {name}"
        for name in SHOWCASE_DIRECTORIES
        if not (root / name).is_dir()
    ]
    manifest_path = root / "showcase_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [*errors, f"unreadable showcase manifest: {error}"]
    errors.extend(validate_showcase_manifest(manifest))
    if not (root / "README_成果索引.md").is_file():
        errors.append("missing README_成果索引.md")
    return errors
