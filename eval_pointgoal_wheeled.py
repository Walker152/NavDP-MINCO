import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A script to run a car control simulation")
parser.add_argument(
    "--scene_dir", type=str, default="/home/alioth/NavDP/assets/scenes/cluttered_easy")
parser.add_argument(
    "--scene_index", type=int, default=0)
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0)
parser.add_argument(
    "--num_envs", type=int, default=1)
parser.add_argument(
    "--num_episodes", type=int, default=10)
parser.add_argument(
    "--speed", type=float, default=1.0)
parser.add_argument("--mpc_max_yaw_rate", type=float, default=0.5)
parser.add_argument("--mpc_max_yaw_acc", type=float, default=2.0)
parser.add_argument(
    "--mpc_max_wheel_speed",
    type=float,
    default=0.0,
    help="Maximum wheel-joint angular speed in rad/s; <= 0 disables the MPC wheel constraint",
)
parser.add_argument(
    "--port", type=int, default=8888)
parser.add_argument("--enable_minco", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--minco_top_k", type=int, default=2)
parser.add_argument("--minco_safe_dist", type=float, default=0.60)
parser.add_argument("--minco_sample_dt", type=float, default=0.05)
parser.add_argument("--minco_max_vel", type=float, default=1.0)
parser.add_argument("--minco_max_acc", type=float, default=2.0)
parser.add_argument("--minco_max_iterations", type=int, default=64)
parser.add_argument("--esdf_resolution", type=float, default=0.05)
parser.add_argument("--esdf_padding", type=float, default=1.0)
parser.add_argument("--esdf_force_rebuild", action="store_true")
parser.add_argument("--esdf_cache_name", type=str, default="esdf_2d.npz")
parser.add_argument("--esdf_obstacle_min_height", type=float, default=0.08)
parser.add_argument("--esdf_obstacle_max_height", type=float, default=1.50)
parser.add_argument("--esdf_fill_footprint", type=int, default=1)
parser.add_argument("--esdf_footprint_inflate_cells", type=int, default=1)
parser.add_argument("--use_robot_base_frame", type=int, default=1)
parser.add_argument("--timing_log_interval", type=int, default=1)
parser.add_argument("--show_timing_overlay", default=True, action=argparse.BooleanOptionalAction)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController
import torchvision.transforms as F
import time
import threading

from utils_tasks.basic_utils import (
    PlanningInput,
    PlanningOutput,
    compute_forward_velocity,
    find_usd_path,
    write_metrics,
    draw_box_with_text,
    adjust_usd_scale,
)
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_close,navigator_reset,pointgoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller
from utils_tasks.esdf_query_utils import query_esdf_polyline
from utils_tasks.timing_utils import (
    StageTimer,
    append_timing_panel,
    format_control_summary,
    format_minco_summary,
    format_planning_summary,
    mean_timing,
)

planning_input = PlanningInput() 
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
navdp_http_lock = threading.Lock()
stop_event = threading.Event()
reset_in_progress = threading.Event()
vis_manager = [VisualizationManager(history_size=5, show_all_candidates=True) for i in range(args_cli.num_envs)]
mpc = [None for _ in range(args_cli.num_envs)]
last_applied_plan_id = [-1 for _ in range(args_cli.num_envs)]
per_env_plan_id = np.zeros(args_cli.num_envs, dtype=np.int64)
minco_hold_cache = [None for _ in range(args_cli.num_envs)]
episode_generation = np.zeros(args_cli.num_envs, dtype=np.int64)
reset_pending = np.zeros(args_cli.num_envs, dtype=bool)
reset_stable_count = np.zeros(args_cli.num_envs, dtype=np.int32)
RESET_CAMERA_BASE_OFFSET_TOL = 0.05
RESET_STABLE_REQUIRED_FRAMES = 3
use_robot_base_frame = bool(args_cli.use_robot_base_frame)
wheel_constraint_enabled = (
    np.isfinite(args_cli.mpc_max_wheel_speed)
    and args_cli.mpc_max_wheel_speed > 0.0
)
mpc_max_wheel_speed = (
    float(args_cli.mpc_max_wheel_speed) if wheel_constraint_enabled else None
)

def transform_navdp_local_point(point_xy, env_idx, camera_pos, camera_rot, robot_pos_w=None):
    point_local = np.array([point_xy[0], point_xy[1], 0.0], dtype=np.float64)
    point_world = camera_pos[env_idx] + camera_rot[env_idx] @ point_local
    if use_robot_base_frame and robot_pos_w is not None:
        point_world[:2] += robot_pos_w[env_idx, :2] - camera_pos[env_idx, :2]
    point_world[2] = 0.0
    return point_world

def transform_navdp_local_traj(traj_local, env_idx, camera_pos, camera_rot, robot_pos_w=None):
    return np.array(
        [
            transform_navdp_local_point(point, env_idx, camera_pos, camera_rot, robot_pos_w)
            for point in traj_local
        ],
        dtype=np.float64,
    )

def minco_cache_entry(result, episode_gen):
    samples = result.get("samples")
    samples_np = np.asarray(samples, dtype=np.float64) if samples is not None else None
    if samples_np is None or samples_np.ndim != 2 or samples_np.shape[0] < 2:
        return None
    duration = float(samples_np[-1, 0] - samples_np[0, 0])
    if not np.isfinite(duration) or duration <= 0.0:
        return None
    return {
        "waypoints": np.asarray(result["waypoints"], dtype=np.float64).copy(),
        "samples": samples_np.copy(),
        "speed_profile": None if result.get("speed_profile") is None else np.asarray(result["speed_profile"], dtype=np.float64).copy(),
        "selected_candidate": None if result.get("selected_candidate") is None else np.asarray(result["selected_candidate"], dtype=np.float64).copy(),
        "sparse_waypoints": None if result.get("sparse_waypoints") is None else np.asarray(result["sparse_waypoints"], dtype=np.float64).copy(),
        "published_time": time.monotonic(),
        "episode_generation": int(episode_gen),
        "duration": duration,
        "info": dict(result),
    }

def is_minco_cache_valid(cache, episode_gen):
    if cache is None or int(cache.get("episode_generation", -1)) != int(episode_gen):
        return False
    elapsed = time.monotonic() - float(cache.get("published_time", 0.0))
    duration = float(cache.get("duration", 0.0))
    return np.isfinite(elapsed) and np.isfinite(duration) and elapsed <= duration

def mark_planning_idle():
    with output_lock:
        planning_output.is_planning = False

def planning_result_is_stale(captured_generation):
    if reset_in_progress.is_set():
        return True, episode_generation.copy()
    with input_lock:
        current_generation = (
            planning_input.episode_generation.copy()
            if planning_input.episode_generation is not None
            else episode_generation.copy()
        )
    return not np.array_equal(captured_generation, current_generation), current_generation

def discard_stale_planning_result(captured_generation):
    stale, current_generation = planning_result_is_stale(captured_generation)
    if stale:
        print(
            f"[Planning] discard stale result captured_generation={captured_generation.tolist()} "
            f"current_generation={current_generation.tolist()}"
        )
        mark_planning_idle()
    return stale

def invalidate_planning_state_for_reset():
    with input_lock:
        planning_input.current_goal = None
        planning_input.current_image = None
        planning_input.current_depth = None
        planning_input.camera_pos = None
        planning_input.camera_rot = None
        planning_input.robot_pos_w = None
        planning_input.robot_yaw_w = None
        planning_input.robot_lin_vel_w = None
        planning_input.robot_ang_vel_w = None
        planning_input.episode_generation = episode_generation.copy()
    with output_lock:
        planning_output.trajectory_points_world = None
        planning_output.all_trajectories_world = None
        planning_output.all_values_camera = None
        planning_output.raw_top1_world = None
        planning_output.selected_candidate_world = None
        planning_output.minco_sparse_waypoints_world = None
        planning_output.minco_samples = None
        planning_output.minco_speed_profile = None
        planning_output.minco_info = None
        planning_output.per_env_plan_id = None
        planning_output.stop_required = None
        planning_output.minco_status = None
        planning_output.planning_timing = None
        planning_output.is_planning = False
        planning_output.planning_error = None
        planning_output.episode_generation = episode_generation.copy()

def planning_thread(env, camera_intrinsic, minco_adapter=None):
    """Thread function that continuously plans trajectories"""
    planning_iter = 0
    while not stop_event.is_set():
        try:
            if reset_in_progress.is_set():
                time.sleep(0.01)
                continue
            # Get latest observations from shared state
            planning_timer = StageTimer()
            planning_total_start = planning_timer.now()
            with planning_timer.section("input_copy_ms"):
                with input_lock:
                    if reset_in_progress.is_set():
                        continue
                    if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                        continue
                    goal = planning_input.current_goal.copy()
                    image = planning_input.current_image.copy()
                    depth = planning_input.current_depth.copy()
                    camera_pos = planning_input.camera_pos.copy()
                    camera_rot = planning_input.camera_rot.copy()
                    robot_pos_w = None if planning_input.robot_pos_w is None else planning_input.robot_pos_w.copy()
                    robot_yaw_w = None if planning_input.robot_yaw_w is None else planning_input.robot_yaw_w.copy()
                    robot_lin_vel_w = None if planning_input.robot_lin_vel_w is None else planning_input.robot_lin_vel_w.copy()
                    robot_ang_vel_w = None if planning_input.robot_ang_vel_w is None else planning_input.robot_ang_vel_w.copy()
                    input_episode_generation = (
                        planning_input.episode_generation.copy()
                        if planning_input.episode_generation is not None
                        else np.zeros(args_cli.num_envs, dtype=np.int64)
                    )
            with output_lock:
                planning_output.is_planning = True
            
            with planning_timer.section("navdp_step_ms"):
                with navdp_http_lock:
                    trajectory_points_camera, all_trajectories_camera, all_values_camera = pointgoal_step(goal, image, depth,port=args_cli.port)
            if stop_event.is_set():
                mark_planning_idle()
                break
            if discard_stale_planning_result(input_episode_generation):
                continue

            with planning_timer.section("raw_transform_ms"):
                raw_top1_world = []
                for idx in range(trajectory_points_camera.shape[0]):
                    trajectory_points_world = transform_navdp_local_traj(
                        trajectory_points_camera[idx], idx, camera_pos, camera_rot, robot_pos_w
                    )[:, :2]
                    raw_top1_world.append(trajectory_points_world)
                raw_top1_world = np.array(raw_top1_world, dtype=object)

            with planning_timer.section("candidate_transform_ms"):
                batch_all_points_world = []
                terminal_goals_world = []
                for idx in range(all_trajectories_camera.shape[0]):
                    terminal_goals_world.append(
                        transform_navdp_local_point(goal[idx], idx, camera_pos, camera_rot, robot_pos_w)
                    )
                    # Transform all trajectories
                    all_trajectories_world = []
                    for traj_camera in all_trajectories_camera[idx]:
                        all_trajectories_world.append(
                            transform_navdp_local_traj(traj_camera, idx, camera_pos, camera_rot, robot_pos_w)
                        )
                    batch_all_points_world.append(all_trajectories_world)
                batch_all_points_world = np.array(batch_all_points_world, dtype=object)
                terminal_goals_world = np.array(terminal_goals_world, dtype=object)

            batch_optimal_points_world = []
            batch_raw_top1_world = []
            batch_selected_candidate_world = []
            batch_sparse_waypoints_world = []
            batch_minco_samples = []
            batch_minco_speed_profile = []
            batch_minco_info = []
            batch_stop_required = []
            batch_minco_status = []
            batch_per_env_plan_id = per_env_plan_id.copy()
            used_minco = minco_adapter is not None and minco_adapter.enabled
            with planning_timer.section("state_build_ms"):
                states = []
                if used_minco:
                    for idx in range(camera_pos.shape[0]):
                        vel = robot_lin_vel_w[idx].copy() if robot_lin_vel_w is not None else np.zeros(3)
                        vel[2] = 0.0
                        yaw_rate = float(robot_ang_vel_w[idx]) if robot_ang_vel_w is not None else 0.0
                        if use_robot_base_frame and robot_pos_w is not None and robot_yaw_w is not None:
                            state_position = np.array([robot_pos_w[idx, 0], robot_pos_w[idx, 1], 0.0])
                            state_yaw = float(robot_yaw_w[idx])
                        else:
                            state_position = np.array([camera_pos[idx, 0], camera_pos[idx, 1], 0.0])
                            state_yaw = float(np.arctan2(camera_rot[idx, 1, 0], camera_rot[idx, 0, 0]))
                        states.append({
                            "position": state_position,
                            "velocity": vel,
                            "acceleration": np.zeros(3),
                            "yaw": state_yaw,
                            "yaw_rate": yaw_rate,
                        })

            minco_results = []
            if used_minco:
                if stop_event.is_set():
                    mark_planning_idle()
                    break
                with planning_timer.section("minco_total_ms"):
                    minco_results = minco_adapter.optimize_candidates(
                        candidates_world=batch_all_points_world,
                        critic_values=all_values_camera,
                        states=states,
                        raw_top1_world=raw_top1_world,
                        terminal_goals_world=terminal_goals_world,
                    )
                if stop_event.is_set():
                    mark_planning_idle()
                    break
            else:
                planning_timer.records["minco_total_ms"] = 0.0

            if discard_stale_planning_result(input_episode_generation):
                continue
            with output_lock:
                hold_cache_snapshot = list(minco_hold_cache)

            with planning_timer.section("mpc_construct_ms"):
                if used_minco:
                    for idx, result in enumerate(minco_results):
                        if result["success"] and result["waypoints"] is not None and len(result["waypoints"]) >= 2:
                            trajectory_points_world = np.asarray(result["waypoints"])[:, :2]
                            cache = minco_cache_entry(result, input_episode_generation[idx])
                            if cache is not None:
                                batch_per_env_plan_id[idx] += 1
                            status = "MINCO_OK"
                            stop_required = False
                            info = dict(result)
                            info["status"] = status
                        else:
                            if is_minco_cache_valid(hold_cache_snapshot[idx], input_episode_generation[idx]):
                                cache = hold_cache_snapshot[idx]
                                trajectory_points_world = np.asarray(cache["waypoints"], dtype=np.float64)[:, :2]
                                status = "MINCO_HOLD_LAST"
                                stop_required = False
                                info = dict(result)
                                info.update({
                                    "status": status,
                                    "fallback_mode": "HOLD_LAST",
                                    "waypoints": None,
                                    "samples": None,
                                    "selected_index": cache["info"].get("selected_index", -1),
                                })
                                print(f"[NavDP-Minco] env={idx} status=MINCO_HOLD_LAST reason={result.get('failure_reason', '')}")
                            else:
                                trajectory_points_world = None
                                status = "MINCO_STOP"
                                stop_required = True
                                info = dict(result)
                                info.update({
                                    "status": status,
                                    "fallback_mode": "STOP",
                                    "waypoints": None,
                                    "samples": None,
                                })
                                print(f"[NavDP-Minco] env={idx} status=MINCO_STOP reason={result.get('failure_reason', '')}")
                        batch_optimal_points_world.append(trajectory_points_world)
                        batch_raw_top1_world.append(result.get("raw_top1"))
                        batch_selected_candidate_world.append(
                            result.get("selected_candidate") if status != "MINCO_HOLD_LAST" else hold_cache_snapshot[idx].get("selected_candidate")
                        )
                        batch_sparse_waypoints_world.append(
                            result.get("sparse_waypoints") if status == "MINCO_OK" else (
                                hold_cache_snapshot[idx].get("sparse_waypoints") if status == "MINCO_HOLD_LAST" else None
                            )
                        )
                        batch_minco_samples.append(result.get("samples") if status == "MINCO_OK" else None)
                        batch_minco_speed_profile.append(
                            result.get("speed_profile") if status == "MINCO_OK" else (
                                hold_cache_snapshot[idx].get("speed_profile") if status == "MINCO_HOLD_LAST" else None
                            )
                        )
                        batch_minco_info.append(info)
                        batch_stop_required.append(stop_required)
                        batch_minco_status.append(status)
                else:
                    for idx in range(len(raw_top1_world)):
                        trajectory_points_world = np.asarray(raw_top1_world[idx])[:, :2]
                        batch_optimal_points_world.append(trajectory_points_world)
                        batch_raw_top1_world.append(trajectory_points_world)
                        batch_selected_candidate_world.append(None)
                        batch_sparse_waypoints_world.append(None)
                        batch_minco_samples.append(None)
                        batch_minco_speed_profile.append(None)
                        batch_stop_required.append(False)
                        batch_minco_status.append("RAW_NAVDP")
                        batch_minco_info.append({
                            "success": False,
                            "fallback": True,
                            "selected_index": -1,
                            "objective": float("inf"),
                            "min_esdf": float("nan"),
                            "py_min_esdf": float("nan"),
                            "safe_dist": args_cli.minco_safe_dist,
                            "adapter_total_ms": 0.0,
                            "selected_cpp_optimize_time_ms": float("nan"),
                        })

            planning_total_ms = planning_timer.elapsed_ms(planning_total_start)

            batch_optimal_points_world = np.array(batch_optimal_points_world, dtype=object if used_minco else None)
            batch_all_points_world_vis = batch_all_points_world[:, :, :, :2] if batch_all_points_world.ndim == 4 else batch_all_points_world
            planning_timing = {
                **planning_timer.snapshot(),
                "planning_total_ms": planning_total_ms,
                "used_minco": used_minco,
                "minco_results": batch_minco_info,
            }

            if discard_stale_planning_result(input_episode_generation):
                continue

            # Update shared state
            with output_lock:
                if reset_in_progress.is_set() or not np.array_equal(input_episode_generation, episode_generation):
                    planning_output.is_planning = False
                    print(
                        f"[Planning] discard stale result captured_generation={input_episode_generation.tolist()} "
                        f"current_generation={episode_generation.tolist()}"
                    )
                    continue
                if used_minco:
                    for idx, result in enumerate(minco_results):
                        if result["success"] and result["waypoints"] is not None and len(result["waypoints"]) >= 2:
                            cache = minco_cache_entry(result, input_episode_generation[idx])
                            if cache is not None:
                                minco_hold_cache[idx] = cache
                                per_env_plan_id[idx] = batch_per_env_plan_id[idx]
                planning_output.plan_id += 1
                planning_output.episode_generation = input_episode_generation.copy()
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world_vis
                planning_output.all_values_camera = all_values_camera
                planning_output.raw_top1_world = np.array(batch_raw_top1_world, dtype=object)
                planning_output.selected_candidate_world = np.array(batch_selected_candidate_world, dtype=object)
                planning_output.minco_sparse_waypoints_world = np.array(batch_sparse_waypoints_world, dtype=object)
                planning_output.minco_samples = batch_minco_samples
                planning_output.minco_speed_profile = batch_minco_speed_profile
                planning_output.minco_info = batch_minco_info
                planning_output.per_env_plan_id = batch_per_env_plan_id.copy()
                planning_output.stop_required = np.array(batch_stop_required, dtype=bool)
                planning_output.minco_status = list(batch_minco_status)
                planning_output.planning_timing = planning_timing
                planning_output.is_planning = False
                planning_output.planning_error = None

            planning_iter += 1
            if planning_iter % max(1, args_cli.timing_log_interval) == 0:
                print(format_planning_summary(planning_timing))
                if used_minco:
                    for env_i, info in enumerate(batch_minco_info):
                        print(format_minco_summary(env_i, info))
                
        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        # Small sleep to prevent CPU overload
        time.sleep(0.1)

scene_path = os.path.join(args_cli.scene_dir,os.listdir(args_cli.scene_dir)[args_cli.scene_index]) + "/"
usd_path,init_path = find_usd_path(scene_path,task='pointgoal')
scene_config = PointNavSceneCfg()
scene_config.num_envs = args_cli.num_envs
scene_config.env_spacing = 0.0
scene_config.terrain = BENCH_TERRAIN_CFG
scene_config.terrain.usd_path = usd_path
scene_config.goal = GOAL_CFG
scene_config.robot = DINGO_CFG
scene_config.camera_sensor = DINGO_CameraCfg
scene_config.contact_sensor = DINGO_ContactCfg
env_config = DingoPointNavCfg()
env_config.scene = scene_config
env_config.events.reset_pose.params = {"init_point_path":init_path, 
                                       'height_offset':0.1,
                                       'robot_visible': True,
                                       'light_enabled': True}
env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)
adjust_usd_scale(scale=args_cli.scene_scale)
_,infos = env.reset()
# warm-up
PREHEAT_STEPS = 10
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
    obs, rewards, dones, infos = env.step(action)
    
camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

minco_adapter = None
if args_cli.enable_minco:
    from utils_tasks.sim_esdf_builder import SimEsdfBuilder
    from utils_tasks.navdp_minco_adapter import NavDPMincoAdapter
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    esdf_builder = SimEsdfBuilder(
        resolution=args_cli.esdf_resolution,
        padding=args_cli.esdf_padding,
        safe_dist=args_cli.minco_safe_dist,
        cache_name=args_cli.esdf_cache_name,
        force_rebuild=args_cli.esdf_force_rebuild,
        obstacle_min_height=args_cli.esdf_obstacle_min_height,
        obstacle_max_height=args_cli.esdf_obstacle_max_height,
        fill_footprint=bool(args_cli.esdf_fill_footprint),
        footprint_inflate_cells=args_cli.esdf_footprint_inflate_cells,
    )
    esdf = esdf_builder.build_or_load_from_stage(
        stage=stage,
        scene_path=scene_path,
        scene_scale=args_cli.scene_scale,
        env_prim_path="/World/Scene/terrain",
    )
    esdf_timing = esdf.get("timing", {}) if isinstance(esdf, dict) else {}
    print(f"[Timing][ESDF] {esdf_timing}")
    camera_pos_debug = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
    ok, dist = esdf_builder.query_grid(esdf, camera_pos_debug[0, :2])
    print(f"[SimESDF] initial camera query ok={ok} dist={dist}")
    minco_adapter = NavDPMincoAdapter(
        esdf=esdf,
        safe_dist=args_cli.minco_safe_dist,
        top_k=args_cli.minco_top_k,
        sample_dt=args_cli.minco_sample_dt,
        speed=args_cli.speed,
        max_vel=args_cli.minco_max_vel,
        max_acc=args_cli.minco_max_acc,
        max_iterations=args_cli.minco_max_iterations,
        max_yaw_rate=args_cli.mpc_max_yaw_rate,
        enable=True,
    )

planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic, minco_adapter))
planning_thread_obj.daemon = True
planning_thread_obj.start()

controller_kwargs = {
    "name": "simple_control",
    "wheel_radius": DINGO_WHEEL_RADIUS,
    "wheel_base": DINGO_WHEEL_BASE,
}
if wheel_constraint_enabled:
    controller_kwargs["max_wheel_speed"] = mpc_max_wheel_speed
controller = DifferentialController(**controller_kwargs)
with navdp_http_lock:
    algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
save_dir = "./pointgoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
os.makedirs(save_dir,exist_ok=True)

euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]
video_frame_shapes = [None for _ in range(args_cli.num_envs)]
video_writer_failed = [False for _ in range(args_cli.num_envs)]

trajectory_length = np.zeros((scene_config.num_envs))
frame_idx = 0

try:
    while simulation_app.is_running() and not stop_event.is_set():
        with torch.inference_mode():
            goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
            images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
            depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
            # get all camera poses
            camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
            camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
            camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
            camera_rot = R.from_quat(camera_rot_quat).as_matrix()
            robot_pos_w = env.unwrapped.scene.articulations['robot'].data.root_pos_w[:, :3].cpu().numpy()
            robot_quat_w = env.unwrapped.scene.articulations['robot'].data.root_quat_w.cpu().numpy()
            robot_quat_xyzw = robot_quat_w[:, [1, 2, 3, 0]]
            robot_rot = R.from_quat(robot_quat_xyzw).as_matrix()
            robot_yaw_w = np.arctan2(robot_rot[:, 1, 0], robot_rot[:, 0, 0])
            robot_lin_vel_w_np = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[:, :3].cpu().numpy().copy()
            robot_lin_vel_w_xy = robot_lin_vel_w_np[:, :2]
            robot_ang_vel_batch = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[:, 2].cpu().numpy()
            robot_vel_batch = compute_forward_velocity(robot_lin_vel_w_xy, robot_yaw_w)

            if frame_idx % max(1, args_cli.timing_log_interval) == 0:
                offset_xy = camera_pos[0, :2] - robot_pos_w[0, :2]
                print(
                    f"[FrameCheck] camera-base offset xy={offset_xy} "
                    f"norm={np.linalg.norm(offset_xy):.3f}"
                )

            if np.any(reset_pending):
                for i in np.flatnonzero(reset_pending):
                    camera_xy = camera_pos[i, :2]
                    robot_xy = robot_pos_w[i, :2]
                    offset_norm = np.linalg.norm(camera_xy - robot_xy)
                    stable = (
                        np.all(np.isfinite(camera_xy))
                        and np.all(np.isfinite(robot_xy))
                        and np.isfinite(offset_norm)
                        and offset_norm <= RESET_CAMERA_BASE_OFFSET_TOL
                    )
                    reset_stable_count[i] = reset_stable_count[i] + 1 if stable else 0
                    if reset_stable_count[i] >= RESET_STABLE_REQUIRED_FRAMES:
                        reset_pending[i] = False
                        print(
                            f"[EpisodeReset] env={i} generation={episode_generation[i]} "
                            f"state=READY stable_frames={reset_stable_count[i]} offset={offset_norm:.3f}"
                        )

            if not np.any(reset_pending):
                with input_lock:
                    planning_input.current_goal = goals.copy()
                    planning_input.current_image = images.copy()
                    planning_input.current_depth = depths.copy()
                    planning_input.camera_pos = camera_pos.copy()
                    planning_input.camera_rot = camera_rot.copy()
                    planning_input.robot_pos_w = robot_pos_w.copy()
                    planning_input.robot_yaw_w = robot_yaw_w.copy()
                    planning_input.robot_lin_vel_w = robot_lin_vel_w_np.copy()
                    planning_input.robot_ang_vel_w = robot_ang_vel_batch.copy()
                    planning_input.episode_generation = episode_generation.copy()
                if reset_in_progress.is_set():
                    reset_in_progress.clear()

            # based on the current world trajectory
            if use_robot_base_frame:
                x0 = np.stack([
                    robot_pos_w[:, 0],
                    robot_pos_w[:, 1],
                    robot_yaw_w,
                    robot_vel_batch,
                    robot_ang_vel_batch,
                ], axis=-1)
            else:
                x0 = np.stack([
                    camera_pos[:, 0],
                    camera_pos[:, 1],
                    np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]),
                    robot_vel_batch,
                    robot_ang_vel_batch,
                ], axis=-1)
            current_trajectory = None
            current_all_trajectories = None
            current_all_values = None
            current_raw_top1 = None
            current_selected_candidate = None
            current_sparse_guide_points = None
            current_minco_samples = None
            current_minco_speed_profile = None
            current_minco_info = None
            current_per_env_plan_id = None
            current_stop_required = None
            current_minco_status = None
            current_planning_timing = None
            current_plan_id = -1
            current_episode_generation = None
            with output_lock:
                if planning_output.trajectory_points_world is not None:
                    current_plan_id = planning_output.plan_id
                    current_episode_generation = (
                        planning_output.episode_generation
                        if planning_output.episode_generation is not None
                        else None
                    )
                    current_trajectory = planning_output.trajectory_points_world
                    current_all_trajectories = planning_output.all_trajectories_world
                    current_all_values = planning_output.all_values_camera
                    current_raw_top1 = planning_output.raw_top1_world
                    current_selected_candidate = planning_output.selected_candidate_world
                    current_sparse_guide_points = planning_output.minco_sparse_waypoints_world
                    current_minco_samples = planning_output.minco_samples
                    current_minco_speed_profile = planning_output.minco_speed_profile
                    current_minco_info = planning_output.minco_info
                    current_per_env_plan_id = planning_output.per_env_plan_id
                    current_stop_required = planning_output.stop_required
                    current_minco_status = planning_output.minco_status
                    current_planning_timing = planning_output.planning_timing
        
            action_list = []
            control_timing_records = []
            video_frames = [None for _ in range(args_cli.num_envs)]
            control_states = ["WAITING_PLAN" for _ in range(args_cli.num_envs)]
            cmd_v_batch = np.zeros(args_cli.num_envs, dtype=np.float64)
            cmd_w_batch = np.zeros(args_cli.num_envs, dtype=np.float64)
            planned_v_batch = np.full(args_cli.num_envs, np.nan, dtype=np.float64)
            planned_w_batch = np.full(args_cli.num_envs, np.nan, dtype=np.float64)
            control_timers = [StageTimer() for _ in range(args_cli.num_envs)]

            for i in range(args_cli.num_envs):
                control_timer = control_timers[i]
                joint_velocities = np.zeros(2, dtype=np.float32)
                if reset_in_progress.is_set() or reset_pending[i]:
                    control_states[i] = "RESETTING"
                elif current_trajectory is None:
                    control_states[i] = "WAITING_PLAN"
                else:
                    output_generation_i = (
                        int(current_episode_generation[i])
                        if current_episode_generation is not None and i < len(current_episode_generation)
                        else -1
                    )
                    if output_generation_i != int(episode_generation[i]):
                        control_states[i] = "STALE_PLAN"
                    else:
                        stop_required_i = (
                            bool(current_stop_required[i])
                            if current_stop_required is not None and i < len(current_stop_required)
                            else False
                        )
                        env_plan_id = (
                            int(current_per_env_plan_id[i])
                            if current_per_env_plan_id is not None and i < len(current_per_env_plan_id)
                            else int(current_plan_id)
                        )
                        status_i = (
                            current_minco_status[i]
                            if current_minco_status is not None and i < len(current_minco_status)
                            else ("MINCO_OK" if args_cli.enable_minco else "RAW_NAVDP")
                        )
                        if stop_required_i:
                            if mpc[i] is not None:
                                mpc[i].reset()
                            control_states[i] = "MINCO_STOP"
                            if frame_idx % max(1, args_cli.timing_log_interval) == 0:
                                print(f"[ControlRef] env={i} status=MINCO_STOP cmd_v=0.00 cmd_w=0.00")
                        else:
                            reference_ok = True
                            if env_plan_id != last_applied_plan_id[i]:
                                path_i = np.asarray(current_trajectory[i], dtype=np.float64)
                                samples_i = (
                                    current_minco_samples[i]
                                    if current_minco_samples is not None and i < len(current_minco_samples)
                                    else None
                                )
                                creating_mpc = mpc[i] is None
                                try:
                                    if creating_mpc:
                                        control_dt = float(env.unwrapped.step_dt)
                                        mpc[i] = MPC_Controller(
                                            path_i,
                                            trajectory_samples=samples_i,
                                            desired_v=args_cli.speed,
                                            v_max=args_cli.speed,
                                            w_max=args_cli.mpc_max_yaw_rate,
                                            max_acc=args_cli.minco_max_acc,
                                            max_yaw_acc=args_cli.mpc_max_yaw_acc,
                                            T=control_dt,
                                            allow_geometric_fallback=not args_cli.enable_minco,
                                            wheel_radius=DINGO_WHEEL_RADIUS,
                                            wheel_base=DINGO_WHEEL_BASE,
                                            max_wheel_speed=mpc_max_wheel_speed,
                                        )
                                    else:
                                        reference_ok = mpc[i].update_reference(
                                            path_i,
                                            trajectory_samples=samples_i,
                                            desired_v=args_cli.speed,
                                        )
                                except Exception as exc:
                                    reference_ok = False
                                    if creating_mpc:
                                        mpc[i] = None
                                        control_states[i] = "MPC_UNAVAILABLE"
                                        print(f"[MPC] env={i} creation failed: {exc}")
                                    else:
                                        control_states[i] = "MPC_REFERENCE_REJECTED"
                                        print(f"[MPC] env={i} reference update failed: {exc}")
                                if reference_ok:
                                    last_applied_plan_id[i] = env_plan_id
                                elif control_states[i] not in ("MPC_UNAVAILABLE", "MPC_REFERENCE_REJECTED"):
                                    control_states[i] = "MPC_REFERENCE_REJECTED"
                                    print(f"[MPC] env={i} rejected missing MINCO samples, stopping")

                            mpc_i = mpc[i]
                            if reference_ok and mpc_i is None:
                                control_states[i] = "MPC_UNAVAILABLE"
                            elif reference_ok:
                                try:
                                    with control_timer.section("mpc_solve_ms"):
                                        opt_u_controls, opt_x_states = mpc_i.solve(x0[i, :5])
                                    v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                                    if not np.isfinite(v) or not np.isfinite(w):
                                        raise ValueError(f"non-finite MPC command v={v}, w={w}")
                                    v = float(np.clip(v, 0.0, args_cli.speed))
                                    w = float(np.clip(w, -args_cli.mpc_max_yaw_rate, args_cli.mpc_max_yaw_rate))
                                    cmd_v_batch[i] = v
                                    cmd_w_batch[i] = w
                                    joint_velocities = controller.forward(np.array([v, w])).joint_velocities
                                    control_states[i] = "CONTROL_ACTIVE"
                                    current_ref = mpc_i.get_current_reference()
                                    if current_ref is not None:
                                        if np.isfinite(current_ref[3]):
                                            planned_v_batch[i] = float(current_ref[3])
                                        if np.isfinite(current_ref[4]):
                                            planned_w_batch[i] = float(current_ref[4])
                                    if frame_idx % max(1, args_cli.timing_log_interval) == 0 and current_ref is not None:
                                        print(
                                            f"[ControlRef] env={i} plan={env_plan_id} status={status_i} idx={mpc_i.progress_idx} "
                                            f"actual_v={robot_vel_batch[i]:.2f} ref_v={planned_v_batch[i]:.2f} "
                                            f"ref_w={planned_w_batch[i]:.2f} cmd_v={v:.2f} cmd_w={w:.2f}"
                                        )
                                except Exception as exc:
                                    cmd_v_batch[i] = 0.0
                                    cmd_w_batch[i] = 0.0
                                    joint_velocities = np.zeros(2, dtype=np.float32)
                                    control_states[i] = "MPC_SOLVE_FAILED"
                                    print(f"[MPC] env={i} solve failed: {exc}")
                action_list.append(joint_velocities)

            for i in range(args_cli.num_envs):
                control_timer = control_timers[i]
                selected_candidate_index = (
                    int(current_minco_info[i].get("selected_index", -1))
                    if current_minco_info is not None
                    and i < len(current_minco_info)
                    and isinstance(current_minco_info[i], dict)
                    else -1
                )
                try:
                    with control_timer.section("visualize_ms"):
                        vis_image = vis_manager[i].visualize_trajectory(
                            images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                            current_trajectory[i] if current_trajectory is not None else None,
                            robot_pose=x0[i],
                            all_trajectories_points=current_all_trajectories[i] if current_all_trajectories is not None else None,
                            all_trajectories_values=current_all_values[i] if current_all_values is not None else None,
                            raw_trajectory_points=current_raw_top1[i] if current_raw_top1 is not None else None,
                            selected_candidate_points=current_selected_candidate[i] if current_selected_candidate is not None else None,
                            selected_candidate_index=selected_candidate_index,
                            sparse_guide_points=current_sparse_guide_points[i] if current_sparse_guide_points is not None else None,
                            esdf=minco_adapter.esdf if minco_adapter is not None else None,
                            minco_info=current_minco_info[i] if current_minco_info is not None else None,
                        )
                    planned_v = planned_v_batch[i] if np.isfinite(planned_v_batch[i]) else None
                    vis_manager[i].update_speed_history(
                        actual_v=float(robot_vel_batch[i]),
                        actual_w=float(robot_ang_vel_batch[i]),
                        cmd_v=float(cmd_v_batch[i]),
                        cmd_w=float(cmd_w_batch[i]),
                        planned_v=planned_v,
                    )
                    vis_image = vis_manager[i].append_speed_plot(
                        vis_image, speed_max=max(1.0, float(args_cli.speed) * 1.5)
                    )
                    with control_timer.section("text_overlay_ms"):
                        vis_image = draw_box_with_text(vis_image,0,0,650,50,f"state: {control_states[i]}")
                        vis_image = draw_box_with_text(vis_image,0,50,650,50,f"episode: {episode_num} generation: {episode_generation[i]}")
                        vis_image = draw_box_with_text(vis_image,0,100,430,50,"desired lin.:%.2f ang.:%.2f"%(cmd_v_batch[i],cmd_w_batch[i]))
                        vis_image = draw_box_with_text(vis_image,0,150,430,50,"actual lin.:%.2f ang.:%.2f"%(robot_vel_batch[i],robot_ang_vel_batch[i]))
                        vis_image = draw_box_with_text(vis_image,0,820,430,50,"point goal:(%.2f, %.2f)"%(goals[i][0],goals[i][1]))
                        if current_all_values is not None:
                            vis_image = draw_box_with_text(vis_image,0,770,430,50,"critic max:%.2f min:%.2f"%(np.max(current_all_values[i]), np.min(current_all_values[i])))
                        if current_minco_info is not None and isinstance(current_minco_info[i], dict):
                            info = current_minco_info[i]
                            vis_image = draw_box_with_text(
                                vis_image,0,870,520,50,"%s idx:%d esdf:%.2f cost:%.1f" % (
                                    info.get("status", "MINCO_OK" if info.get("success", False) else "MINCO_STOP"),
                                    info.get("selected_index", -1), info.get("min_esdf", float("nan")),
                                    info.get("objective", float("inf")),
                                )
                            )
                    control_timing = control_timer.snapshot()
                    control_timing["video_write_ms"] = 0.0
                    control_timing["env_step_ms"] = 0.0
                    if args_cli.show_timing_overlay:
                        vis_image = append_timing_panel(
                            vis_image, current_planning_timing, control_timing, env_index=i
                        )
                    video_frames[i] = vis_image
                except Exception as exc:
                    control_states[i] = "VISUALIZATION_FAILED"
                    print(f"[Video] env={i} episode={episode_num} state={control_states[i]} error={exc}")
                    try:
                        vis_image = vis_manager[i].visualize_trajectory(
                            images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                            None, robot_pose=x0[i], esdf=None,
                        )
                        vis_image = vis_manager[i].append_speed_plot(vis_image)
                        video_frames[i] = draw_box_with_text(vis_image,0,0,650,50,"state: VISUALIZATION_FAILED")
                    except Exception:
                        panel_count = 3 if vis_manager[i].show_all_candidates else 2
                        video_frames[i] = np.zeros(
                            (
                                images[i].shape[0] + 180,
                                images[i].shape[1] + images[i].shape[0] * (panel_count - 1),
                                3,
                            ),
                            dtype=np.uint8,
                        )

            for i, frame in enumerate(video_frames):
                try:
                    frame = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise ValueError(f"invalid frame shape {frame.shape}")
                    if video_frame_shapes[i] is None:
                        video_frame_shapes[i] = frame.shape
                    elif frame.shape != video_frame_shapes[i]:
                        raise ValueError(
                            f"frame shape mismatch expected={video_frame_shapes[i]} actual={frame.shape}"
                        )
                    if fps_writer[i] is not None and not stop_event.is_set():
                        with control_timers[i].section("video_write_ms"):
                            fps_writer[i].append_data(frame)
                    timing = control_timers[i].snapshot()
                    timing["env_step_ms"] = 0.0
                    control_timing_records.append(timing)
                except Exception as exc:
                    if not video_writer_failed[i]:
                        print(
                            f"[Video] env={i} episode={episode_num} state={control_states[i]} "
                            f"expected_shape={video_frame_shapes[i]} actual_shape={getattr(frame, 'shape', None)} error={exc}"
                        )
                        video_writer_failed[i] = True
                        if fps_writer[i] is not None:
                            try:
                                fps_writer[i].close()
                            except Exception:
                                pass
                            fps_writer[i] = None

            action = torch.as_tensor(np.stack(action_list, axis=0),device="cuda:0")
            env_step_timer = StageTimer()
            with env_step_timer.section("env_step_ms"):
                obs, rewards, dones, infos = env.step(action)
            env_step_ms = env_step_timer.records["env_step_ms"]
            for record in control_timing_records:
                record["env_step_ms"] = env_step_ms
            if control_timing_records and frame_idx % max(1, args_cli.timing_log_interval) == 0:
                print(format_control_summary(mean_timing(control_timing_records)))
            trajectory_length += (infos['observations']['policy'][:,0] * env.unwrapped.step_dt).cpu().numpy()
        
            for i in range(args_cli.num_envs):
                if stop_event.is_set():
                    break
                if dones[i] == True:
                    reset_in_progress.set()
                    episode_generation[i] += 1
                    print(f"[EpisodeReset] env={i} generation={episode_generation[i]} state=BEGIN")
                    if mpc[i] is not None:
                        mpc[i].reset()
                    last_applied_plan_id[i] = -1
                    with output_lock:
                        minco_hold_cache[i] = None
                        per_env_plan_id[i] += 1
                    reset_pending[i] = True
                    reset_stable_count[i] = 0
                    vis_manager[i].reset()
                    invalidate_planning_state_for_reset()
                    episode_num += 1
                    with navdp_http_lock:
                        navigator_reset(env_id=i,port=args_cli.port)
                    success_flag = (np.sqrt(np.square(goals[i]).sum())<1.0).astype(np.float32)
                    if fps_writer[i] is not None:
                        fps_writer[i].close()
                        fps_writer[i] = None
                    evaluation_metrics.append({'success':success_flag,
                                               'spl': np.clip(euclidean[i] / trajectory_length[i],0,1) * success_flag,
                                               'distance':euclidean[i]})
                    write_metrics(evaluation_metrics,save_dir+"metric.csv")
                    euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
                    if episode_num < args_cli.num_episodes:
                        fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
                        video_frame_shapes[i] = None
                        video_writer_failed[i] = False
                    trajectory_length[i] = 0.0
        
            if episode_num >= args_cli.num_episodes:
                break
            frame_idx += 1

except KeyboardInterrupt:
    print("[Shutdown] KeyboardInterrupt received.")
finally:
    stop_event.set()
    try:
        planning_thread_obj.join(timeout=6.0)
    except Exception as exc:
        print(f"[Shutdown] planning thread join failed: {exc}")
    try:
        navigator_close(
            port=args_cli.port,
            intrinsic=camera_intrinsic.cpu().numpy(),
            stop_threshold=args_cli.stop_threshold,
            batch_size=scene_config.num_envs,
        )
    except Exception as exc:
        print(f"[Shutdown] NavDP video writer close failed: {exc}")
    for writer in fps_writer:
        try:
            if writer is not None:
                writer.close()
        except Exception as exc:
            print(f"[Shutdown] video writer close failed: {exc}")
    try:
        env.close()
    except Exception as exc:
        print(f"[Shutdown] env close failed: {exc}")
    try:
        simulation_app.close()
    except Exception as exc:
        print(f"[Shutdown] simulation app close failed: {exc}")
