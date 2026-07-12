import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A script to run a car control simulation")
parser.add_argument(
    "--scene_dir", type=str, default="./assets/scenes/cluttered_hard")
parser.add_argument(
    "--scene_index", type=int, default=8)
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0)
parser.add_argument(
    "--num_envs", type=int, default=1)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--speed", type=float, default=1.0)
parser.add_argument("--mpc_max_yaw_rate", type=float, default=1.0)
parser.add_argument("--mpc_max_yaw_acc", type=float, default=2.0)
parser.add_argument(
    "--mpc_max_wheel_speed",
    type=float,
    default=0.0,
    help="Maximum wheel-joint angular speed in rad/s; <= 0 disables the MPC wheel constraint",
)
parser.add_argument(
    "--port", type=int, default=8888)
parser.add_argument("--enable_minco", action="store_true")
parser.add_argument("--minco_top_k", type=int, default=4)
parser.add_argument("--minco_safe_dist", type=float, default=0.30)
parser.add_argument("--minco_sample_dt", type=float, default=0.05)
parser.add_argument("--minco_max_vel", type=float, default=1.5)
parser.add_argument("--minco_max_acc", type=float, default=1.0)
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
parser.add_argument("--timing_log_interval", type=int, default=10)
parser.add_argument("--show_timing_overlay", action="store_true")
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
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = [None for _ in range(args_cli.num_envs)]
last_applied_plan_id = [-1 for _ in range(args_cli.num_envs)]
per_env_plan_id = np.zeros(args_cli.num_envs, dtype=np.int64)
minco_hold_cache = [None for _ in range(args_cli.num_envs)]
episode_generation = np.zeros(args_cli.num_envs, dtype=np.int64)
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

def planning_thread(env, camera_intrinsic, minco_adapter=None):
    """Thread function that continuously plans trajectories"""
    planning_iter = 0
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
            planning_timer = StageTimer()
            planning_total_start = planning_timer.now()
            with planning_timer.section("input_copy_ms"):
                with input_lock:
                    if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                        time.sleep(0.01)
                        continue
                    goal = planning_input.current_goal
                    image = planning_input.current_image
                    depth = planning_input.current_depth
                    camera_pos = planning_input.camera_pos
                    camera_rot = planning_input.camera_rot
                    robot_pos_w = planning_input.robot_pos_w
                    robot_yaw_w = planning_input.robot_yaw_w
                    robot_lin_vel_w = planning_input.robot_lin_vel_w
                    robot_ang_vel_w = planning_input.robot_ang_vel_w
                    input_episode_generation = (
                        planning_input.episode_generation
                        if planning_input.episode_generation is not None
                        else np.zeros(args_cli.num_envs, dtype=np.int64)
                    )
            with output_lock:
                planning_output.is_planning = True
            
            with planning_timer.section("navdp_step_ms"):
                trajectory_points_camera, all_trajectories_camera, all_values_camera = pointgoal_step(goal, image, depth,port=args_cli.port)
            if stop_event.is_set():
                break

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
                    break
            else:
                planning_timer.records["minco_total_ms"] = 0.0

            with planning_timer.section("mpc_construct_ms"):
                if used_minco:
                    for idx, result in enumerate(minco_results):
                        if result["success"] and result["waypoints"] is not None and len(result["waypoints"]) >= 2:
                            trajectory_points_world = np.asarray(result["waypoints"])[:, :2]
                            cache = minco_cache_entry(result, input_episode_generation[idx])
                            if cache is not None:
                                minco_hold_cache[idx] = cache
                                per_env_plan_id[idx] += 1
                                batch_per_env_plan_id[idx] = per_env_plan_id[idx]
                            status = "MINCO_OK"
                            stop_required = False
                            info = dict(result)
                            info["status"] = status
                        else:
                            if is_minco_cache_valid(minco_hold_cache[idx], input_episode_generation[idx]):
                                cache = minco_hold_cache[idx]
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
                            result.get("selected_candidate") if status != "MINCO_HOLD_LAST" else minco_hold_cache[idx].get("selected_candidate")
                        )
                        batch_sparse_waypoints_world.append(
                            result.get("sparse_waypoints") if status == "MINCO_OK" else (
                                minco_hold_cache[idx].get("sparse_waypoints") if status == "MINCO_HOLD_LAST" else None
                            )
                        )
                        batch_minco_samples.append(result.get("samples") if status == "MINCO_OK" else None)
                        batch_minco_speed_profile.append(
                            result.get("speed_profile") if status == "MINCO_OK" else (
                                minco_hold_cache[idx].get("speed_profile") if status == "MINCO_HOLD_LAST" else None
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

            # Update shared state
            with output_lock:
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
algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
save_dir = "./pointgoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
os.makedirs(save_dir,exist_ok=True)

euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]

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
        
            if current_trajectory is not None:
                action_list = []
                control_timing_records = []
                for i in range(args_cli.num_envs):
                    control_timer = StageTimer()
                    output_generation_i = (
                        int(current_episode_generation[i])
                        if current_episode_generation is not None and i < len(current_episode_generation)
                        else -1
                    )
                    if output_generation_i != int(episode_generation[i]):
                        action_list.append(np.zeros(2, dtype=np.float32))
                        continue

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
                        action_list.append(np.zeros(2, dtype=np.float32))
                        if frame_idx % max(1, args_cli.timing_log_interval) == 0:
                            print(f"[ControlRef] env={i} status=MINCO_STOP cmd_v=0.00 cmd_w=0.00")
                        continue

                    if env_plan_id != last_applied_plan_id[i]:
                        path_i = np.asarray(current_trajectory[i], dtype=np.float64)
                        samples_i = (
                            current_minco_samples[i]
                            if current_minco_samples is not None and i < len(current_minco_samples)
                            else None
                        )
                        if mpc[i] is None:
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
                            updated = mpc[i].update_reference(
                                path_i,
                                trajectory_samples=samples_i,
                                desired_v=args_cli.speed,
                            )
                            if not updated:
                                action_list.append(np.zeros(2, dtype=np.float32))
                                print(f"[MPC] env={i} rejected missing MINCO samples, stopping")
                                continue
                        last_applied_plan_id[i] = env_plan_id

                    with control_timer.section("visualize_ms"):
                        vis_image = vis_manager[i].visualize_trajectory(
                            images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                            current_trajectory[i],
                            robot_pose=x0[i],
                            all_trajectories_points=current_all_trajectories[i] if current_all_trajectories is not None else None,
                            all_trajectories_values=current_all_values[i] if current_all_values is not None else None,
                            raw_trajectory_points=current_raw_top1[i] if current_raw_top1 is not None else None,
                            selected_candidate_points=current_selected_candidate[i] if current_selected_candidate is not None else None,
                            sparse_guide_points=current_sparse_guide_points[i] if current_sparse_guide_points is not None else None,
                            esdf=minco_adapter.esdf if minco_adapter is not None else None,
                            minco_info=current_minco_info[i] if current_minco_info is not None else None,
                        )

                    if (
                        minco_adapter is not None
                        and current_minco_info is not None
                        and current_trajectory is not None
                        and frame_idx % max(1, args_cli.timing_log_interval) == 0
                    ):
                        py_min = query_esdf_polyline(minco_adapter.esdf, current_trajectory[i])
                        cpp_min = current_minco_info[i].get("min_esdf", float("nan"))
                        adapter_py_min = current_minco_info[i].get("py_min_esdf", float("nan"))
                        print(
                            f"[ESDFCheck] env={i} cpp_min={cpp_min:.3f} "
                            f"adapter_py_min={adapter_py_min:.3f} vis_py_min={py_min:.3f}"
                        )

                    mpc_i = mpc[i]
                    if mpc_i is None:
                        action_list.append(np.zeros(2, dtype=np.float32))
                        continue
                    with control_timer.section("mpc_solve_ms"):
                        opt_u_controls, opt_x_states = mpc_i.solve(x0[i, :5])
                    v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                    if not np.isfinite(v) or not np.isfinite(w):
                        v, w = 0.0, 0.0
                    v = float(np.clip(v, 0.0, args_cli.speed))
                    w = float(np.clip(w, -args_cli.mpc_max_yaw_rate, args_cli.mpc_max_yaw_rate))
                    action = torch.tensor([v, w], device="cuda:0")
                    action_cpu = action.cpu().numpy()
                    joint_velocities = controller.forward(action_cpu).joint_velocities
                    action_list.append(joint_velocities)

                    current_ref = mpc_i.get_current_reference()
                    planned_v = float(current_ref[3]) if current_ref is not None and np.isfinite(current_ref[3]) else None
                    planned_w = float(current_ref[4]) if current_ref is not None and np.isfinite(current_ref[4]) else None
                    if frame_idx % max(1, args_cli.timing_log_interval) == 0 and current_ref is not None:
                        ref_v_print = planned_v if planned_v is not None else float("nan")
                        ref_w_print = planned_w if planned_w is not None else float("nan")
                        print(
                            f"[ControlRef] env={i} plan={env_plan_id} status={status_i} idx={mpc_i.progress_idx} "
                            f"actual_v={robot_vel_batch[i]:.2f} ref_v={ref_v_print:.2f} "
                            f"ref_w={ref_w_print:.2f} cmd_v={v:.2f} cmd_w={w:.2f}"
                        )

                    with control_timer.section("speed_plot_ms"):
                        vis_manager[i].update_speed_history(
                            actual_v=float(robot_vel_batch[i]),
                            actual_w=float(robot_ang_vel_batch[i]),
                            cmd_v=float(v),
                            cmd_w=float(w),
                            planned_v=planned_v,
                        )
                        vis_image = vis_manager[i].append_speed_plot(
                            vis_image,
                            speed_max=max(1.0, float(args_cli.speed) * 1.5),
                        )

                    try:
                        with control_timer.section("text_overlay_ms"):
                            vis_image = draw_box_with_text(vis_image,0,0,430,50,"desired lin.:%.2f ang.:%.2f"%(v,w))
                            vis_image = draw_box_with_text(vis_image,0,50,430,50,"actual lin.:%.2f ang.:%.2f"%(robot_vel_batch[i],robot_ang_vel_batch[i]))
                            if current_all_values is not None:
                                vis_image = draw_box_with_text(vis_image,0,770,430,50,"critic max:%.2f min:%.2f"%(np.max(current_all_values[i]), np.min(current_all_values[i])))
                            vis_image = draw_box_with_text(vis_image,0,820,430,50,"point goal:(%.2f, %.2f)"%(goals[i][0],goals[i][1]))
                            if current_minco_info is not None:
                                info = current_minco_info[i]
                                status = info.get("status", "MINCO_OK" if info.get("success", False) else "MINCO_STOP")
                                vis_image = draw_box_with_text(
                                    vis_image,
                                    0,
                                    870,
                                    520,
                                    50,
                                    "%s idx:%d esdf:%.2f cost:%.1f" % (
                                        status,
                                        info.get("selected_index", -1),
                                        info.get("min_esdf", float("nan")),
                                        info.get("objective", float("inf")),
                                    ),
                                )

                        control_timing = control_timer.snapshot()
                        control_timing["video_write_ms"] = 0.0
                        control_timing["env_step_ms"] = 0.0

                        if args_cli.show_timing_overlay:
                            vis_image = append_timing_panel(
                                vis_image,
                                current_planning_timing,
                                control_timing,
                                env_index=i,
                            )

                        if not stop_event.is_set():
                            vis_image = np.ascontiguousarray(np.asarray(vis_image, dtype=np.uint8))
                            cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                            with control_timer.section("video_write_ms"):
                                fps_writer[i].append_data(vis_image)
                            control_timing = control_timer.snapshot()
                            control_timing["env_step_ms"] = 0.0
                            control_timing_records.append(control_timing)
                    except Exception as exc:
                        print(f"[Video] write failed env={i}: {exc}")
                
                action = torch.as_tensor(np.stack(action_list, axis=0),device="cuda:0")
                env_step_timer = StageTimer()
                with env_step_timer.section("env_step_ms"):
                    obs, rewards, dones, infos = env.step(action)
                env_step_ms = env_step_timer.records["env_step_ms"]
                for record in control_timing_records:
                    record["env_step_ms"] = env_step_ms
                if control_timing_records and frame_idx % max(1, args_cli.timing_log_interval) == 0:
                    print(format_control_summary(mean_timing(control_timing_records)))
                # Get actual joint velocities from Isaac Sim
                actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :2].cpu().numpy()
                desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :2].cpu().numpy()
                trajectory_length += (infos['observations']['policy'][:,0] * env.unwrapped.step_dt).cpu().numpy()
            else:
                action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
                env_step_timer = StageTimer()
                with env_step_timer.section("env_step_ms"):
                    obs, rewards, dones, infos = env.step(action)
                env_step_ms = env_step_timer.records["env_step_ms"]
                print("No trajectory available, using zero action")
        
            for i in range(args_cli.num_envs):
                if stop_event.is_set():
                    break
                if dones[i] == True:
                    episode_generation[i] += 1
                    if mpc[i] is not None:
                        mpc[i].reset()
                    last_applied_plan_id[i] = -1
                    minco_hold_cache[i] = None
                    per_env_plan_id[i] += 1
                    with input_lock:
                        if planning_input.episode_generation is None:
                            planning_input.episode_generation = episode_generation.copy()
                        else:
                            planning_input.episode_generation = planning_input.episode_generation.copy()
                            planning_input.episode_generation[i] = episode_generation[i]
                    with output_lock:
                        if planning_output.episode_generation is None:
                            planning_output.episode_generation = np.full(args_cli.num_envs, -1, dtype=np.int64)
                        else:
                            planning_output.episode_generation = planning_output.episode_generation.copy()
                        planning_output.episode_generation[i] = episode_generation[i] - 1
                    episode_num += 1
                    navigator_reset(env_id=i,port=args_cli.port)
                    success_flag = (np.sqrt(np.square(goals[i]).sum())<1.0).astype(np.float32)
                    fps_writer[i].close()
                    evaluation_metrics.append({'success':success_flag,
                                               'spl': np.clip(euclidean[i] / trajectory_length[i],0,1) * success_flag,
                                               'distance':euclidean[i]})
                    write_metrics(evaluation_metrics,save_dir+"metric.csv")
                    euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
                    fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
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
