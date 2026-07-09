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
    "--speed", type=float, default=0.5)
parser.add_argument(
    "--port", type=int, default=8888)
parser.add_argument("--enable_minco", action="store_true")
parser.add_argument("--minco_top_k", type=int, default=4)
parser.add_argument("--minco_safe_dist", type=float, default=0.30)
parser.add_argument("--minco_sample_dt", type=float, default=0.05)
parser.add_argument("--minco_fallback_to_raw", action="store_true", default=True)
parser.add_argument("--esdf_resolution", type=float, default=0.05)
parser.add_argument("--esdf_padding", type=float, default=1.0)
parser.add_argument("--esdf_force_rebuild", action="store_true")
parser.add_argument("--esdf_cache_name", type=str, default="esdf_2d.npz")
parser.add_argument("--esdf_obstacle_min_height", type=float, default=0.08)
parser.add_argument("--esdf_obstacle_max_height", type=float, default=1.50)
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

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text,adjust_usd_scale
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_reset,pointgoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller
from utils_tasks.timing_utils import (
    StageTimer,
    draw_timing_overlay,
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

def planning_thread(env, camera_intrinsic, minco_adapter=None):
    global mpc
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
                    goal = planning_input.current_goal.copy()
                    image = planning_input.current_image.copy()
                    depth = planning_input.current_depth.copy()
                    camera_pos = planning_input.camera_pos.copy()
                    camera_rot = planning_input.camera_rot.copy()
                    robot_lin_vel_w = planning_input.robot_lin_vel_w.copy() if planning_input.robot_lin_vel_w is not None else None
                    robot_ang_vel_w = planning_input.robot_ang_vel_w.copy() if planning_input.robot_ang_vel_w is not None else None
            with output_lock:
                planning_output.is_planning = True
            
            with planning_timer.section("navdp_step_ms"):
                trajectory_points_camera, all_trajectories_camera, all_values_camera = pointgoal_step(goal, image, depth,port=args_cli.port)

            with planning_timer.section("raw_transform_ms"):
                raw_top1_world = []
                for idx in range(trajectory_points_camera.shape[0]):
                    trajectory_points_world = []
                    for i, point in enumerate(trajectory_points_camera[idx]):
                        if i < 0:
                            continue
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        trajectory_points_world.append(point_world[:2])
                    trajectory_points_world = np.array(trajectory_points_world)
                    raw_top1_world.append(trajectory_points_world)
                raw_top1_world = np.array(raw_top1_world, dtype=object)

            with planning_timer.section("candidate_transform_ms"):
                batch_all_points_world = []
                for idx in range(all_trajectories_camera.shape[0]):
                    # Transform all trajectories
                    all_trajectories_world = []
                    for traj_camera in all_trajectories_camera[idx]:
                        traj_world = []
                        for point in traj_camera:
                            point_local = np.array([point[0], point[1], 0.0])
                            point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                            traj_world.append(point_world[:3])
                        all_trajectories_world.append(np.array(traj_world))
                    batch_all_points_world.append(all_trajectories_world)
                batch_all_points_world = np.array(batch_all_points_world, dtype=object)

            batch_optimal_points_world = []
            batch_raw_top1_world = []
            batch_selected_candidate_world = []
            batch_control_points_world = []
            batch_minco_samples = []
            batch_minco_speed_profile = []
            batch_minco_info = []
            used_minco = minco_adapter is not None and minco_adapter.enabled
            with planning_timer.section("state_build_ms"):
                states = []
                if used_minco:
                    for idx in range(camera_pos.shape[0]):
                        vel = robot_lin_vel_w[idx].copy() if robot_lin_vel_w is not None else np.zeros(3)
                        vel[2] = 0.0
                        yaw_rate = float(robot_ang_vel_w[idx]) if robot_ang_vel_w is not None else 0.0
                        states.append({
                            "position": np.array([camera_pos[idx, 0], camera_pos[idx, 1], 0.0]),
                            "velocity": vel,
                            "acceleration": np.zeros(3),
                            "yaw": float(np.arctan2(camera_rot[idx, 1, 0], camera_rot[idx, 0, 0])),
                            "yaw_rate": yaw_rate,
                        })

            minco_results = []
            if used_minco:
                with planning_timer.section("minco_total_ms"):
                    minco_results = minco_adapter.optimize_candidates(
                        candidates_world=batch_all_points_world,
                        critic_values=all_values_camera,
                        states=states,
                        raw_top1_world=raw_top1_world,
                    )
            else:
                planning_timer.records["minco_total_ms"] = 0.0

            with planning_timer.section("mpc_construct_ms"):
                next_mpc = [None for _ in range(args_cli.num_envs)]
                if used_minco:
                    for idx, result in enumerate(minco_results):
                        if result["success"] and result["waypoints"] is not None and len(result["waypoints"]) >= 2:
                            trajectory_points_world = np.asarray(result["waypoints"])[:, :2]
                        else:
                            trajectory_points_world = np.asarray(raw_top1_world[idx])[:, :2]
                        batch_optimal_points_world.append(trajectory_points_world)
                        next_mpc[idx] = MPC_Controller(trajectory_points_world,
                                                       desired_v=args_cli.speed,
                                                       v_max=args_cli.speed,
                                                       w_max=args_cli.speed)
                        batch_raw_top1_world.append(result.get("raw_top1"))
                        batch_selected_candidate_world.append(result.get("selected_candidate"))
                        batch_control_points_world.append(result.get("control_points"))
                        batch_minco_samples.append(result.get("samples"))
                        batch_minco_speed_profile.append(result.get("speed_profile"))
                        batch_minco_info.append(result)
                else:
                    for idx in range(len(raw_top1_world)):
                        trajectory_points_world = np.asarray(raw_top1_world[idx])[:, :2]
                        batch_optimal_points_world.append(trajectory_points_world)
                        next_mpc[idx] = MPC_Controller(trajectory_points_world,
                                                       desired_v=args_cli.speed,
                                                       v_max=args_cli.speed,
                                                       w_max=args_cli.speed)
                        batch_raw_top1_world.append(trajectory_points_world)
                        batch_selected_candidate_world.append(None)
                        batch_control_points_world.append(None)
                        batch_minco_samples.append(None)
                        batch_minco_speed_profile.append(None)
                        batch_minco_info.append({
                            "success": False,
                            "fallback": True,
                            "selected_index": -1,
                            "objective": float("inf"),
                            "min_esdf": float("nan"),
                            "adapter_total_ms": 0.0,
                            "selected_cpp_optimize_time_ms": float("nan"),
                        })
                mpc = next_mpc

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
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world_vis
                planning_output.all_values_camera = all_values_camera
                planning_output.raw_top1_world = np.array(batch_raw_top1_world, dtype=object)
                planning_output.selected_candidate_world = np.array(batch_selected_candidate_world, dtype=object)
                planning_output.minco_control_points_world = np.array(batch_control_points_world, dtype=object)
                planning_output.minco_samples = batch_minco_samples
                planning_output.minco_speed_profile = batch_minco_speed_profile
                planning_output.minco_info = batch_minco_info
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
        enable=True,
        fallback_to_raw=args_cli.minco_fallback_to_raw,
    )

planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic, minco_adapter))
planning_thread_obj.daemon = True
planning_thread_obj.start()

controller = DifferentialController(name="simple_control", 
                                    wheel_radius=DINGO_WHEEL_RADIUS,
                                    wheel_base=DINGO_WHEEL_BASE)
algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
save_dir = "./pointgoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
os.makedirs(save_dir,exist_ok=True)

euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]

trajectory_length = np.zeros((scene_config.num_envs))
frame_idx = 0

while simulation_app.is_running():
    with torch.inference_mode():
        goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
        images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
        # get all camera poses
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        
        with input_lock:
            planning_input.current_goal = goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()
            planning_input.robot_lin_vel_w = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[:, :3].cpu().numpy().copy()
            planning_input.robot_ang_vel_w = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[:, 2].cpu().numpy().copy()

        # based on the current world trajectory
        robot_vel_batch = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[:, :2].norm(dim=1).cpu().numpy()
        robot_ang_vel_batch = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[:, 2].cpu().numpy()

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
        current_control_points = None
        current_minco_samples = None
        current_minco_speed_profile = None
        current_minco_info = None
        current_planning_timing = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy() if planning_output.trajectory_points_world is not None else None
                current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
                current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None
                current_raw_top1 = planning_output.raw_top1_world.copy() if planning_output.raw_top1_world is not None else None
                current_selected_candidate = planning_output.selected_candidate_world.copy() if planning_output.selected_candidate_world is not None else None
                current_control_points = planning_output.minco_control_points_world.copy() if planning_output.minco_control_points_world is not None else None
                current_minco_samples = planning_output.minco_samples
                current_minco_speed_profile = planning_output.minco_speed_profile
                current_minco_info = planning_output.minco_info
                current_planning_timing = planning_output.planning_timing.copy() if planning_output.planning_timing is not None else None
        
        if current_trajectory is not None:
            action_list = []
            control_timing_records = []
            for i in range(args_cli.num_envs):
                control_timer = StageTimer()

                with control_timer.section("visualize_ms"):
                    vis_image = vis_manager[i].visualize_trajectory(
                        images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                        current_trajectory[i],
                        robot_pose=x0[i],
                        all_trajectories_points=current_all_trajectories[i] if current_all_trajectories is not None else None,
                        all_trajectories_values=current_all_values[i] if current_all_values is not None else None,
                        raw_trajectory_points=current_raw_top1[i] if current_raw_top1 is not None else None,
                        selected_candidate_points=current_selected_candidate[i] if current_selected_candidate is not None else None,
                        control_points=current_control_points[i] if current_control_points is not None else None,
                        esdf=minco_adapter.esdf if minco_adapter is not None else None,
                        minco_info=current_minco_info[i] if current_minco_info is not None else None,
                    )

                mpc_i = mpc[i] if isinstance(mpc, list) and i < len(mpc) else mpc
                if mpc_i is None:
                    action_list.append(np.zeros(2, dtype=np.float32))
                    continue
                with control_timer.section("mpc_solve_ms"):
                    opt_u_controls, opt_x_states = mpc_i.solve(x0[i, :3])
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                action = torch.tensor([v, w], device="cuda:0")
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                action_list.append(joint_velocities)

                planned_v = None
                if current_minco_speed_profile is not None:
                    profile = current_minco_speed_profile[i]
                    if profile is not None:
                        profile_np = np.asarray(profile, dtype=np.float64).reshape(-1)
                        if profile_np.size > 0 and np.isfinite(profile_np[0]):
                            planned_v = float(profile_np[0])

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
                            status = "MINCO ok" if info.get("success", False) else "MINCO fallback"
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
                        vis_image = draw_timing_overlay(vis_image, current_planning_timing, control_timing)

                    cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    with control_timer.section("video_write_ms"):
                        fps_writer[i].append_data(vis_image)
                    control_timing = control_timer.snapshot()
                    control_timing["env_step_ms"] = 0.0
                    control_timing_records.append(control_timing)
                except:
                    pass
                
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
            if dones[i] == True:
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
       
                
   

        
