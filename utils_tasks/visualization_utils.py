import numpy as np
import cv2
from collections import deque
from scipy.ndimage import binary_dilation

RGB_GREEN = (0, 255, 0)
RGB_ORANGE = (255, 165, 0)
RGB_YELLOW = (255, 255, 0)
RGB_CYAN = (0, 255, 255)
RGB_RED = (255, 0, 0)
RGB_BLUE = (0, 0, 255)
RGB_WHITE = (255, 255, 255)
RGB_GRAY = (128, 128, 128)
RGB_BLACK = (0, 0, 0)


class VisualizationManager:
    def __init__(self, history_size=5, show_all_candidates=False):
        self.history_size = history_size
        self.occupancy_history = deque(maxlen=history_size)  # Will store (grid, min_coords, robot_pose)
        self.resolution = 0.05  # 5cm per pixel
        self.inflation = 5      # inflation radius in pixels
        self.speed_history = deque(maxlen=150)
        self.esdf_overlay_cache = None
        self.esdf_overlay_pose = None
        self.show_all_candidates = bool(show_all_candidates)
    
    def reset(self):
        self.occupancy_history.clear()
        self.speed_history.clear()
        self.esdf_overlay_cache = None
        self.esdf_overlay_pose = None
        
    def build_occupancy_grid(self, depth_map, intrinsic, camera_roll=0):
        try:
            """Convert depth image to occupancy grid in BEV"""
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:, :, 0]

            depth_map = np.asarray(depth_map, dtype=np.float32)
            depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

            stride = 4
            depth_small = depth_map[::stride, ::stride]
            h, w = depth_small.shape
            if h <= 0 or w <= 0:
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])

            uu, vv = np.meshgrid(
                np.arange(0, depth_map.shape[1], stride)[:w],
                np.arange(0, depth_map.shape[0], stride)[:h],
            )

            z = depth_small
            valid_mask = (z > 0.05) & np.isfinite(z) & (z < 8.0)
            if np.count_nonzero(valid_mask) < 50:
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])

            zv = z[valid_mask]
            x = (uu[valid_mask] - intrinsic[0, 2]) * zv / intrinsic[0, 0]
            y = (vv[valid_mask] - intrinsic[1, 2]) * zv / intrinsic[1, 1]
            points_3d = np.stack((x, y, zv), axis=-1)
            
            # Apply camera roll
            roll = camera_roll * np.pi / 180
            rotation_matrix_x = np.array([[1, 0, 0], 
                                        [0, np.cos(roll), -np.sin(roll)], 
                                        [0, np.sin(roll), np.cos(roll)]])
            point_3d_flat = (rotation_matrix_x @ points_3d.transpose()).transpose()
            
            # Transform to world coordinates
            point_3d_world = np.zeros((point_3d_flat.shape[0], 3))
            point_3d_world[:, 0] = point_3d_flat[:, 2]
            point_3d_world[:, 1] = -point_3d_flat[:, 0]
            point_3d_world[:, 2] = -point_3d_flat[:, 1]

            if point_3d_world.shape[0] == 0:
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])

            point_3d_world = point_3d_world[np.all(np.isfinite(point_3d_world), axis=1)]
            if point_3d_world.shape[0] == 0:
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])

            z_min = float(np.min(point_3d_world[:, 2]))
            z_max = float(np.max(point_3d_world[:, 2]))
            bins = np.arange(z_min, z_max, 0.05)
            if bins.size >= 2:
                hist, bin_edges = np.histogram(point_3d_world[:, 2], bins=bins)
                max_freq_index = np.argmax(hist)
                point_3d_world[:, 2] -= bin_edges[max_freq_index]
            else:
                point_3d_world[:, 2] -= -0.5
            
            # Filter points within height range
            filtered_points = point_3d_world[(point_3d_world[:, 2] >= 0.2) & (point_3d_world[:, 2] <= 1.5)]
            if filtered_points.shape[0] == 0:
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])

            filtered_points = filtered_points[np.all(np.isfinite(filtered_points[:, :2]), axis=1)]
            if filtered_points.shape[0] == 0:
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])
                
            # Create occupancy grid
            min_coords = np.min(filtered_points, axis=0)
            max_coords = np.max(filtered_points, axis=0)
            if not np.all(np.isfinite(min_coords)) or not np.all(np.isfinite(max_coords)):
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])

            grid_size_float = np.ceil((max_coords - min_coords) / self.resolution + 1)
            max_grid_size = 400
            if (
                not np.all(np.isfinite(grid_size_float)) or
                grid_size_float[0] <= 0 or grid_size_float[1] <= 0 or
                grid_size_float[0] > max_grid_size or grid_size_float[1] > max_grid_size
            ):
                return np.zeros((100, 100), dtype=np.int8), np.array([0.0, 0.0])

            grid_size = grid_size_float.astype(int)
            occupancy_grid = np.zeros(grid_size[:2], dtype=np.int8)
            
            grid_coords = ((filtered_points[:, :2] - min_coords[:2]) / self.resolution).astype(int)
            valid_grid = (
                (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < occupancy_grid.shape[0]) &
                (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < occupancy_grid.shape[1])
            )
            grid_coords = grid_coords[valid_grid]
            occupancy_grid[grid_coords[:, 0], grid_coords[:, 1]] = 1
            
        except Exception:
            occupancy_grid = np.zeros((100, 100), dtype=np.int8)
            min_coords = np.array([0.0, 0.0])
        
        return occupancy_grid, min_coords

    def world_to_vis_points(self, points_xy, robot_pose, grid_size, center_offset):
        points_xy = np.asarray(points_xy, dtype=np.float64)
        if points_xy.ndim != 2 or points_xy.shape[1] < 2 or points_xy.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int32)

        dx = points_xy[:, 0] - robot_pose[0]
        dy = points_xy[:, 1] - robot_pose[1]

        grid_points = np.stack([dx, dy], axis=1) / self.resolution
        grid_points = grid_points.astype(np.int32)

        valid_mask = (
            (np.abs(grid_points[:, 0]) < grid_size // 2) &
            (np.abs(grid_points[:, 1]) < grid_size // 2)
        )
        grid_points = grid_points[valid_mask]

        vis_points = np.zeros_like(grid_points)
        vis_points[:, 0] = -grid_points[:, 1] + center_offset
        vis_points[:, 1] = -grid_points[:, 0] + center_offset

        valid_mask = (
            (vis_points[:, 0] >= 0) & (vis_points[:, 0] < grid_size) &
            (vis_points[:, 1] >= 0) & (vis_points[:, 1] < grid_size)
        )
        return vis_points[valid_mask].astype(np.int32)

    def draw_polyline_world(
        self,
        vis_image,
        points,
        robot_pose,
        color,
        thickness=2,
        dashed=False,
    ):
        grid_size = vis_image.shape[0]
        center_offset = grid_size // 2
        vis_points = self.world_to_vis_points(points, robot_pose, grid_size, center_offset)
        if len(vis_points) < 2:
            return vis_image

        for k in range(len(vis_points) - 1):
            if dashed and (k % 2 == 1):
                continue
            cv2.line(vis_image, tuple(vis_points[k]), tuple(vis_points[k + 1]), color, thickness, cv2.LINE_AA)
        return vis_image

    def draw_points_world(
        self,
        vis_image,
        points,
        robot_pose,
        color,
        radius=4,
    ):
        grid_size = vis_image.shape[0]
        center_offset = grid_size // 2
        vis_points = self.world_to_vis_points(points, robot_pose, grid_size, center_offset)
        for p in vis_points:
            cv2.circle(vis_image, tuple(p), radius + 1, RGB_BLACK, -1, cv2.LINE_AA)
            cv2.circle(vis_image, tuple(p), radius, color, -1, cv2.LINE_AA)
        return vis_image

    def render_esdf_overlay(self, vis_image, robot_pose, esdf, safe_dist=0.30):
        distance = np.asarray(esdf["distance"], dtype=np.float64)
        origin = np.asarray(esdf["origin"], dtype=np.float64)
        esdf_res = float(esdf["resolution"])

        grid_size = vis_image.shape[0]
        center = grid_size // 2
        rows, cols = np.indices((grid_size, grid_size))
        dx = -(rows - center) * self.resolution
        dy = -(cols - center) * self.resolution

        world_x = robot_pose[0] + dx
        world_y = robot_pose[1] + dy

        mx = np.floor((world_x - origin[0]) / esdf_res).astype(np.int32)
        my = np.floor((world_y - origin[1]) / esdf_res).astype(np.int32)

        valid = (mx >= 0) & (mx < distance.shape[1]) & (my >= 0) & (my < distance.shape[0])
        dist = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
        dist[valid] = distance[my[valid], mx[valid]]

        # Render ESDF as a continuous gradient (not discrete TSDF-like buckets).
        # Clamp distances to a fixed range for visualization, then map through
        # a perceptually uniform colormap so the user sees the true field shape.
        unknown = ~np.isfinite(dist)
        known = np.isfinite(dist)

        colored = np.zeros_like(vis_image)
        colored[unknown] = (30, 30, 30)  # dark gray for unknown / out-of-bounds

        if np.any(known):
            vmin, vmax = -0.5, 2.0  # meters: inside obstacle … far free
            d_clamped = np.clip(dist, vmin, vmax)
            gray = np.zeros((grid_size, grid_size), dtype=np.uint8)
            gray[known] = ((d_clamped[known] - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
            colored[unknown] = (30, 30, 30)

        alpha = 0.35
        vis_image[:] = cv2.addWeighted(vis_image, 1.0 - alpha, colored, alpha, 0)
        return vis_image

    def query_esdf_distance(self, esdf, points_xy):
        points_xy = np.asarray(points_xy, dtype=np.float64)
        if points_xy.ndim != 2 or points_xy.shape[1] < 2:
            return np.zeros((0,), dtype=np.float64)

        distance = np.asarray(esdf["distance"], dtype=np.float64)
        origin = np.asarray(esdf["origin"], dtype=np.float64)
        res = float(esdf["resolution"])

        out = []
        for p in points_xy[:, :2]:
            if not np.all(np.isfinite(p)):
                out.append(float("nan"))
                continue
            mx = int(np.floor((p[0] - origin[0]) / res))
            my = int(np.floor((p[1] - origin[1]) / res))
            if 0 <= mx < distance.shape[1] and 0 <= my < distance.shape[0]:
                out.append(float(distance[my, mx]))
            else:
                out.append(float("nan"))
        return np.asarray(out, dtype=np.float64)

    def update_speed_history(self, actual_v, actual_w, cmd_v=None, cmd_w=None, planned_v=None):
        self.speed_history.append({
            "actual_v": float(actual_v) if actual_v is not None else np.nan,
            "actual_w": float(actual_w) if actual_w is not None else np.nan,
            "cmd_v": float(cmd_v) if cmd_v is not None else np.nan,
            "cmd_w": float(cmd_w) if cmd_w is not None else np.nan,
            "planned_v": float(planned_v) if planned_v is not None else np.nan,
        })

    def append_speed_plot(self, image, speed_max=1.0):
        plot_h = 180
        plot_w = image.shape[1]
        plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)

        if len(self.speed_history) < 2:
            return np.concatenate([image, plot], axis=0)

        hist = list(self.speed_history)
        actual = np.array([h["actual_v"] for h in hist], dtype=np.float64)
        cmd = np.array([h["cmd_v"] for h in hist], dtype=np.float64)
        planned = np.array([h["planned_v"] for h in hist], dtype=np.float64)

        all_values = np.concatenate([
            actual[np.isfinite(actual)],
            cmd[np.isfinite(cmd)],
            planned[np.isfinite(planned)],
        ])
        if all_values.size == 0:
            max_v = max(float(speed_max), 0.5)
        else:
            max_v = max(float(speed_max), float(np.max(np.abs(all_values))), 0.5)
        max_v = min(max_v * 1.2, 3.0)

        def series_to_points(series):
            n = len(series)
            xs = np.linspace(40, plot_w - 20, n)
            clipped = np.clip(series, 0.0, max_v)
            ys = plot_h - 30 - clipped / max_v * (plot_h - 55)
            valid = np.isfinite(series)
            if np.count_nonzero(valid) < 2:
                return np.zeros((0, 2), dtype=np.int32)
            return np.stack([xs[valid], ys[valid]], axis=1).astype(np.int32)

        cv2.line(plot, (40, 15), (40, plot_h - 30), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(plot, (40, plot_h - 30), (plot_w - 20, plot_h - 30), (80, 80, 80), 1, cv2.LINE_AA)

        for pts, color in [
            (series_to_points(actual), RGB_WHITE),
            (series_to_points(cmd), RGB_GREEN),
            (series_to_points(planned), RGB_YELLOW),
        ]:
            if len(pts) >= 2:
                cv2.polylines(plot, [pts], False, color, 2, cv2.LINE_AA)

        cv2.putText(plot, "speed history", (50, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(plot, "actual white | cmd green | planned yellow", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        latest = hist[-1]
        text = f"actual={latest['actual_v']:.2f} cmd={latest['cmd_v']:.2f}"
        if np.isfinite(latest["planned_v"]):
            text += f" planned={latest['planned_v']:.2f}"
        cv2.putText(plot, text, (50, plot_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        return np.concatenate([image, plot], axis=0)
        
    def visualize_trajectory(
        self,
        rgb_image,
        depth_image,
        intrinsic,
        trajectory_points,
        robot_pose,
        camera_roll=0,
        all_trajectories_points=None,
        all_trajectories_values=None,
        raw_trajectory_points=None,
        selected_candidate_points=None,
        control_points=None,
        esdf=None,
        minco_info=None,
    ):
        # Calculate visualization size based on 10m×10m range
        grid_size = int(10.0 / self.resolution)  # 20m in grid cells
        vis_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)
         
        # Build current occupancy grid
        occupancy_grid, min_coords = self.build_occupancy_grid(depth_image[..., 0], intrinsic, camera_roll)
        if occupancy_grid is None:
            return combined_image
        
        # Add to history with robot pose
        self.occupancy_history.append((occupancy_grid, min_coords, robot_pose))
        
        # Calculate center offset (assuming robot is at center)
        center_offset = grid_size // 2
        if esdf is not None:
            try:
                overlay_safe_dist = 0.30
                if isinstance(minco_info, dict):
                    overlay_safe_dist = float(minco_info.get("safe_dist", overlay_safe_dist))
                vis_image = self.render_esdf_overlay(vis_image, robot_pose, esdf, safe_dist=overlay_safe_dist)
            except Exception:
                pass
        
        # Draw historical occupancy grids
        all_hist_world_points_list = []
        current_world_points = np.array([])

        # Process historical frames first
        for i, (hist_grid, hist_min_coords, hist_pose) in enumerate(self.occupancy_history):
            # Get occupied points in the grid's local frame
            grid_coords = np.where(hist_grid > 0)
            points = np.array([
                grid_coords[0] * self.resolution + hist_min_coords[0],
                grid_coords[1] * self.resolution + hist_min_coords[1]
            ]).T
            
            # Transform points from the grid's local frame to world frame
            hist_rotation = np.array([
                [np.cos(hist_pose[2]), -np.sin(hist_pose[2])],
                [np.sin(hist_pose[2]), np.cos(hist_pose[2])]
            ])
            world_points = (hist_rotation @ points.T).T + hist_pose[:2]

            if i == len(self.occupancy_history) - 1:  # Current frame
                current_world_points = world_points
            else:  # Historical frame
                if world_points.size > 0:
                    all_hist_world_points_list.append(world_points)

        # Combine all historical points
        if all_hist_world_points_list:
            all_hist_world_points = np.concatenate(all_hist_world_points_list, axis=0)
        else:
            all_hist_world_points = np.array([])

        # Draw historical points (Gray)
        vis_coords_hist = self.world_to_vis_points(all_hist_world_points, robot_pose, grid_size, center_offset)
        if vis_coords_hist.size > 0:
            vis_image[vis_coords_hist[:, 1], vis_coords_hist[:, 0]] = RGB_GRAY

        # Draw current points (Red)
        vis_coords_current = self.world_to_vis_points(current_world_points, robot_pose, grid_size, center_offset)
        if vis_coords_current.size > 0:
            vis_image[vis_coords_current[:, 1], vis_coords_current[:, 0]] = RGB_RED
        
        if raw_trajectory_points is not None:
            self.draw_polyline_world(
                vis_image,
                raw_trajectory_points,
                robot_pose,
                color=RGB_ORANGE,
                thickness=1,
                dashed=True,
            )

        if selected_candidate_points is not None:
            self.draw_polyline_world(
                vis_image,
                selected_candidate_points,
                robot_pose,
                color=RGB_CYAN,
                thickness=1,
                dashed=False,
            )

        vis_points = np.zeros((0, 2), dtype=np.int32)
        if trajectory_points is not None:
            self.draw_polyline_world(
                vis_image,
                trajectory_points,
                robot_pose,
                color=RGB_GREEN,
                thickness=2,
                dashed=False,
            )
            vis_points = self.world_to_vis_points(trajectory_points, robot_pose, grid_size, center_offset)
            if len(vis_points) > 0:
                cv2.circle(vis_image, tuple(vis_points[0]), 3, RGB_BLUE, -1, cv2.LINE_AA)
                cv2.circle(vis_image, tuple(vis_points[-1]), 3, RGB_RED, -1, cv2.LINE_AA)

            if esdf is not None:
                try:
                    dists = self.query_esdf_distance(esdf, trajectory_points)
                    traj_np = np.asarray(trajectory_points, dtype=np.float64)
                    finite_mask = np.isfinite(dists)

                    min_py_esdf = np.nan
                    if np.any(finite_mask):
                        finite_indices = np.flatnonzero(finite_mask)
                        min_local_idx = int(np.argmin(dists[finite_mask]))
                        min_idx = int(finite_indices[min_local_idx])
                        min_py_esdf = float(dists[min_idx])
                        min_point = traj_np[min_idx:min_idx + 1]
                        if min_py_esdf <= 0.0:
                            self.draw_points_world(
                                vis_image,
                                min_point,
                                robot_pose,
                                color=RGB_RED,
                                radius=5,
                            )
                        elif min_py_esdf <= 0.50:
                            self.draw_points_world(
                                vis_image,
                                min_point,
                                robot_pose,
                                color=RGB_ORANGE,
                                radius=4,
                            )

                    cv2.putText(
                        vis_image,
                        f"py min esdf:{min_py_esdf:.2f}",
                        (grid_size - 112, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        RGB_WHITE,
                        1,
                        cv2.LINE_AA,
                    )
                except Exception:
                    pass

        if control_points is not None:
            self.draw_points_world(
                vis_image,
                control_points,
                robot_pose,
                color=RGB_YELLOW,
                radius=4,
            )

        rect_length = 10
        rect_width = 5
        start_point = (center_offset, center_offset)
        yaw = -robot_pose[2]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        corners = np.array([
            [-rect_width / 2, -rect_length / 2],
            [rect_width / 2, -rect_length / 2],
            [rect_width / 2, rect_length / 2],
            [-rect_width / 2, rect_length / 2],
        ])
        rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_corners = (rot_matrix @ corners.T).T + start_point
        corners_int = rotated_corners.astype(np.int32)
        cv2.polylines(vis_image, [corners_int], True, RGB_WHITE, 1, cv2.LINE_AA)

        legend_y = 16
        cv2.putText(vis_image, "green MINCO | orange raw", (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RGB_GREEN, 1, cv2.LINE_AA)
        cv2.putText(vis_image, "cyan selected | yellow waypoint", (8, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RGB_CYAN, 1, cv2.LINE_AA)
        cv2.putText(vis_image, "red obstacle | gray history", (8, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RGB_RED, 1, cv2.LINE_AA)
        cv2.putText(vis_image, "red/orange dot min ESDF", (8, legend_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RGB_ORANGE, 1, cv2.LINE_AA)
        
        # Resize visualization to match RGB image height with better interpolation
        vis_resized = cv2.resize(vis_image, (int(rgb_image.shape[0]), int(rgb_image.shape[0])), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges (adjust sigma as needed)
        vis_resized = cv2.GaussianBlur(vis_resized, (3, 3), 0.5)
        # Concatenate images
        combined_image = np.concatenate((rgb_image, vis_resized), axis=1)

        if (
            not self.show_all_candidates
            or all_trajectories_points is None
            or len(all_trajectories_points) == 0
        ):
            return combined_image
        
        # --- Create additional visualization for all trajectories ---
        # Create a new image for all trajectories visualization
        vis_image_all = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        # Draw the same occupancy grid
        if vis_coords_hist.size > 0:
            vis_image_all[vis_coords_hist[:, 1], vis_coords_hist[:, 0]] = RGB_GRAY
        if vis_coords_current.size > 0:
            vis_image_all[vis_coords_current[:, 1], vis_coords_current[:, 0]] = RGB_RED
            
        has_all_trajectories = all_trajectories_points is not None and len(all_trajectories_points) > 0

        # Draw all trajectories with colors based on values
        # Define color mapping function from value to color (blue to red gradient)
        def value_to_color(value, values_min, values_max):
            # Normalize value to [0, 1] based on a fixed range [-2, 0.5]
            fixed_min = -1.2
            fixed_max = 0.2

            # Clamp the value to be within the fixed range
            value = np.clip(value, fixed_min, fixed_max)

            # Normalize value to [0, 1]
            normalized = (value - fixed_min) / (fixed_max - fixed_min)

            # Map to blue (low) -> green (mid) -> red (high)
            if normalized < 0.5:
                # Blue to green
                b = 255 * (1 - 2 * normalized)
                g = 255 * (2 * normalized)
                r = 0
            else:
                # Green to red
                b = 0
                g = 255 * (2 - 2 * normalized)
                r = 255 * (2 * normalized - 1)
            
            return (int(r), int(g), int(b))
        
        if has_all_trajectories:
            # Set default colors if no values provided
            if all_trajectories_values is None:
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
                trajectory_colors = [colors[idx % len(colors)] for idx in range(len(all_trajectories_points))]
            else:
                # Get min and max values for normalization
                values_min = np.min(all_trajectories_values)
                values_max = np.max(all_trajectories_values)
                # Generate color for each trajectory
                trajectory_colors = [value_to_color(v, values_min, values_max) for v in all_trajectories_values]

            for idx, traj in enumerate(all_trajectories_points):
                color = trajectory_colors[idx]
                vis_points_all = self.world_to_vis_points(traj, robot_pose, grid_size, center_offset)

                # Draw trajectory with anti-aliased lines
                for i in range(len(vis_points_all) - 1):
                    cv2.line(vis_image_all, tuple(vis_points_all[i]), tuple(vis_points_all[i+1]), color, 1, cv2.LINE_AA)

                # Draw start and end points with anti-aliasing
                if len(vis_points_all) > 0:
                    cv2.circle(vis_image_all, tuple(vis_points_all[0]), 2, color, -1, cv2.LINE_AA)
        
        # Draw robot position with anti-aliasing
        cv2.polylines(vis_image_all, [corners_int], True, RGB_WHITE, 1, cv2.LINE_AA)
        
        # Resize all trajectories visualization - align width with rgb_image
        # Get target width (same as rgb_image width)
        target_width = rgb_image.shape[1]
        # Calculate the height to maintain aspect ratio
        target_height = int(vis_image_all.shape[0] * (target_width / vis_image_all.shape[1]))
        # Resize with the calculated dimensions using better interpolation
        vis_resized_all = cv2.resize(vis_image_all, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        # Apply slight Gaussian blur to smooth pixelated edges
        vis_resized_all = cv2.GaussianBlur(vis_resized_all, (3, 3), 0.5)
        
        # Create black padding to match combined_image width
        if combined_image.shape[1] > target_width:
            # Add black padding to the right
            padding_width = combined_image.shape[1] - target_width
            padding = np.zeros((target_height, padding_width, 3), dtype=np.uint8)
            vis_resized_all = np.concatenate((vis_resized_all, padding), axis=1)
        
        # Stack vertically: combined_image (top) and vis_resized_all (bottom)
        final_combined_image = np.concatenate((combined_image, vis_resized_all), axis=0)
        
        return final_combined_image
