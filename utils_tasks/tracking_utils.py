import math
import time

import casadi as ca
import numpy as np

from utils_tasks.differential_drive_utils import DifferentialDriveLimits


SAMPLE_T = 0
SAMPLE_POS_X = 1
SAMPLE_POS_Y = 2
SAMPLE_VEL_X = 4
SAMPLE_VEL_Y = 5
SAMPLE_ACC_X = 7
SAMPLE_ACC_Y = 8
SAMPLE_JERK_X = 10
SAMPLE_JERK_Y = 11
SAMPLE_YAW = 13
SAMPLE_YAW_DOT = 14
MIN_SAMPLE_COLS = 15

YAW_MOVING_SPEED = 1e-3
YAW_CONSISTENCY_MIN_SAMPLES = 3
YAW_SEVERE_ERROR = 0.75 * np.pi
YAW_SEVERE_FRACTION = 0.75


class MPC_Controller:
    def __init__(
        self,
        global_planed_traj,
        N=15,
        desired_v=0.5,
        v_max=0.5,
        w_max=0.5,
        ref_gap=3,
        trajectory_samples=None,
        T=0.1,
        max_acc=4.0,
        max_yaw_acc=4.0,
        allow_geometric_fallback=True,
        wheel_radius=None,
        wheel_base=None,
        max_wheel_speed=None,
    ):
        self.N = int(N)
        self.T = float(T)
        self.desired_v = float(desired_v)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.ref_gap = int(ref_gap)
        self.max_acc = float(max_acc)
        self.max_yaw_acc = float(max_yaw_acc)
        self.allow_geometric_fallback = bool(allow_geometric_fallback)
        self._horizon_time_offsets = np.arange(self.N + 1, dtype=np.float64) * self.T
        self._configure_wheel_constraints(wheel_radius, wheel_base, max_wheel_speed)

        self.q_xy = 20.0
        self.q_yaw = 4.0
        self.q_v = 4.0
        self.q_w = 1.5
        self.r_dv = 3.0
        self.r_dw = 1.0
        self.terminal_xy_scale = 1.5
        self.terminal_yaw_scale = 1.0

        self.reference = None
        self._minco_motion_samples = None
        self.progress_idx = 0
        self.progress_idx_float = 0.0
        self.progress_time = 0.0
        self._needs_global_alignment = True
        self._current_reference = None
        self._last_reference_horizon = None
        self._last_solve_error = None
        self._last_error_print_time = 0.0
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        self.last_command = np.zeros(2, dtype=np.float64)
        self.has_valid_last_command = False

        self._build_problem()
        if not self.update_reference(
            global_planed_traj,
            trajectory_samples=trajectory_samples,
            desired_v=desired_v,
        ):
            raise ValueError("MPC reference requires valid trajectory_samples when geometric fallback is disabled")

    def _build_problem(self):
        opti = ca.Opti()
        opt_controls = opti.variable(self.N, 2)
        opt_states = opti.variable(self.N + 1, 3)
        opt_x0 = opti.parameter(3)
        opt_refs = opti.parameter(self.N + 1, 5)
        opt_u_prev = opti.parameter(2)

        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(self.N):
            x_i = opt_states[i, :]
            u_i = opt_controls[i, :]
            x_next = ca.horzcat(
                x_i[0] + u_i[0] * ca.cos(x_i[2]) * self.T,
                x_i[1] + u_i[0] * ca.sin(x_i[2]) * self.T,
                x_i[2] + u_i[1] * self.T,
            )
            opti.subject_to(opt_states[i + 1, :] == x_next)
            prev_u = opt_u_prev if i == 0 else opt_controls[i - 1, :]
            opti.subject_to(opti.bounded(-self.max_acc * self.T, u_i[0] - prev_u[0], self.max_acc * self.T))
            opti.subject_to(opti.bounded(-self.max_yaw_acc * self.T, u_i[1] - prev_u[1], self.max_yaw_acc * self.T))
            if self.wheel_constraints_enabled:
                wheel_left, wheel_right = self._drive_limits.wheel_speeds(u_i[0], u_i[1])
                opti.subject_to(opti.bounded(-self.max_wheel_speed, wheel_left, self.max_wheel_speed))
                opti.subject_to(opti.bounded(-self.max_wheel_speed, wheel_right, self.max_wheel_speed))

        opti.subject_to(opti.bounded(0.0, opt_controls[:, 0], self.v_max))
        opti.subject_to(opti.bounded(-self.w_max, opt_controls[:, 1], self.w_max))

        obj = 0
        for i in range(self.N):
            yaw_error = self._casadi_yaw_error(opt_states[i, 2], opt_refs[i, 2])
            obj += self.q_xy * (
                (opt_states[i, 0] - opt_refs[i, 0]) ** 2
                + (opt_states[i, 1] - opt_refs[i, 1]) ** 2
            )
            obj += self.q_yaw * yaw_error ** 2
            obj += self.q_v * (opt_controls[i, 0] - opt_refs[i, 3]) ** 2
            obj += self.q_w * (opt_controls[i, 1] - opt_refs[i, 4]) ** 2
            prev_u = opt_u_prev if i == 0 else opt_controls[i - 1, :]
            obj += self.r_dv * (opt_controls[i, 0] - prev_u[0]) ** 2
            obj += self.r_dw * (opt_controls[i, 1] - prev_u[1]) ** 2

        terminal_yaw_error = self._casadi_yaw_error(opt_states[self.N, 2], opt_refs[self.N, 2])
        obj += (
            self.terminal_xy_scale
            * self.q_xy
            * (
                (opt_states[self.N, 0] - opt_refs[self.N, 0]) ** 2
                + (opt_states[self.N, 1] - opt_refs[self.N, 1]) ** 2
            )
            + self.terminal_yaw_scale * self.q_yaw * terminal_yaw_error ** 2
        )
        opti.minimize(obj)

        opts_setting = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }
        opti.solver("ipopt", opts_setting)

        self.opti = opti
        self.opt_x0 = opt_x0
        self.opt_refs = opt_refs
        self.opt_u_prev = opt_u_prev
        self.opt_controls = opt_controls
        self.opt_states = opt_states

    @staticmethod
    def _casadi_yaw_error(yaw, yaw_ref):
        return ca.atan2(ca.sin(yaw - yaw_ref), ca.cos(yaw - yaw_ref))

    def _configure_wheel_constraints(self, wheel_radius, wheel_base, max_wheel_speed):
        self._drive_limits = DifferentialDriveLimits.create(
            wheel_radius, wheel_base, max_wheel_speed
        )
        self.wheel_constraints_enabled = self._drive_limits.enabled
        self.wheel_radius = self._drive_limits.wheel_radius
        self.wheel_base = self._drive_limits.wheel_base
        self.max_wheel_speed = self._drive_limits.max_wheel_speed

    def _project_wheel_feasible_command(self, command):
        return self._drive_limits.project(command, self.v_max, self.w_max)

    def update_reference(self, global_planed_traj, trajectory_samples=None, desired_v=None):
        if desired_v is not None:
            self.desired_v = float(desired_v)
        reference = self._parse_minco_samples(trajectory_samples)
        if reference is None:
            if not self.allow_geometric_fallback:
                return False
            self._minco_motion_samples = None
            reference = self._build_fallback_reference(global_planed_traj)
        self.reference = reference
        self._cache_reference_segments()
        self.progress_idx = 0
        self.progress_idx_float = 0.0
        self.progress_time = float(reference[0, 0])
        self._needs_global_alignment = True
        self._current_reference = None
        self._last_reference_horizon = None
        return True

    def _cache_reference_segments(self):
        points = np.ascontiguousarray(self.reference[:, 1:3], dtype=np.float64)
        self._segment_start = points[:-1]
        self._segment_delta = np.diff(points, axis=0)
        self._segment_len_sq = np.einsum("ij,ij->i", self._segment_delta, self._segment_delta)
        self._segment_dt = np.diff(self.reference[:, 0])
        self._valid_segment = (self._segment_len_sq > 1e-12) & (self._segment_dt > 1e-12)

    def _parse_minco_samples(self, samples):
        if samples is None:
            return None
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < MIN_SAMPLE_COLS:
            return None

        required = samples[:, [
            SAMPLE_T,
            SAMPLE_POS_X,
            SAMPLE_POS_Y,
            SAMPLE_VEL_X,
            SAMPLE_VEL_Y,
            SAMPLE_ACC_X,
            SAMPLE_ACC_Y,
        ]]
        valid_rows = np.all(np.isfinite(required), axis=1)
        if np.count_nonzero(valid_rows) < 2:
            return None
        samples = samples[valid_rows]

        t = samples[:, SAMPLE_T]
        if np.any(np.diff(t) < -1e-9):
            return None
        keep = np.ones(t.shape[0], dtype=bool)
        keep[1:] = np.diff(t) > 1e-9
        samples = samples[keep]
        if samples.shape[0] < 2:
            return None

        t = samples[:, SAMPLE_T] - samples[0, SAMPLE_T]
        x = samples[:, SAMPLE_POS_X]
        y = samples[:, SAMPLE_POS_Y]
        vx = samples[:, SAMPLE_VEL_X]
        vy = samples[:, SAMPLE_VEL_Y]
        ax = samples[:, SAMPLE_ACC_X]
        ay = samples[:, SAMPLE_ACC_Y]
        jx = samples[:, SAMPLE_JERK_X]
        jy = samples[:, SAMPLE_JERK_Y]
        yaw_raw = samples[:, SAMPLE_YAW]
        yaw_dot_raw = samples[:, SAMPLE_YAW_DOT]
        reference = self._derive_unicycle_reference(
            t, x, y, vx, vy, ax, ay, yaw_raw, yaw_dot_raw
        )
        if reference is None or reference.shape[0] < 2 or not np.all(np.isfinite(reference)):
            return None
        self._minco_motion_samples = np.ascontiguousarray(
            np.column_stack((t, x, y, vx, vy, ax, ay, jx, jy, reference[:, 3], reference[:, 5])),
            dtype=np.float64,
        )
        return reference

    def _derive_unicycle_reference(
        self, t, x, y, vx, vy, ax, ay, yaw_raw=None, yaw_dot_raw=None
    ):
        speed = np.hypot(vx, vy)
        if not self.allow_geometric_fallback:
            if yaw_raw is None or yaw_dot_raw is None:
                return None
            yaw_raw = np.asarray(yaw_raw, dtype=np.float64).reshape(-1)
            yaw_dot_raw = np.asarray(yaw_dot_raw, dtype=np.float64).reshape(-1)
            if yaw_raw.size != x.size or yaw_dot_raw.size != x.size:
                return None
            if not np.all(np.isfinite(yaw_raw)) or not np.all(np.isfinite(yaw_dot_raw)):
                return None
            yaw = np.unwrap(yaw_raw)
            v_ref = np.clip(speed, 0.0, self.v_max)
            w_ref = np.clip(yaw_dot_raw, -self.w_max, self.w_max)
            return np.column_stack((t, x, y, yaw, v_ref, w_ref))

        yaw = self._derive_reference_yaw(x, y, vx, vy, yaw_raw)
        v_ref = np.clip(speed, 0.0, self.v_max)

        speed_sq = vx * vx + vy * vy
        curvature_eps = 1e-6
        curvature_speed_eps = 1e-3
        w_curvature = (vx * ay - vy * ax) / np.maximum(speed_sq, curvature_eps)
        w_from_yaw = np.gradient(yaw, t, edge_order=1)
        w_fallback = np.where(speed_sq > curvature_speed_eps ** 2, w_curvature, w_from_yaw)
        if yaw_dot_raw is None:
            w_ref = w_fallback
        else:
            yaw_dot_raw = np.asarray(yaw_dot_raw, dtype=np.float64)
            w_ref = np.where(np.isfinite(yaw_dot_raw), yaw_dot_raw, w_fallback)
        w_ref = np.clip(w_ref, -self.w_max, self.w_max)
        return np.column_stack((t, x, y, yaw, v_ref, w_ref))

    def _derive_reference_yaw(self, x, y, vx, vy, yaw_raw=None):
        speed = np.hypot(vx, vy)
        path_yaw = self._path_tangent_yaw(np.column_stack((x, y)))
        geometric_yaw = np.where(speed > YAW_MOVING_SPEED, np.arctan2(vy, vx), path_yaw)
        if yaw_raw is not None:
            raw = np.asarray(yaw_raw, dtype=np.float64)
            if np.all(np.isfinite(raw)):
                raw = np.unwrap(raw)
                moving = speed > YAW_MOVING_SPEED
                errors = np.abs(self._wrap_to_pi(raw[moving] - np.arctan2(vy[moving], vx[moving])))
                consistent = (
                    errors.size < YAW_CONSISTENCY_MIN_SAMPLES
                    or np.mean(errors > YAW_SEVERE_ERROR) < YAW_SEVERE_FRACTION
                )
                if consistent:
                    return raw

        yaw = np.asarray(geometric_yaw, dtype=np.float64).copy()
        if not np.all(np.isfinite(yaw)):
            finite = np.flatnonzero(np.isfinite(yaw))
            if finite.size == 0:
                return None
            first = finite[0]
            yaw[:first] = yaw[first]
            for i in range(first + 1, yaw.shape[0]):
                if not np.isfinite(yaw[i]):
                    yaw[i] = yaw[i - 1]
        return np.unwrap(yaw)

    def _build_fallback_reference(self, path):
        path = self._clean_path(path)
        if path.shape[0] < 2:
            x, y = (path[0] if path.shape[0] == 1 else np.zeros(2, dtype=np.float64))
            return np.array(
                [
                    [0.0, x, y, 0.0, 0.0, 0.0],
                    [self.T, x, y, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )

        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total = arc[-1]
        remaining = np.maximum(0.0, total - arc)
        desired = max(0.0, min(float(self.desired_v), self.v_max))
        v_stop_limit = np.sqrt(np.maximum(0.0, 2.0 * max(self.max_acc, 1e-6) * remaining))
        v_ref = np.minimum(desired, v_stop_limit)
        v_ref[-1] = 0.0

        t = np.zeros(path.shape[0], dtype=np.float64)
        for i in range(1, path.shape[0]):
            avg_v = max(0.5 * (v_ref[i - 1] + v_ref[i]), 0.05)
            t[i] = t[i - 1] + segment_lengths[i - 1] / avg_v

        yaw = np.unwrap(self._path_tangent_yaw(path))
        if not np.all(np.isfinite(yaw)):
            yaw = np.nan_to_num(yaw, nan=0.0)
        w_ref = np.gradient(yaw, t, edge_order=1) if path.shape[0] > 2 else np.zeros_like(yaw)
        w_ref = np.clip(w_ref, -self.w_max, self.w_max)
        return np.column_stack((t, path[:, 0], path[:, 1], yaw, v_ref, w_ref))

    @staticmethod
    def _clean_path(path):
        if path is None:
            return np.zeros((0, 2), dtype=np.float64)
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float64)
        path = path[:, :2]
        path = path[np.all(np.isfinite(path), axis=1)]
        if path.shape[0] <= 1:
            return path
        keep = np.ones(path.shape[0], dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6
        return path[keep]

    @staticmethod
    def _path_tangent_yaw(path):
        path = np.asarray(path, dtype=np.float64)
        if path.shape[0] < 2:
            return np.zeros(path.shape[0], dtype=np.float64)
        deltas = np.diff(path[:, :2], axis=0)
        seg_yaw = np.arctan2(deltas[:, 1], deltas[:, 0])
        yaw = np.empty(path.shape[0], dtype=np.float64)
        yaw[:-1] = seg_yaw
        yaw[-1] = seg_yaw[-1]
        return yaw

    def _find_progress_index(self, current_state):
        segment_count = self._segment_delta.shape[0]
        if segment_count == 0 or not np.any(self._valid_segment):
            self.progress_idx = 0
            self.progress_idx_float = 0.0
            self.progress_time = float(self.reference[0, 0])
            self._needs_global_alignment = False
            return self.progress_idx
        if self._needs_global_alignment:
            start = 0
            end = segment_count
        else:
            start = max(0, self.progress_idx - 3)
            forward_window = max(30, 3 * (self.N + 1))
            end = min(segment_count, self.progress_idx + forward_window)
        indices = np.arange(start, end, dtype=np.int64)
        indices = indices[self._valid_segment[indices]]
        if indices.size == 0:
            indices = np.flatnonzero(self._valid_segment)

        starts = self._segment_start[indices]
        deltas = self._segment_delta[indices]
        offsets = np.asarray(current_state[:2], dtype=np.float64) - starts
        alpha = np.einsum("ij,ij->i", offsets, deltas) / self._segment_len_sq[indices]
        alpha = np.clip(alpha, 0.0, 1.0)
        projected = starts + alpha[:, None] * deltas
        distance_sq = np.einsum(
            "ij,ij->i", projected - current_state[:2], projected - current_state[:2]
        )
        min_distance_sq = float(np.min(distance_sq))
        close = np.flatnonzero(distance_sq <= min_distance_sq + 1e-8)
        if close.size > 1:
            segment_yaw = np.arctan2(deltas[close, 1], deltas[close, 0])
            heading_error = np.abs(self._wrap_to_pi(segment_yaw - current_state[2]))
            local_idx = int(close[np.argmin(heading_error)])
        else:
            local_idx = int(close[0])

        candidate_idx = int(indices[local_idx])
        candidate_alpha = float(alpha[local_idx])
        candidate_time = float(
            self.reference[candidate_idx, 0] + candidate_alpha * self._segment_dt[candidate_idx]
        )
        if not self._needs_global_alignment and candidate_time + 1e-4 < self.progress_time:
            return self.progress_idx
        self.progress_idx = candidate_idx
        self.progress_idx_float = candidate_idx + candidate_alpha
        self.progress_time = candidate_time
        self._needs_global_alignment = False
        return self.progress_idx

    def _build_reference_horizon(self, current_state):
        if self.reference is None or self.reference.shape[0] == 0:
            self.reference = self._build_fallback_reference(None)
        self._find_progress_index(current_state)
        t_ref = self.progress_time + self._horizon_time_offsets
        src_t = self.reference[:, 0]
        right = np.searchsorted(src_t, t_ref, side="right")
        right = np.clip(right, 1, src_t.size - 1)
        left = right - 1
        interval = src_t[right] - src_t[left]
        alpha = np.clip((t_ref - src_t[left]) / interval, 0.0, 1.0)
        values_left = self.reference[left, 1:6]
        values_right = self.reference[right, 1:6]
        horizon = values_left + alpha[:, None] * (values_right - values_left)
        beyond_tail = t_ref > src_t[-1]
        horizon[beyond_tail, :3] = self.reference[-1, 1:4]
        horizon[beyond_tail, 3] = 0.0
        horizon[beyond_tail, 4] = 0.0
        horizon[:, 2] = np.unwrap(horizon[:, 2])
        yaw_offset = 2.0 * np.pi * np.round(
            (current_state[2] - horizon[0, 2]) / (2.0 * np.pi)
        )
        horizon[:, 2] += yaw_offset
        self._current_reference = horizon[0].copy()
        self._last_reference_horizon = horizon.copy()
        return horizon

    def solve(self, x00):
        state, actual_v, actual_w = self._coerce_state(x00)
        horizon = self._build_reference_horizon(state)
        u_prev = self._previous_command_for_solve(actual_v, actual_w)
        self.opti.set_value(self.opt_x0, state)
        self.opti.set_value(self.opt_refs, horizon)
        self.opti.set_value(self.opt_u_prev, u_prev)
        u_guess = self._initial_u_guess(u_prev, horizon)
        self.opti.set_initial(self.opt_controls, u_guess)
        self.opti.set_initial(self.opt_states, self._initial_x_guess(state, u_guess))
        try:
            sol = self.opti.solve()
            u_sol = np.asarray(sol.value(self.opt_controls), dtype=np.float64)
            x_sol = np.asarray(sol.value(self.opt_states), dtype=np.float64)
            if not np.all(np.isfinite(u_sol)) or not np.all(np.isfinite(x_sol)):
                raise RuntimeError("MPC solve returned non-finite values")
        except Exception as exc:
            return self._safe_deceleration_solution(state, u_prev, exc)

        self.last_command = self._clip_command(u_sol[0])
        self.has_valid_last_command = True
        self.last_opt_u_controls = np.vstack([u_sol[1:], u_sol[-1:]])
        self.last_opt_x_states = np.vstack([x_sol[1:], x_sol[-1:]])
        self._last_solve_error = None
        return u_sol, x_sol

    def _coerce_state(self, x00):
        state_in = np.asarray(x00, dtype=np.float64).reshape(-1)
        state = np.zeros(3, dtype=np.float64)
        count = min(state_in.size, 3)
        state[:count] = state_in[:count]
        actual_v = self.last_command[0]
        actual_w = self.last_command[1]
        if state_in.size >= 4:
            actual_v = state_in[3]
        if state_in.size >= 5:
            actual_w = state_in[4]
        if state_in.size < 4:
            actual_v = self.last_command[0]
        elif state_in.size < 5:
            actual_w = self.last_command[1]
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        actual_v = float(np.nan_to_num(actual_v, nan=0.0, posinf=0.0, neginf=0.0))
        actual_w = float(np.nan_to_num(actual_w, nan=0.0, posinf=0.0, neginf=0.0))
        actual_v = max(0.0, actual_v)
        return state, actual_v, actual_w

    def _previous_command_for_solve(self, actual_v, actual_w):
        if self.has_valid_last_command:
            return self._clip_command(self.last_command)
        return np.array(
            [
                np.clip(actual_v, 0.0, self.v_max),
                np.clip(actual_w, -self.w_max, self.w_max),
            ],
            dtype=np.float64,
        )

    def _initial_u_guess(self, u_prev, horizon):
        if self.last_opt_u_controls is not None and self.last_opt_u_controls.shape == (self.N, 2):
            return self.last_opt_u_controls
        guess = np.zeros((self.N, 2), dtype=np.float64)
        current_v = u_prev[0]
        current_w = u_prev[1]
        for i in range(self.N):
            target_v = np.clip(horizon[min(i + 1, self.N), 3], 0.0, self.v_max)
            target_w = np.clip(horizon[min(i + 1, self.N), 4], -self.w_max, self.w_max)
            current_v = self._move_toward(current_v, target_v, self.max_acc * self.T)
            current_w = self._move_toward(current_w, target_w, self.max_yaw_acc * self.T)
            guess[i] = [current_v, current_w]
        return guess

    def _initial_x_guess(self, state, controls):
        if self.last_opt_x_states is not None and self.last_opt_x_states.shape == (self.N + 1, 3):
            return self.last_opt_x_states
        return self._rollout_controls(state, controls)

    def _safe_deceleration_solution(self, state, u_prev, exc):
        now = time.time()
        error_text = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        if error_text != self._last_solve_error or now - self._last_error_print_time > 2.0:
            print(f"[MPC] solve failed, using safe deceleration: {error_text}")
            self._last_solve_error = error_text
            self._last_error_print_time = now

        controls = np.zeros((self.N, 2), dtype=np.float64)
        current_v = float(np.clip(u_prev[0], 0.0, self.v_max))
        current_w = float(np.clip(u_prev[1], -self.w_max, self.w_max))
        for i in range(self.N):
            current_v = max(0.0, current_v - self.max_acc * self.T)
            current_w = self._move_toward(current_w, 0.0, self.max_yaw_acc * self.T)
            current_v = float(np.clip(current_v, 0.0, self.v_max))
            current_w = float(np.clip(current_w, -self.w_max, self.w_max))
            controls[i] = self._project_wheel_feasible_command([current_v, current_w])
            current_v, current_w = controls[i]
        states = self._rollout_controls(state, controls)

        self.last_command = controls[0].copy()
        self.has_valid_last_command = True
        self.last_opt_u_controls = controls.copy()
        self.last_opt_x_states = states.copy()
        return controls, states

    def _rollout_controls(self, state, controls):
        states = np.zeros((self.N + 1, 3), dtype=np.float64)
        states[0] = state
        for i in range(self.N):
            v, w = controls[i]
            states[i + 1, 0] = states[i, 0] + v * math.cos(states[i, 2]) * self.T
            states[i + 1, 1] = states[i, 1] + v * math.sin(states[i, 2]) * self.T
            states[i + 1, 2] = states[i, 2] + w * self.T
        return states

    def _clip_command(self, command):
        return self._project_wheel_feasible_command(command)

    @staticmethod
    def _move_toward(value, target, step):
        if value < target:
            return min(value + step, target)
        return max(value - step, target)

    def get_current_reference(self):
        if self._current_reference is None:
            return None
        return self._current_reference.copy()

    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        self.last_command = np.zeros(2, dtype=np.float64)
        self.has_valid_last_command = False
        self.progress_idx = 0
        self.progress_idx_float = 0.0
        self.progress_time = float(self.reference[0, 0]) if self.reference is not None else 0.0
        self._needs_global_alignment = True
        self._current_reference = None
        self._last_reference_horizon = None
        self._last_solve_error = None

    @staticmethod
    def _wrap_to_pi(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
