import math
import os
import sys
import time
from dataclasses import dataclass
from queue import Queue
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


SAMPLE_T = 0
SAMPLE_POS_X = 1
SAMPLE_POS_Y = 2
SAMPLE_VEL_X = 4
SAMPLE_VEL_Y = 5
SAMPLE_ACC_X = 7
SAMPLE_ACC_Y = 8
SAMPLE_YAW = 13
SAMPLE_YAW_DOT = 14
MIN_SAMPLE_COLS = 15


@dataclass
class PlanningInput:
    current_goal: Optional[np.ndarray] = None
    current_image: Optional[np.ndarray] = None
    current_depth: Optional[np.ndarray] = None
    camera_pos: Optional[np.ndarray] = None
    camera_rot: Optional[np.ndarray] = None
    robot_lin_vel_w: Optional[np.ndarray] = None
    robot_ang_vel_w: Optional[np.ndarray] = None


@dataclass
class PlanningOutput:
    trajectory_points_world: Optional[np.ndarray] = None
    all_trajectories_world: Optional[List[np.ndarray]] = None
    all_values_camera: Optional[np.ndarray] = None
    plan_id: int = 0
    is_planning: bool = False
    planning_error: Optional[str] = None


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

        self.q_xy = 20.0
        self.q_yaw = 4.0
        self.q_v = 4.0
        self.q_w = 1.5
        self.r_dv = 3.0
        self.r_dw = 1.0
        self.terminal_xy_scale = 1.5
        self.terminal_yaw_scale = 1.0

        self.reference = None
        self.progress_idx = 0
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

    def update_reference(self, global_planed_traj, trajectory_samples=None, desired_v=None):
        if desired_v is not None:
            self.desired_v = float(desired_v)
        reference = self._parse_minco_samples(trajectory_samples)
        if reference is None:
            if not self.allow_geometric_fallback:
                return False
            reference = self._build_fallback_reference(global_planed_traj)
        self.reference = reference
        self.progress_idx = 0
        self._needs_global_alignment = True
        self._current_reference = None
        self._last_reference_horizon = None
        return True

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
        yaw_raw = samples[:, SAMPLE_YAW] if samples.shape[1] > SAMPLE_YAW else None
        reference = self._derive_unicycle_reference(t, x, y, vx, vy, ax, ay, yaw_raw)
        if reference.shape[0] < 2 or not np.all(np.isfinite(reference)):
            return None
        return reference

    def _derive_unicycle_reference(self, t, x, y, vx, vy, ax, ay, yaw_raw=None):
        speed = np.hypot(vx, vy)
        yaw = self._derive_reference_yaw(x, y, vx, vy, yaw_raw)
        v_ref = np.clip(speed, 0.0, self.v_max)

        speed_sq = vx * vx + vy * vy
        curvature_eps = 1e-6
        curvature_speed_eps = 1e-3
        w_curvature = (vx * ay - vy * ax) / np.maximum(speed_sq, curvature_eps)
        w_from_yaw = np.gradient(yaw, t, edge_order=1)
        w_ref = np.where(speed_sq > curvature_speed_eps ** 2, w_curvature, w_from_yaw)
        w_ref = np.clip(w_ref, -self.w_max, self.w_max)
        return np.column_stack((t, x, y, yaw, v_ref, w_ref))

    def _derive_reference_yaw(self, x, y, vx, vy, yaw_raw=None):
        speed = np.hypot(vx, vy)
        yaw = np.full(speed.shape, np.nan, dtype=np.float64)
        moving = speed > 1e-3
        yaw[moving] = np.arctan2(vy[moving], vx[moving])

        path_yaw = self._path_tangent_yaw(np.column_stack((x, y)))
        raw = None
        if yaw_raw is not None:
            raw = np.asarray(yaw_raw, dtype=np.float64)

        for i in range(yaw.size):
            if np.isfinite(yaw[i]):
                continue
            if np.isfinite(path_yaw[i]):
                yaw[i] = path_yaw[i]
            elif i > 0 and np.isfinite(yaw[i - 1]):
                yaw[i] = yaw[i - 1]
            elif raw is not None and i < raw.size and np.isfinite(raw[i]):
                yaw[i] = raw[i]
            else:
                future = np.flatnonzero(np.isfinite(yaw[i + 1 :]))
                if future.size > 0:
                    yaw[i] = yaw[i + 1 + future[0]]
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
        ref_xy = self.reference[:, 1:3]
        if self._needs_global_alignment:
            start = 0
            end = self.reference.shape[0]
        else:
            start = max(0, self.progress_idx - 3)
            forward_window = max(30, 3 * (self.N + 1))
            end = min(self.reference.shape[0], self.progress_idx + forward_window)
        indices = np.arange(start, end, dtype=np.int64)
        position_distance = np.linalg.norm(ref_xy[indices] - current_state[:2].reshape(1, 2), axis=1)
        heading_error = np.abs(self._wrap_to_pi(self.reference[indices, 3] - current_state[2]))
        heading_weight = 0.05
        local_idx = int(np.argmin(position_distance + heading_weight * heading_error))
        self.progress_idx = int(indices[local_idx])
        self._needs_global_alignment = False
        return self.progress_idx

    def _build_reference_horizon(self, current_state):
        if self.reference is None or self.reference.shape[0] == 0:
            self.reference = self._build_fallback_reference(None)
        idx = self._find_progress_index(current_state)
        t0 = self.reference[idx, 0]
        t_ref = t0 + np.arange(self.N + 1, dtype=np.float64) * self.T
        src_t = self.reference[:, 0]
        horizon = np.empty((self.N + 1, 5), dtype=np.float64)
        for out_col, ref_col in enumerate([1, 2, 3, 4, 5]):
            horizon[:, out_col] = np.interp(
                t_ref,
                src_t,
                self.reference[:, ref_col],
                left=self.reference[0, ref_col],
                right=self.reference[-1, ref_col],
            )
        beyond_tail = t_ref > src_t[-1]
        horizon[beyond_tail, 3] = 0.0
        horizon[beyond_tail, 4] = 0.0
        horizon[:, 2] = np.unwrap(horizon[:, 2])
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
            controls[i] = [current_v, current_w]
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
        command = np.asarray(command, dtype=np.float64).reshape(2)
        return np.array(
            [
                np.clip(command[0], 0.0, self.v_max),
                np.clip(command[1], -self.w_max, self.w_max),
            ],
            dtype=np.float64,
        )

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
        self._needs_global_alignment = True
        self._current_reference = None
        self._last_reference_horizon = None
        self._last_solve_error = None

    @staticmethod
    def _wrap_to_pi(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
