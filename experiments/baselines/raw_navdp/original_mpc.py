"""Minimal adapter of navdp_raw/utils_tasks/tracking_utils.py::MPC_Controller.

The optimization model, weights, bounds, interpolation ratio and IPOPT options
are intentionally kept identical. CasADi is imported only on construction so
static experiment tooling remains lightweight.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


ORIGINAL_MPC_SPEC = {
    "N": 15, "desired_v": 0.5, "v_max": 0.5, "w_max": 0.5,
    "ref_gap": 3, "T": 0.1, "dense_ratio": 50,
    "Q": (10.0, 10.0, 0.0), "R": (0.02, 0.15),
    "ipopt.max_iter": 100, "ipopt.acceptable_tol": 1e-8,
    "ipopt.acceptable_obj_change_tol": 1e-6,
}


class RawNavDPMPCController:
    def __init__(self, global_planed_traj, N=15, desired_v=0.5, v_max=0.5, w_max=0.5, ref_gap=3, T=0.1):
        import casadi as ca

        self.N, self.desired_v, self.ref_gap, self.T = int(N), float(desired_v), int(ref_gap), float(T)
        self.v_max, self.w_max = float(v_max), float(w_max)
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = self.N // self.ref_gap + 1
        self.progress_idx = 0
        self._current_reference = None

        opti = ca.Opti()
        opt_controls = opti.variable(self.N, 2)
        v, w = opt_controls[:, 0], opt_controls[:, 1]
        opt_states = opti.variable(self.N + 1, 3)
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len)
        dynamics = lambda state, control: ca.vertcat(
            *[control[0] * ca.cos(state[2]), control[0] * ca.sin(state[2]), control[1]]
        )
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(self.N):
            next_state = opt_states[i, :] + dynamics(opt_states[i, :], opt_controls[i, :]).T * self.T
            opti.subject_to(opt_states[i + 1, :] == next_state)
        Q = np.diag(ORIGINAL_MPC_SPEC["Q"])
        R = np.diag(ORIGINAL_MPC_SPEC["R"])
        objective = 0
        for i in range(self.N):
            objective += ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
            if i % self.ref_gap == 0:
                ref_index = i // self.ref_gap
                error = opt_states[i, :] - opt_xs[ref_index * 3:ref_index * 3 + 3].T
                objective += ca.mtimes([error, Q, error.T])
        opti.minimize(objective)
        opti.subject_to(opti.bounded(0.0, v, self.v_max))
        opti.subject_to(opti.bounded(-self.w_max, w, self.w_max))
        opti.solver("ipopt", {
            "ipopt.max_iter": 100, "ipopt.print_level": 0, "print_time": 0,
            "ipopt.acceptable_tol": 1e-8, "ipopt.acceptable_obj_change_tol": 1e-6,
        })
        self.opti, self.opt_xs, self.opt_x0 = opti, opt_xs, opt_x0
        self.opt_controls, self.opt_states = opt_controls, opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    @staticmethod
    def make_ref_denser(ref_traj, ratio=50):
        ref_traj = np.asarray(ref_traj, dtype=np.float64)
        if ref_traj.ndim != 2 or ref_traj.shape[0] < 2 or ref_traj.shape[1] < 2:
            raise ValueError("RAW MPC requires at least two finite XY points")
        ref_traj = ref_traj[:, :2]
        if not np.all(np.isfinite(ref_traj)):
            raise ValueError("RAW MPC path contains non-finite coordinates")
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)
        uniform_x = interp1d(x_orig, ref_traj[:, 0], kind="linear")(new_x)
        uniform_y = interp1d(x_orig, ref_traj[:, 1], kind="linear")(new_x)
        return np.stack((uniform_x, uniform_y), axis=1)

    def update_reference(self, path, desired_v=None):
        self.ref_traj = self.make_ref_denser(path)
        if desired_v is not None:
            self.desired_v = float(desired_v)
        self.progress_idx = 0
        self.reset()
        return True

    def find_reference_traj(self, x0, global_planed_traj):
        reference_points = []
        nearest_idx = int(np.argmin(np.linalg.norm(global_planed_traj - np.asarray(x0)[:2].reshape((1, 2)), axis=1)))
        self.progress_idx = nearest_idx
        desired_arc_length = self.desired_v * self.ref_gap * self.T
        cumulative_distance = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if cumulative_distance[i] - cumulative_distance[nearest_idx] >= desired_arc_length * len(reference_points):
                reference_points.append(global_planed_traj[i, :])
                if len(reference_points) == self.ref_traj_len:
                    break
        while len(reference_points) < self.ref_traj_len:
            reference_points.append(global_planed_traj[-1, :])
        return np.asarray(reference_points)

    def solve(self, x00):
        state = np.asarray(x00, dtype=np.float64)[:3]
        reference = self.find_reference_traj(state, self.ref_traj)
        self._current_reference = np.array([reference[0, 0], reference[0, 1], 0.0, self.desired_v, np.nan])
        flat_reference = np.concatenate((reference, np.zeros((reference.shape[0], 1))), axis=1).reshape(-1, 1)
        self.opti.set_value(self.opt_xs, flat_reference)
        self.opti.set_value(self.opt_x0, state)
        controls = np.zeros((self.N, 2)) if self.last_opt_u_controls is None else self.last_opt_u_controls
        states = np.zeros((self.N + 1, 3)) if self.last_opt_x_states is None else self.last_opt_x_states
        self.opti.set_initial(self.opt_controls, controls)
        self.opti.set_initial(self.opt_states, states)
        solution = self.opti.solve()
        self.last_opt_u_controls = solution.value(self.opt_controls)
        self.last_opt_x_states = solution.value(self.opt_states)
        return self.last_opt_u_controls, self.last_opt_x_states

    def get_current_reference(self):
        return None if self._current_reference is None else self._current_reference.copy()

    def reset(self):
        self.last_opt_x_states = None
        self.last_opt_u_controls = None
        self._current_reference = None
