from __future__ import annotations

import numpy as np


RAW_MPC_DEFAULTS = {"N":15, "desired_v":.5, "v_max":.5, "w_max":.5, "ref_gap":3, "T":.1, "dense_ratio":50}
RAW_MPC_WEIGHTS = {"Q":[10.0, 10.0, 0.0], "R":[0.02, 0.15]}


class RawReferenceGenerator:
    """Original reference selection without constructing or solving CasADi MPC."""
    def __init__(self, path_xy, desired_v=.5, ref_gap=3, horizon=15, dt=.1, dense_ratio=50):
        path = np.asarray(path_xy, float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2: raise ValueError("RAW path must have at least two xy points")
        x = np.arange(len(path)); dense_x = np.linspace(0, len(path)-1, len(path)*dense_ratio)
        self.path = np.column_stack([np.interp(dense_x, x, path[:,0]), np.interp(dense_x, x, path[:,1])])
        self.desired_v, self.ref_gap, self.horizon, self.dt = desired_v, ref_gap, horizon, dt
        self.reference_length = horizon // ref_gap + 1

    def find_reference(self, state):
        nearest = int(np.argmin(np.linalg.norm(self.path - np.asarray(state[:2]).reshape(1,2), axis=1)))
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        spacing = self.desired_v * self.ref_gap * self.dt; points = []
        for index in range(nearest, len(self.path)-1):
            if cumulative[index] - cumulative[nearest] >= spacing * len(points):
                points.append(self.path[index])
                if len(points) == self.reference_length: break
        while len(points) < self.reference_length: points.append(self.path[-1])
        return np.asarray(points)
