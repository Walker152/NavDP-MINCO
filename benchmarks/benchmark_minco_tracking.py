import statistics
import sys
import time
import types

import numpy as np

try:
    import casadi  # noqa: F401
except ModuleNotFoundError:
    sys.modules["casadi"] = types.ModuleType("casadi")

from utils_tasks.navdp_minco_adapter import NavDPMincoAdapter
from utils_tasks.esdf_query_utils import EsdfGridView
from utils_tasks.tracking_utils import MPC_Controller


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(name, values):
    print(
        f"{name}: mean={statistics.fmean(values):.6f} ms "
        f"p95={percentile(values, 95):.6f} ms max={max(values):.6f} ms n={len(values)}"
    )


def make_controller(sample_count=401):
    t = np.linspace(0.0, 20.0, sample_count)
    samples = np.zeros((sample_count, 15), dtype=np.float64)
    samples[:, 0] = t
    samples[:, 1] = t
    samples[:, 4] = 1.0
    controller = MPC_Controller.__new__(MPC_Controller)
    controller.N = 15
    controller.T = 0.1
    controller._horizon_time_offsets = np.arange(16) * 0.1
    controller.desired_v = 1.0
    controller.v_max = 1.0
    controller.w_max = 1.0
    controller.ref_gap = 3
    controller.max_acc = 1.0
    controller.max_yaw_acc = 1.0
    controller.allow_geometric_fallback = False
    controller.reference = None
    controller._minco_motion_samples = None
    controller.progress_idx = 0
    controller.progress_idx_float = 0.0
    controller.progress_time = 0.0
    controller._needs_global_alignment = True
    controller._current_reference = None
    controller._last_reference_horizon = None
    controller.update_reference(samples[:, 1:3], samples)
    return controller


def benchmark_reference(iterations=2000):
    controller = make_controller()
    values = []
    for index in range(iterations + 100):
        x = min(19.0, index * 0.009)
        start = time.perf_counter()
        controller._build_reference_horizon(np.array([x, 0.05, 0.0]))
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= 100:
            values.append(elapsed)
    summarize("mpc_reference_build", values)


def benchmark_esdf(iterations=1000, point_count=1000):
    adapter = NavDPMincoAdapter.__new__(NavDPMincoAdapter)
    adapter._esdf_grid = EsdfGridView(
        np.random.default_rng(1).random((512, 512)), np.array([-10.0, -10.0]), 0.05
    )
    points = np.random.default_rng(2).uniform(-8.0, 8.0, size=(point_count, 2))
    vectorized_values = []
    legacy_values = []
    for index in range(iterations + 20):
        start = time.perf_counter()
        adapter._query_min_esdf(points)
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= 20:
            vectorized_values.append(elapsed)

        start = time.perf_counter()
        values = []
        for point in points:
            mx, my = np.floor(
                (point - adapter._esdf_grid.origin) / adapter._esdf_grid.resolution
            ).astype(np.int64)
            if 0 <= mx < 512 and 0 <= my < 512:
                values.append(adapter._esdf_grid.distance[my, mx])
        min(values)
        elapsed = (time.perf_counter() - start) * 1000.0
        if index >= 20:
            legacy_values.append(elapsed)
    summarize("adapter_esdf_query_vectorized", vectorized_values)
    summarize("adapter_esdf_query_legacy", legacy_values)
    print(
        "adapter_esdf_speedup: "
        f"{statistics.fmean(legacy_values) / statistics.fmean(vectorized_values):.2f}x"
    )


if __name__ == "__main__":
    benchmark_reference()
    benchmark_esdf()
