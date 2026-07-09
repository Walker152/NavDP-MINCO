# Codex 任务提示词：增强 NavDP + MINCO 可视化与全链路耗时统计

## 0. 任务目标

当前工程已经实现了 NavDP → MINCO → MPC 的基本闭环：

```text
NavDP server 输出 trajectory_points_camera / all_trajectories_camera / all_values_camera
    ↓
eval 中将 camera/local frame 轨迹转换到 Isaac world frame
    ↓
NavDPMincoAdapter 调用 minco_processor.optimize_candidates()
    ↓
成功：使用 MINCO optimized waypoints
失败：fallback 到 raw NavDP top-1 world trajectory
    ↓
MPC_Controller 跟踪二维 world trajectory
    ↓
写入视频
```

现在需要在现有代码基础上做两个增强：

1. **视频可视化增强**：
   - 展示优化后的轨迹；
   - 展示 MINCO 控制点 / sparse waypoints；
   - 展示 raw NavDP top-1 与 selected candidate；
   - BEV 图中叠加 ESDF；
   - 视频底部实时显示速度折线图。

2. **耗时统计增强**：
   - 统计首次 ESDF 生成耗时；
   - 统计 ESDF 缓存加载耗时；
   - 统计 ESDF 构建内部各阶段耗时；
   - 统计 NavDP 请求耗时；
   - 统计 trajectory camera→world 转换耗时；
   - 统计 MINCO 从“传入 world candidate trajectory”到“传出 optimized trajectory”的端到端耗时；
   - 统计 C++ `MincoPipeline::optimize()` 内部耗时；
   - 统计 MPC 构造耗时与 MPC solve 耗时；
   - 统计可视化绘制耗时与视频写入耗时。

本任务优先只改 `eval_pointgoal_wheeled.py`，pointgoal 跑通后再同步到其他 eval。

---

## 1. 必须遵守的改造原则

1. 不修改 NavDP 网络结构。
2. 不修改 `policy_network.py`。
3. 不重构 `MPC_Controller`。
4. 不迁移 ROGMap。
5. 不使用 depth 在线建图。
6. 不改变现有 NavDP → MINCO → MPC 主链路。
7. 不改变 `--enable_minco` 关闭时的原始行为。
8. 不在 `planning_thread` 中做视频绘制。
9. 不在 `visualization_utils.py` 中调用 MINCO。
10. 新增可视化字段全部要可选，字段缺失时不能导致仿真中断。
11. 耗时统计默认打印简洁摘要，不要每帧刷屏过多。
12. 不做 git commit。

---

## 2. 需要修改的文件

重点修改：

```text
minco_processor/bindings/minco_pybind.cpp
utils_tasks/navdp_minco_adapter.py
utils_tasks/basic_utils.py
utils_tasks/visualization_utils.py
utils_tasks/sim_esdf_builder.py
eval_pointgoal_wheeled.py
```

暂时不要同步其他 eval 文件，pointgoal 验证成功后再复制逻辑。

---

## 3. 第一部分：C++ pybind 返回 control points 与 C++ optimize 耗时

### 3.1 当前情况

`minco_pybind.cpp` 当前已经返回：

```text
success
failure_reason
objective
optimizer_return_code
duration
min_esdf
samples
waypoints
```

其中 `samples` 是 15 列：

```text
0:  t
1:  px
2:  py
3:  pz
4:  vx
5:  vy
6:  vz
7:  ax
8:  ay
9:  az
10: jx
11: jy
12: jz
13: yaw
14: yaw_dot
```

当前 `duration` 是 Python 进入 `MincoProcessorPy::optimize()` 到 `pipeline_.optimize(request)` 返回后的 C++ 侧总耗时。

### 3.2 新增 sparse_waypoints / control_points 返回

在 `MincoProcessorPy::toDict()` 中新增：

```cpp
Eigen::MatrixXd sparse_waypoints(
  static_cast<int>(result.sparse_waypoints.size()), 3);

for (int i = 0; i < sparse_waypoints.rows(); ++i) {
  sparse_waypoints.row(i) =
    result.sparse_waypoints[static_cast<size_t>(i)].transpose();
}

out["sparse_waypoints"] = sparse_waypoints;
out["control_points"] = sparse_waypoints;
```

第一版将 `result.sparse_waypoints` 作为视频中的 MINCO 控制点 / 优化锚点即可，不需要导出多项式内部真正控制点。

### 3.3 增加 C++ 字段命名

保留原有：

```cpp
out["duration"] = duration;
```

同时新增一个更明确的字段名，便于 Python 侧统计：

```cpp
out["cpp_optimize_time_ms"] = duration * 1000.0;
```

### 3.4 检查 include

`minco_pybind.cpp` 中如果还没有这些 include，请补充：

```cpp
#include <algorithm>
#include <cmath>
#include <limits>
```

因为当前使用了：

```cpp
std::min
std::isfinite
std::numeric_limits
```

---

## 4. 第二部分：NavDPMincoAdapter 返回可视化字段与 MINCO 端到端耗时

修改 `utils_tasks/navdp_minco_adapter.py`。

### 4.1 统计范围定义

需要区分两类 MINCO 耗时：

```text
adapter_total_ms:
  Python adapter 从收到 candidates_world 开始，到最终返回 selected optimized trajectory 为止的总耗时。
  包括 top-K 排序、candidate 格式转换、pybind 调用、结果筛选。

cpp_optimize_time_ms:
  单条 candidate 进入 C++ MincoProcessorPy::optimize 后，到 MincoPipeline::optimize 返回的 C++ 耗时。
```

用户重点需要的“MINCO 从传入轨迹到传出轨迹耗时”对应：

```text
adapter_total_ms / env
```

其中还要记录被选中 candidate 的：

```text
selected_cpp_optimize_time_ms
```

### 4.2 在 optimize_candidates() 中增加 candidate 级 timing

当前循环中对每个 candidate 调用：

```python
result = self.processor.optimize(...)
```

请改成：

```python
candidate_call_start = time.perf_counter()
result = self.processor.optimize(...)
candidate_call_ms = (time.perf_counter() - candidate_call_start) * 1000.0
cpp_ms = float(result.get("cpp_optimize_time_ms", result.get("duration", 0.0) * 1000.0))
```

记录到：

```python
candidate_timings.append({
    "selected_index": int(selected_idx),
    "python_call_ms": float(candidate_call_ms),
    "cpp_optimize_time_ms": float(cpp_ms),
    "success": bool(result.get("success", False)),
    "objective": float(result.get("objective", np.inf)),
    "min_esdf": float(result.get("min_esdf", np.nan)),
    "failure_reason": str(result.get("failure_reason", "")),
})
```

如果 pybind 调用抛异常，也记录：

```python
candidate_timings.append({
    "selected_index": int(selected_idx),
    "python_call_ms": float(candidate_call_ms),
    "cpp_optimize_time_ms": float("nan"),
    "success": False,
    "objective": float("inf"),
    "min_esdf": float("nan"),
    "failure_reason": str(exc),
})
```

### 4.3 成功结果新增可视化字段

在 `best is not None` 分支中，新增：

```python
samples = best.get("samples")
samples_np = None
speed_profile = None

if samples is not None:
    samples_np = np.asarray(samples, dtype=np.float64)
    if samples_np.ndim == 2 and samples_np.shape[1] >= 6:
        speed_profile = np.linalg.norm(samples_np[:, 4:6], axis=1)

control_points = best.get("control_points", best.get("sparse_waypoints", None))
if control_points is not None:
    control_points = np.asarray(control_points, dtype=np.float64)
    if control_points.ndim == 2 and control_points.shape[1] >= 2:
        control_points = control_points[:, :2]
    else:
        control_points = None

selected_candidate = np.asarray(
    candidates_world[env_idx][best["selected_index"]],
    dtype=np.float64
)
if selected_candidate.ndim == 2 and selected_candidate.shape[1] >= 2:
    selected_candidate = selected_candidate[:, :2]
else:
    selected_candidate = None

raw_top1 = np.asarray(raw_top1_world[env_idx], dtype=np.float64)
if raw_top1.ndim == 2 and raw_top1.shape[1] >= 2:
    raw_top1 = raw_top1[:, :2]
else:
    raw_top1 = None
```

在 result 中增加：

```python
"control_points": control_points,
"selected_candidate": selected_candidate,
"raw_top1": raw_top1,
"speed_profile": speed_profile,
"adapter_total_ms": elapsed_ms,
"candidate_timings": candidate_timings,
"selected_cpp_optimize_time_ms": float(best.get("cpp_optimize_time_ms", best.get("duration", 0.0) * 1000.0)),
"selected_python_call_ms": float(best.get("python_call_ms", np.nan)),
```

注意：需要在 `scored = dict(result)` 后，将 candidate timing 写入 best 候选：

```python
scored["python_call_ms"] = candidate_call_ms
scored["cpp_optimize_time_ms"] = cpp_ms
scored["selected_index"] = int(selected_idx)
```

### 4.4 失败结果新增字段

在 `_fallback_result()` 中增加：

```python
"control_points": None,
"selected_candidate": None,
"raw_top1": waypoints[:, :2] if waypoints.ndim == 2 and waypoints.shape[1] >= 2 else None,
"speed_profile": None,
"adapter_total_ms": elapsed_ms,
"candidate_timings": [],
"selected_cpp_optimize_time_ms": float("nan"),
"selected_python_call_ms": float("nan"),
```

并让 `_fallback_result()` 支持传入 `candidate_timings`：

```python
def _fallback_result(self, env_idx, raw_top1_world, reason, elapsed_ms, candidate_timings=None):
    ...
    "candidate_timings": candidate_timings or [],
```

### 4.5 日志增加耗时摘要

成功日志改成：

```python
print(
    "[NavDP-Minco] "
    f"env={env_idx} success=1 fallback=0 selected_idx={result['selected_index']} "
    f"objective={result['objective']:.4f} min_esdf={result['min_esdf']:.4f} "
    f"adapter_ms={elapsed_ms:.2f} cpp_ms={result['selected_cpp_optimize_time_ms']:.2f}"
)
```

失败日志改成：

```python
print(
    f"[NavDP-Minco] env={env_idx} success=0 fallback=1 "
    f"adapter_ms={elapsed_ms:.2f} reason={reason}"
)
```

不要打印每个采样点。

---

## 5. 第三部分：PlanningOutput 增加可视化和 timing 字段

修改 `utils_tasks/basic_utils.py`。

当前 `PlanningOutput` 已有：

```python
trajectory_points_world
all_trajectories_world
all_values_camera
sub_pointgoal_pd
is_planning
planning_error
```

新增：

```python
raw_top1_world: Optional[np.ndarray] = None
selected_candidate_world: Optional[np.ndarray] = None
minco_control_points_world: Optional[np.ndarray] = None
minco_samples: Optional[List[np.ndarray]] = None
minco_speed_profile: Optional[List[np.ndarray]] = None
minco_info: Optional[List[dict]] = None
planning_timing: Optional[dict] = None
```

最终类似：

```python
@dataclass
class PlanningOutput:
    trajectory_points_world: Optional[np.ndarray] = None
    all_trajectories_world: Optional[List[np.ndarray]] = None
    all_values_camera: Optional[np.ndarray] = None

    raw_top1_world: Optional[np.ndarray] = None
    selected_candidate_world: Optional[np.ndarray] = None
    minco_control_points_world: Optional[np.ndarray] = None
    minco_samples: Optional[List[np.ndarray]] = None
    minco_speed_profile: Optional[List[np.ndarray]] = None
    minco_info: Optional[List[dict]] = None
    planning_timing: Optional[dict] = None

    sub_pointgoal_pd: Optional[np.ndarray] = None
    is_planning: bool = False
    planning_error: Optional[str] = None
```

---

## 6. 第四部分：SimEsdfBuilder 增加首次 ESDF 构建耗时统计

修改 `utils_tasks/sim_esdf_builder.py`。

### 6.1 在 build_or_load_from_stage() 中统计总耗时

使用 `time.perf_counter()`，不要用 `time.time()`。

新增 import：

```python
import time
```

在 `build_or_load_from_stage()` 开始：

```python
total_start = time.perf_counter()
```

### 6.2 缓存加载耗时

在 `_load_cache()` 前后统计：

```python
cache_start = time.perf_counter()
cached = self._load_cache(cache_path, scene_scale)
cache_ms = (time.perf_counter() - cache_start) * 1000.0
```

如果命中缓存：

```python
cached["timing"] = {
    "source": "cache",
    "cache_load_ms": cache_ms,
    "total_ms": (time.perf_counter() - total_start) * 1000.0,
}
print(f"[SimESDF][Timing] source=cache cache_load_ms={cache_ms:.2f} total_ms={cached['timing']['total_ms']:.2f}")
```

### 6.3 stage 构建内部耗时

统计以下阶段：

```text
extract_mesh_ms
bounds_ms
rasterize_ms
distance_transform_ms
cache_write_ms
total_ms
```

伪代码：

```python
extract_start = time.perf_counter()
points, triangles = self._extract_static_mesh(stage, env_prim_path, scene_scale)
extract_mesh_ms = (time.perf_counter() - extract_start) * 1000.0

bounds_start = time.perf_counter()
# xy bounds / grid size / ground_z / z_low / z_high
bounds_ms = (time.perf_counter() - bounds_start) * 1000.0

raster_start = time.perf_counter()
# triangle sampling / occupied marking
rasterize_ms = (time.perf_counter() - raster_start) * 1000.0

dt_start = time.perf_counter()
# distance_transform_edt
# distance / free
_distance_transform_ms = (time.perf_counter() - dt_start) * 1000.0

cache_write_start = time.perf_counter()
np.savez(cache_path, **esdf)
cache_write_ms = (time.perf_counter() - cache_write_start) * 1000.0

total_ms = (time.perf_counter() - total_start) * 1000.0
```

然后写入 esdf：

```python
esdf["timing"] = {
    "source": "stage",
    "cache_load_ms": cache_ms,
    "extract_mesh_ms": extract_mesh_ms,
    "bounds_ms": bounds_ms,
    "rasterize_ms": rasterize_ms,
    "distance_transform_ms": distance_transform_ms,
    "cache_write_ms": cache_write_ms,
    "total_ms": total_ms,
}
```

注意：`np.savez(cache_path, **esdf)` 不一定能直接保存 dict 类型的 `timing`，因为当前 `_load_cache(... allow_pickle=False)` 不允许 pickle。为避免 cache 问题，建议：

1. 不把 `timing` 写入 npz；
2. `np.savez()` 前构造 `save_esdf = {k: v for k, v in esdf.items() if k != "timing"}`；
3. 运行时返回的 `esdf` dict 可以包含 `timing`。

实现：

```python
save_esdf = {k: v for k, v in esdf.items() if k != "timing"}
np.savez(cache_path, **save_esdf)
```

### 6.4 打印 timing

新增打印：

```python
print(
    "[SimESDF][Timing] "
    f"source=stage extract={extract_mesh_ms:.2f}ms "
    f"bounds={bounds_ms:.2f}ms raster={rasterize_ms:.2f}ms "
    f"dt={distance_transform_ms:.2f}ms cache_write={cache_write_ms:.2f}ms "
    f"total={total_ms:.2f}ms"
)
```

### 6.5 注意当前搜索根路径

当前场景实际 terrain 在：

```text
/World/Scene/terrain
```

默认 `env_prim_path` 应保持：

```python
env_prim_path: str = "/World/Scene/terrain"
```

并且在 `build_or_load_from_stage()` 开头打印：

```python
print(f"[SimESDF] search env_prim_path={env_prim_path}")
```

---

## 7. 第五部分：eval_pointgoal 中统计每个 part 耗时

修改 `eval_pointgoal_wheeled.py`。

### 7.1 新增命令行参数

新增：

```python
parser.add_argument("--timing_log_interval", type=int, default=10)
parser.add_argument("--show_timing_overlay", action="store_true")
```

含义：

```text
timing_log_interval:
  每隔多少次 planning_thread 规划打印一次 timing summary。

show_timing_overlay:
  是否在视频中显示 timing overlay。
```

### 7.2 ESDF 生成耗时记录

在构建 ESDF 后：

```python
esdf_timing = esdf.get("timing", {}) if isinstance(esdf, dict) else {}
print(f"[Timing][ESDF] {esdf_timing}")
```

传入 `VisualizationManager` 或在主循环 overlay 中显示时使用。

### 7.3 planning_thread 内分段统计

在 `planning_thread()` 中增加计数器：

```python
planning_iter = 0
```

可以定义在函数外全局，也可以作为函数局部变量在 while 前初始化。

在每轮规划中统计：

```python
planning_total_start = time.perf_counter()

t0 = time.perf_counter()
# copy input from planning_input
input_copy_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
trajectory_points_camera, all_trajectories_camera, all_values_camera = pointgoal_step(...)
navdp_step_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
# raw top1 camera->world
raw_transform_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
# all candidates camera->world
candidate_transform_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
# construct states
state_build_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
# minco_adapter.optimize_candidates(...)
minco_total_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
# MPC_Controller construction for selected trajectory
mpc_construct_ms = (time.perf_counter() - t0) * 1000.0

planning_total_ms = (time.perf_counter() - planning_total_start) * 1000.0
```

注意：要把 timing 放在真实代码块周围，不要伪统计。

### 7.4 planning_timing 字典

每轮规划结束时构造：

```python
planning_timing = {
    "input_copy_ms": input_copy_ms,
    "navdp_step_ms": navdp_step_ms,
    "raw_transform_ms": raw_transform_ms,
    "candidate_transform_ms": candidate_transform_ms,
    "state_build_ms": state_build_ms,
    "minco_total_ms": minco_total_ms if used_minco else 0.0,
    "mpc_construct_ms": mpc_construct_ms,
    "planning_total_ms": planning_total_ms,
    "used_minco": used_minco,
    "minco_results": batch_minco_info,
}
```

写入：

```python
planning_output.planning_timing = planning_timing
```

### 7.5 定期打印 timing summary

每 `args_cli.timing_log_interval` 次规划打印一次：

```python
if planning_iter % max(1, args_cli.timing_log_interval) == 0:
    print(
        "[Timing][Planning] "
        f"total={planning_total_ms:.2f}ms "
        f"navdp={navdp_step_ms:.2f}ms "
        f"transform_raw={raw_transform_ms:.2f}ms "
        f"transform_candidates={candidate_transform_ms:.2f}ms "
        f"state={state_build_ms:.2f}ms "
        f"minco={planning_timing['minco_total_ms']:.2f}ms "
        f"mpc_construct={mpc_construct_ms:.2f}ms"
    )
```

如果有 MINCO 结果，额外打印每个 env 的摘要：

```python
for env_i, info in enumerate(batch_minco_info):
    print(
        "[Timing][MINCO] "
        f"env={env_i} success={info['success']} fallback={info['fallback']} "
        f"adapter={info.get('adapter_total_ms', np.nan):.2f}ms "
        f"cpp={info.get('selected_cpp_optimize_time_ms', np.nan):.2f}ms "
        f"selected_idx={info.get('selected_index', -1)}"
    )
```

注意：`batch_minco_info` 中也要加入 adapter / cpp timing 字段。

---

## 8. 第六部分：main loop 统计 control / visualization / video 耗时

在主循环中统计：

```text
visualize_ms
mpc_solve_ms
speed_plot_ms
text_overlay_ms
video_write_ms
control_total_ms
env_step_ms
```

### 8.1 读取 planning_timing

主循环中从 output_lock 读取：

```python
current_planning_timing = planning_output.planning_timing.copy() if planning_output.planning_timing is not None else None
```

### 8.2 可视化耗时

调用 `visualize_trajectory()` 前后：

```python
t0 = time.perf_counter()
vis_image = vis_manager[i].visualize_trajectory(...)
visualize_ms = (time.perf_counter() - t0) * 1000.0
```

### 8.3 MPC solve 耗时

当前已有：

```python
t0 = time.time()
opt_u_controls, opt_x_states = mpc.solve(x0[i,:3])
print(f"solve mpc cost {time.time() - t0}")
```

请改成：

```python
t0 = time.perf_counter()
opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])
mpc_solve_ms = (time.perf_counter() - t0) * 1000.0
```

不要每帧无条件打印 `solve mpc cost`，用 timing summary 统一打印。

### 8.4 速度图耗时

```python
t0 = time.perf_counter()
vis_manager[i].update_speed_history(...)
vis_image = vis_manager[i].append_speed_plot(...)
speed_plot_ms = (time.perf_counter() - t0) * 1000.0
```

### 8.5 文本 overlay 和 video write 耗时

```python
t0 = time.perf_counter()
# draw_box_with_text...
text_overlay_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
fps_writer[i].append_data(vis_image)
video_write_ms = (time.perf_counter() - t0) * 1000.0
```

### 8.6 env.step 耗时

```python
t0 = time.perf_counter()
obs, rewards, dones, infos = env.step(action)
env_step_ms = (time.perf_counter() - t0) * 1000.0
```

### 8.7 打印控制侧 timing

每隔固定帧数打印一次，例如使用主循环 `frame_idx`：

```python
if frame_idx % max(1, args_cli.timing_log_interval) == 0:
    print(
        "[Timing][Control] "
        f"vis={visualize_ms:.2f}ms "
        f"mpc_solve={mpc_solve_ms:.2f}ms "
        f"speed_plot={speed_plot_ms:.2f}ms "
        f"text={text_overlay_ms:.2f}ms "
        f"video={video_write_ms:.2f}ms "
        f"env_step={env_step_ms:.2f}ms"
    )
```

---

## 9. 第七部分：VisualizationManager 增强接口

修改 `utils_tasks/visualization_utils.py`。

### 9.1 扩展 visualize_trajectory() 参数

当前函数签名：

```python
def visualize_trajectory(
    self,
    rgb_image,
    depth_image,
    intrinsic,
    trajectory_points,
    robot_pose,
    camera_roll=0,
    all_trajectories_points=None,
    all_trajectories_values=None
):
```

改为兼容扩展：

```python
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
```

新增参数都默认 None，旧代码调用不能崩。

### 9.2 __init__ 增加历史

```python
self.speed_history = deque(maxlen=150)
self.esdf_overlay_cache = None
self.esdf_overlay_pose = None
```

`reset()` 中增加：

```python
self.speed_history.clear()
self.esdf_overlay_cache = None
self.esdf_overlay_pose = None
```

### 9.3 新增 world_to_vis_points()

新增统一坐标转换函数，保证 ESDF、轨迹、控制点不偏移：

```python
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
```

### 9.4 新增 draw_polyline_world()

```python
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
```

### 9.5 新增 draw_points_world()

```python
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
        cv2.circle(vis_image, tuple(p), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis_image, tuple(p), radius, color, -1, cv2.LINE_AA)
    return vis_image
```

---

## 10. 第八部分：BEV 中渲染 ESDF

在 `VisualizationManager` 中新增：

```python
def render_esdf_overlay(self, vis_image, robot_pose, esdf, safe_dist=0.30):
    ...
```

### 10.1 输入

`esdf` dict 包含：

```text
distance
occupied
free
origin
resolution
ground_z
scene_scale
```

### 10.2 映射规则

必须与 `world_to_vis_points()` 互逆：

```python
grid_size = vis_image.shape[0]
center = grid_size // 2

rows, cols = np.indices((grid_size, grid_size))
# vis row = -dy/res + center
# vis col = -dx/res + center
dx = -(cols - center) * self.resolution
dy = -(rows - center) * self.resolution

world_x = robot_pose[0] + dx
world_y = robot_pose[1] + dy
```

注意 OpenCV 图像坐标是 `[row, col]`，不要把 row/col 与 x/y 搞反。

### 10.3 ESDF 查询

```python
distance = np.asarray(esdf["distance"], dtype=np.float64)
origin = np.asarray(esdf["origin"], dtype=np.float64)
esdf_res = float(esdf["resolution"])

mx = np.floor((world_x - origin[0]) / esdf_res).astype(np.int32)
my = np.floor((world_y - origin[1]) / esdf_res).astype(np.int32)

valid = (mx >= 0) & (mx < distance.shape[1]) & (my >= 0) & (my < distance.shape[0])
dist = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
dist[valid] = distance[my[valid], mx[valid]]
```

### 10.4 着色

OpenCV 使用 BGR：

```python
overlay = np.zeros_like(vis_image)

unknown = ~np.isfinite(dist)
occupied = dist <= 0.0
unsafe = (dist > 0.0) & (dist <= safe_dist)
near = (dist > safe_dist) & (dist <= 1.0)
free = dist > 1.0

overlay[unknown] = (30, 30, 30)
overlay[occupied] = (40, 40, 160)   # dark red
overlay[unsafe] = (0, 140, 255)     # orange
overlay[near] = (100, 120, 60)      # muted green/blue
overlay[free] = (30, 50, 30)        # dark free

alpha = 0.45
vis_image[:] = cv2.addWeighted(vis_image, 1.0 - alpha, overlay, alpha, 0)
return vis_image
```

### 10.5 插入位置

在 `visualize_trajectory()` 中，`vis_image = np.zeros(...)` 后、绘制 occupancy 点和轨迹前：

```python
if esdf is not None:
    try:
        vis_image = self.render_esdf_overlay(vis_image, robot_pose, esdf, safe_dist=0.30)
    except Exception as exc:
        print(f"[Visualization] ESDF overlay failed: {exc}")
```

不要让 ESDF 可视化异常中断仿真。

---

## 11. 第九部分：绘制 raw / selected / optimized / control points

在 `visualize_trajectory()` 中，原本已经绘制了 `trajectory_points`。请改为或补充以下绘制顺序：

```python
if raw_trajectory_points is not None:
    self.draw_polyline_world(
        vis_image,
        raw_trajectory_points,
        robot_pose,
        color=(0, 165, 255),   # orange, raw NavDP top1
        thickness=1,
        dashed=True,
    )

if selected_candidate_points is not None:
    self.draw_polyline_world(
        vis_image,
        selected_candidate_points,
        robot_pose,
        color=(255, 255, 0),   # cyan, selected candidate
        thickness=1,
        dashed=False,
    )

if trajectory_points is not None:
    self.draw_polyline_world(
        vis_image,
        trajectory_points,
        robot_pose,
        color=(0, 255, 0),     # green, MINCO optimized or fallback
        thickness=2,
        dashed=False,
    )

if control_points is not None:
    self.draw_points_world(
        vis_image,
        control_points,
        robot_pose,
        color=(0, 255, 255),   # yellow, control points
        radius=4,
    )
```

如果原有 trajectory 绘制逻辑仍然存在，避免重复画两遍。建议把原有轨迹绘制逻辑替换为 helper 调用。

### 11.1 BEV legend

在 BEV 左上角加小 legend：

```python
legend_y = 16
cv2.putText(vis_image, "green: MINCO opt", (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
cv2.putText(vis_image, "orange: raw", (8, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1, cv2.LINE_AA)
cv2.putText(vis_image, "yellow: ctrl", (8, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
```

---

## 12. 第十部分：速度折线图

在 `VisualizationManager` 中新增：

```python
def update_speed_history(self, actual_v, actual_w, cmd_v=None, cmd_w=None, planned_v=None):
    self.speed_history.append({
        "actual_v": float(actual_v) if actual_v is not None else np.nan,
        "actual_w": float(actual_w) if actual_w is not None else np.nan,
        "cmd_v": float(cmd_v) if cmd_v is not None else np.nan,
        "cmd_w": float(cmd_w) if cmd_w is not None else np.nan,
        "planned_v": float(planned_v) if planned_v is not None else np.nan,
    })
```

新增：

```python
def append_speed_plot(self, image, speed_max=1.0):
    ...
```

实现要求：

1. 在 image 底部追加 180px 高速度图；
2. 黑底；
3. x 轴显示最近 150 帧；
4. y 轴范围自适应，但至少覆盖 `[0, speed_max]`；
5. 三条曲线：
   - actual_v：白色；
   - cmd_v：绿色；
   - planned_v：黄色；
6. 显示当前数值。

参考实现：

```python
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
        (series_to_points(actual), (255, 255, 255)),
        (series_to_points(cmd), (0, 255, 0)),
        (series_to_points(planned), (0, 255, 255)),
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
```

---

## 13. 第十一部分：视频中显示 timing overlay

如果 `args_cli.show_timing_overlay=True`，在 `eval_pointgoal_wheeled.py` 的主循环中，将 timing 信息画到视频上。

建议新增 helper：

```python
def draw_timing_overlay(image, planning_timing, control_timing):
    if planning_timing is None and control_timing is None:
        return image
    y = 20
    lines = []
    if planning_timing is not None:
        lines.append("PLAN total %.1f navdp %.1f minco %.1f" % (
            planning_timing.get("planning_total_ms", 0.0),
            planning_timing.get("navdp_step_ms", 0.0),
            planning_timing.get("minco_total_ms", 0.0),
        ))
        lines.append("TF raw %.1f cand %.1f mpc_build %.1f" % (
            planning_timing.get("raw_transform_ms", 0.0),
            planning_timing.get("candidate_transform_ms", 0.0),
            planning_timing.get("mpc_construct_ms", 0.0),
        ))
    if control_timing is not None:
        lines.append("CTRL vis %.1f mpc %.1f video %.1f env %.1f" % (
            control_timing.get("visualize_ms", 0.0),
            control_timing.get("mpc_solve_ms", 0.0),
            control_timing.get("video_write_ms", 0.0),
            control_timing.get("env_step_ms", 0.0),
        ))
    for line in lines:
        cv2.putText(image, line, (450, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += 20
    return image
```

调用位置：在 `fps_writer[i].append_data(vis_image)` 之前。

---

## 14. eval_pointgoal 主循环：完整调用逻辑

主循环中，在 `with output_lock:` 后读取新增字段：

```python
current_raw_top1 = None
current_selected_candidate = None
current_control_points = None
current_minco_samples = None
current_minco_speed_profile = None
current_minco_info = None
current_planning_timing = None

with output_lock:
    if planning_output.trajectory_points_world is not None:
        current_trajectory = planning_output.trajectory_points_world.copy()
        current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
        current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None
        current_raw_top1 = planning_output.raw_top1_world.copy() if planning_output.raw_top1_world is not None else None
        current_selected_candidate = planning_output.selected_candidate_world.copy() if planning_output.selected_candidate_world is not None else None
        current_control_points = planning_output.minco_control_points_world.copy() if planning_output.minco_control_points_world is not None else None
        current_minco_samples = planning_output.minco_samples
        current_minco_speed_profile = planning_output.minco_speed_profile
        current_minco_info = planning_output.minco_info
        current_planning_timing = planning_output.planning_timing.copy() if planning_output.planning_timing is not None else None
```

调用可视化：

```python
t0 = time.perf_counter()
vis_image = vis_manager[i].visualize_trajectory(
    images[i],
    depths[i][:, :, None],
    camera_intrinsic.cpu().numpy(),
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
visualize_ms = (time.perf_counter() - t0) * 1000.0
```

MPC solve：

```python
t0 = time.perf_counter()
opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])
mpc_solve_ms = (time.perf_counter() - t0) * 1000.0
v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
```

planned speed：

```python
planned_v = None
if current_minco_speed_profile is not None:
    profile = current_minco_speed_profile[i]
    if profile is not None:
        profile_np = np.asarray(profile, dtype=np.float64).reshape(-1)
        if profile_np.size > 0 and np.isfinite(profile_np[0]):
            planned_v = float(profile_np[0])
```

速度图：

```python
t0 = time.perf_counter()
vis_manager[i].update_speed_history(
    actual_v=float(robot_vel),
    actual_w=float(robot_ang_vel),
    cmd_v=float(v),
    cmd_w=float(w),
    planned_v=planned_v,
)
vis_image = vis_manager[i].append_speed_plot(
    vis_image,
    speed_max=max(1.0, float(args_cli.speed) * 1.5),
)
speed_plot_ms = (time.perf_counter() - t0) * 1000.0
```

文本 overlay：

```python
t0 = time.perf_counter()
# existing draw_box_with_text calls
# add MINCO status box
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
text_overlay_ms = (time.perf_counter() - t0) * 1000.0
```

Timing overlay：

```python
control_timing = {
    "visualize_ms": visualize_ms,
    "mpc_solve_ms": mpc_solve_ms,
    "speed_plot_ms": speed_plot_ms,
    "text_overlay_ms": text_overlay_ms,
    "video_write_ms": 0.0,
    "env_step_ms": 0.0,
}

if args_cli.show_timing_overlay:
    vis_image = draw_timing_overlay(vis_image, current_planning_timing, control_timing)
```

video write：

```python
t0 = time.perf_counter()
fps_writer[i].append_data(vis_image)
video_write_ms = (time.perf_counter() - t0) * 1000.0
control_timing["video_write_ms"] = video_write_ms
```

---

## 15. 颜色约定

OpenCV 是 BGR：

```text
MINCO optimized trajectory: 绿色  (0, 255, 0)
raw NavDP top-1:           橙色  (0, 165, 255)
selected candidate:        青色  (255, 255, 0)
control points:            黄色  (0, 255, 255)
actual_v:                  白色  (255, 255, 255)
cmd_v:                     绿色  (0, 255, 0)
planned_v:                 黄色  (0, 255, 255)
ESDF occupied:             暗红  (40, 40, 160)
ESDF unsafe:               橙色  (0, 140, 255)
ESDF free:                 暗绿  (30, 50, 30)
ESDF unknown:              深灰  (30, 30, 30)
```

---

## 16. 验证流程

### 16.1 import 检查

```bash
python -c "import minco_processor; p=minco_processor.MincoProcessor(); print(dir(p))"
```

应能看到：

```text
configure
set_static_esdf_2d
optimize
```

### 16.2 pointgoal 最小运行

```bash
python eval_pointgoal_wheeled.py \
  --port 8888 \
  --scene_dir /home/alioth/NavDP/assets/scenes/cluttered_easy \
  --scene_index 0 \
  --scene_scale 1.0 \
  --enable_minco \
  --minco_top_k 1 \
  --esdf_force_rebuild \
  --num_envs 1 \
  --timing_log_interval 10 \
  --show_timing_overlay
```

### 16.3 期望日志

```text
[SimESDF] search env_prim_path=/World/Scene/terrain
[SimESDF] source=stage 或 cache
[SimESDF][Timing] source=stage extract=...ms bounds=...ms raster=...ms dt=...ms cache_write=...ms total=...ms
[SimESDF] initial camera query ok=True dist=...
[NavDP-Minco] env=0 success=... fallback=... adapter_ms=... cpp_ms=...
[Timing][Planning] total=...ms navdp=...ms transform_raw=...ms transform_candidates=...ms state=...ms minco=...ms mpc_construct=...ms
[Timing][Control] vis=...ms mpc_solve=...ms speed_plot=...ms text=...ms video=...ms env_step=...ms
```

### 16.4 视频检查

视频中必须能看到：

```text
1. BEV 背景有 ESDF 色块；
2. raw NavDP top-1 橙色虚线；
3. selected candidate 青色线；
4. MINCO optimized trajectory 绿色线；
5. control points 黄色点；
6. all NavDP candidates 仍然显示；
7. 底部速度折线图持续更新；
8. 视频中显示 MINCO success / fallback 状态；
9. 如果开启 --show_timing_overlay，视频中显示 planning / control 耗时摘要。
```

---

## 17. 不要做的事

1. 不要修改 NavDP server 接口。
2. 不要修改 policy 网络。
3. 不要引入 ROS / ROGMap。
4. 不要重新设计 MPC。
5. 不要在 planning_thread 中绘图。
6. 不要让可视化字段缺失导致仿真中断。
7. 不要每帧无节制打印大量 candidate timing。
8. 不要提交 git commit。

---

## 18. 最终回复要求

完成后请说明：

1. 修改了哪些文件；
2. `minco_pybind.cpp` 新增了哪些返回字段；
3. `NavDPMincoAdapter` 新增了哪些可视化字段和 timing 字段；
4. `PlanningOutput` 新增了哪些字段；
5. `SimEsdfBuilder` 新增了哪些 ESDF timing；
6. `eval_pointgoal_wheeled.py` 统计了哪些 planning / control 耗时；
7. `VisualizationManager` 新增了哪些绘制能力；
8. 视频中每种颜色分别代表什么；
9. pointgoal 最小验证命令；
10. 当前是否只在 pointgoal 中实现，其他 eval 是否尚未同步。
