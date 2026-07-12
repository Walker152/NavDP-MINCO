# Codex 任务：优先修复 NavDP 多 Episode 切换时的 Reset 竞争与冷启动失步

## 任务要求

请直接检查并修改当前仓库代码，不要只给方案，不要生成额外计划文档，不要执行任何 Git 提交操作。

本轮只修复 **PointGoal 评测链路的 episode reset 问题**。不要顺带修改 MINCO、MPC、速度振荡、候选数量、视频保存策略或其他评测脚本。完成后给出修改文件、关键改动和测试结果。

当前对应文件主要为：

- `eval_pointgoal_wheeled.py`，以上传的 `eval_pointgoal_wheeled(7).py` 为当前基线；
- `navdp_server.py`；
- `policy_agent.py`；
- `utils_tasks/client_utils.py` 只允许检查接口行为，除非现有封装使同步无法实现，否则不要修改。

---

## 一、已确认的问题

当前 episode 结束后，主线程会执行：

1. 增加 `episode_generation`；
2. 清空 MINCO 缓存并 reset MPC；
3. 调用 `navigator_reset(env_id=i)`；
4. 下一轮继续向规划线程写入新的观测。

但规划线程可能仍在执行旧 episode 的 `pointgoal_step()`，而主线程同时调用 `/navigator_reset_env`。当前二者没有统一同步，导致以下竞争：

```text
旧 episode 的 pointgoal_step 正在执行
→ 新 episode 调用 navigator_reset_env 清空 memory_queue
→ 旧请求继续写回旧图像历史，或 reset 与 step 顺序反转
→ 新 episode 的 NavDP 历史帧被旧数据污染
```

同时，环境自动 reset 后的前几帧可能出现 camera pose 与 robot pose 尚未同步的过渡状态，日志中曾出现 camera-base XY 偏差瞬间达到数米，下一帧又恢复到约 0.005 m。当前这些过渡帧仍可能被送进 NavDP，并被用于局部轨迹到世界坐标的变换。

另外，`policy_agent.py` 在 `memory_queue` 为空时只放入一张当前图像，其余历史位使用全零 padding。新 episode 的第一次推理因此使用“多张黑图 + 一张当前图”，会进一步降低首轮候选质量。

本轮目标是把 reset 做成严格、有序、可验证的状态切换，避免旧 episode 请求、过渡传感器帧和新 episode 历史互相混用。

---

## 二、本轮必须实现的行为

### 1. 在评测端增加 Reset 屏障

在 `eval_pointgoal_wheeled.py` 的全局共享状态附近增加最少量同步状态：

```python
navdp_http_lock = threading.Lock()
reset_in_progress = threading.Event()
reset_pending = np.zeros(args_cli.num_envs, dtype=bool)
reset_stable_count = np.zeros(args_cli.num_envs, dtype=np.int32)
```

增加两个明确常量，不要做成命令行参数：

```python
RESET_CAMERA_BASE_OFFSET_TOL = 0.05  # m
RESET_STABLE_REQUIRED_FRAMES = 3
```

含义：

- `reset_in_progress` 为批量规划屏障；只要任意环境处于 reset 过渡阶段，整个 NavDP batch 暂停。当前 NavDP 请求本身就是 batch 接口，不要在本轮引入复杂的 per-env 部分批处理。
- `reset_pending[i]` 表示对应环境尚未完成传感器稳定检查。
- `reset_stable_count[i]` 记录 camera-base 偏差连续满足阈值的帧数。

不要新增队列、后台线程或复杂状态机类。

---

### 2. 所有 NavDP HTTP step/reset 请求必须按顺序串行化

在评测端：

- 规划线程调用 `pointgoal_step(...)` 时，必须放在 `with navdp_http_lock:` 中；
- 主线程调用 `navigator_reset(env_id=i, ...)` 时，也必须使用同一个 `navdp_http_lock`；
- 主线程必须先设置 `reset_in_progress`，再等待并获取该锁调用 reset。

这样可以保证：

```text
若旧 pointgoal_step 已开始：旧 step 完整结束 → reset_env 执行
若 reset 已开始：规划线程不能再发起新的 step
```

不要只依赖 Flask 服务端锁，因为两个 HTTP 请求到达服务端的先后顺序可能与调用线程的预期不一致。

---

### 3. 规划线程必须拒绝旧 generation 结果

修改 `planning_thread()`，要求：

#### 3.1 请求前检查 reset

每轮开始时先判断：

```python
if reset_in_progress.is_set():
    time.sleep(0.01)
    continue
```

进入 `input_lock` 后再次检查，避免检查与复制之间发生 reset。

#### 3.2 输入必须做快照

当前从 `planning_input` 读取的数组必须使用 `.copy()`，至少包括：

- goal；
- image；
- depth；
- camera position / rotation；
- robot position / yaw；
- linear / angular velocity；
- `episode_generation`。

不要持有主线程下一轮会替换或修改的共享引用。

#### 3.3 HTTP 返回后立即检查

`pointgoal_step()` 返回并释放 `navdp_http_lock` 后，立即检查：

- `reset_in_progress` 是否已置位；
- 当前 `planning_input.episode_generation` 是否仍与请求前捕获的 generation 完全一致。

只要任一条件不满足：

- 丢弃整个结果；
- 不执行坐标变换；
- 不调用 MINCO；
- 不更新 hold cache；
- 不增加 per-env plan id；
- 不发布 planning output；
- 必须把 `planning_output.is_planning` 恢复为 `False`，不能让 UI 或主线程永久认为还在规划。

#### 3.4 发布前再次检查

MINCO 运行期间也可能发生 episode reset。因此在写入 `planning_output` 前，再执行一次相同的 reset/generation 检查。

只有请求 generation 与当前 generation 完全一致，且没有 reset 正在进行时，才能发布轨迹。

允许增加一个很小的辅助函数，例如：

```python
def mark_planning_idle():
    with output_lock:
        planning_output.is_planning = False
```

或一个集中判断 stale generation 的小函数。不要把规划线程整体重构成新类。

---

### 4. Episode done 时原子失效旧状态

当 `dones[i] == True` 时，按下面顺序执行，不能沿用当前“只更新 generation、旧输入仍保留”的方式：

1. `reset_in_progress.set()`；
2. `episode_generation[i] += 1`；
3. 保留当前已有的 MPC reset、`last_applied_plan_id` 失效、MINCO hold cache 清空、per-env plan id 更新；
4. `reset_pending[i] = True`，`reset_stable_count[i] = 0`；
5. 在 `input_lock` 下清空整批旧规划输入；
6. 在 `output_lock` 下清空整批旧规划输出；
7. 在 `navdp_http_lock` 下调用 `navigator_reset(env_id=i, ...)`；
8. 不要在 reset HTTP 返回后立即 `reset_in_progress.clear()`；必须等待传感器稳定。

#### 4.1 必须清空的 planning input

至少设为 `None`：

- `current_goal`；
- `current_image`；
- `current_depth`；
- `camera_pos`；
- `camera_rot`；
- `robot_pos_w`；
- `robot_yaw_w`；
- `robot_lin_vel_w`；
- `robot_ang_vel_w`。

同时把 `planning_input.episode_generation` 更新为当前 generation 的副本。

#### 4.2 必须清空的 planning output

至少清空：

- `trajectory_points_world`；
- `all_trajectories_world`；
- `all_values_camera`；
- `raw_top1_world`；
- `selected_candidate_world`；
- `minco_sparse_waypoints_world`；
- `minco_samples`；
- `minco_speed_profile`；
- `minco_info`；
- `per_env_plan_id`；
- `stop_required`；
- `minco_status`；
- `planning_timing`。

并设置：

```python
planning_output.is_planning = False
planning_output.planning_error = None
planning_output.episode_generation = episode_generation.copy()
```

可以增加一个小型 `invalidate_planning_state_for_reset()` 辅助函数集中完成这些赋值，避免漏项；不要引入新的状态类。

---

### 5. Reset 后等待传感器稳定，再恢复规划

主循环每帧取得 `camera_pos` 和 `robot_pos_w` 后，先处理 `reset_pending`。

对每个 pending env 计算：

```python
offset_norm = np.linalg.norm(camera_pos[i, :2] - robot_pos_w[i, :2])
```

只有同时满足以下条件才算稳定帧：

- camera position、robot position、offset 均为有限数；
- `offset_norm <= RESET_CAMERA_BASE_OFFSET_TOL`。

连续满足 `RESET_STABLE_REQUIRED_FRAMES` 帧后：

```python
reset_pending[i] = False
```

不满足时将对应 `reset_stable_count[i]` 重新置零。

#### 5.1 pending 期间的行为

只要仍有任意 `reset_pending=True`：

- 不向 `planning_input` 写入当前观测；
- 不清除 `reset_in_progress`；
- 因为旧输出已经清空，控制自然使用零动作推进仿真；
- 不允许过渡帧进入 NavDP、世界坐标变换或 MINCO。

#### 5.2 全部稳定后的恢复顺序

当所有 pending env 都稳定后：

1. 先在 `input_lock` 下写入同一帧完整且一致的 goal/image/depth/camera/robot/velocity/generation 快照；
2. 再执行 `reset_in_progress.clear()`；
3. 规划线程从这份完整快照开始新 episode 的第一次请求。

不要先 clear event 再写输入，否则规划线程可能读到空值或半更新状态。

正常非 reset 帧仍按原逻辑更新 `planning_input`。

---

### 6. 服务端保护 `memory_queue`

在 `navdp_server.py` 中增加：

```python
import threading
navdp_state_lock = threading.RLock()
```

使用同一个锁包住以下三个 endpoint 的完整核心逻辑：

- `/navigator_reset`；
- `/navigator_reset_env`；
- `/pointgoal_step`。

要求：

- reset 与 pointgoal inference 不能同时访问或修改 `navdp_navigator`、`memory_queue`；
- 不改变 URL、请求字段、返回 JSON 格式；
- 不修改模型、推理参数或服务端视频输出行为；
- 使用 `with navdp_state_lock:`，确保异常时自动释放锁。

本轮不要求改造 imagegoal、pixelgoal、nogoal 等 endpoint；不要扩大修改范围。

---

### 7. 修复新 episode 的历史帧冷启动

只修改 `policy_agent.py` 中 `step_pointgoal()` 的历史帧构造。

当前队列为空时，不要再使用“一张当前图像 + 前部全零 padding”。改为：

```python
if len(self.memory_queue[i]) == 0:
    self.memory_queue[i] = [process_images[i].copy() for _ in range(self.memory_size)]
    input_image = np.asarray(self.memory_queue[i])
```

之后队列非空时继续保持原来的滑动更新逻辑。

目标是让初始 episode 和每次 `reset_env()` 后的第一次 pointgoal 推理都使用“重复的首张稳定图像”填满历史，而不是黑帧。

不要修改网络结构、输入归一化、depth 处理、critic 排序或轨迹后处理。

---

## 三、日志要求

只增加少量状态切换日志，不要每帧刷屏，不要新增性能统计。

建议保留以下日志：

```text
[EpisodeReset] env=0 generation=3 state=BEGIN
[EpisodeReset] env=0 generation=3 state=READY stable_frames=3 offset=0.005
[Planning] discard stale result captured_generation=[2] current_generation=[3]
```

`WAIT_STABLE` 不要每帧打印；仅第一次异常或固定较低频率打印即可。原有 `[FrameCheck]` 可以保留。

---

## 四、严格禁止的改动

本轮不要修改以下内容：

- `--minco_top_k` 默认值与候选回退策略；
- MINCO 优化器、稀疏化、速度分配、ESDF 安全距离、validation；
- MPC 权重、参考推进、控制约束、速度或角速度；
- DifferentialController；
- 视频写入位置、fps writer 生命周期或 metric 统计；
- 其他 `eval_*_wheeled.py`；
- NavDP 模型权重、网络结构、采样数量；
- 不增加独立 reset 线程、请求队列或复杂类；
- 不进行无关格式化和重构；
- 不执行任何 Git commit、push、reset、checkout。

本轮必须保持 `--minco_top_k 1` 进行验证，以便单独确认 reset 修复效果，不把候选容错改动混入结果。

---

## 五、实现细节注意事项

1. 所有锁的持有范围必须清晰，不能在持有 `input_lock` 或 `output_lock` 时执行 HTTP 请求、NavDP 推理、MINCO 优化或 `env.step()`。
2. `navdp_http_lock` 只包住客户端 HTTP 调用，不要包住后续坐标变换和 MINCO。
3. stale 结果被丢弃时，不能更新 `minco_hold_cache` 或 `per_env_plan_id`。
4. 任意提前 `continue` 前都要保证 `planning_output.is_planning=False`。
5. reset 期间应允许主仿真循环继续以零动作推进，使相机和机器人状态自然恢复同步。
6. 当前 batch 规划设计下，任意 env reset 时短暂停止整个 batch 是可接受且有意的最小改动方案。
7. 保持 Ctrl+C 和原有 `stop_event` 退出逻辑可用，不能因锁导致退出死锁。

---

## 六、测试与验收

### 6.1 静态检查

至少执行：

```bash
python -m py_compile eval_pointgoal_wheeled.py navdp_server.py policy_agent.py
```

若仓库实际路径不同，使用对应真实路径。

### 6.2 运行验证

保持当前其他参数不变，继续使用 Top-1：

```bash
python eval_pointgoal_wheeled.py \
  --port 8888 \
  --scene_dir /home/alioth/NavDP/assets/scenes/cluttered_easy \
  --scene_index 0 \
  --scene_scale 1.0 \
  --num_envs 1 \
  --num_episodes 10 \
  --enable_minco \
  --minco_top_k 1 \
  --minco_safe_dist 0.60 \
  --minco_sample_dt 0.05 \
  --minco_max_vel 1.0 \
  --minco_max_acc 2.0 \
  --minco_max_iterations 64 \
  --esdf_resolution 0.05 \
  --esdf_padding 1.0
```

### 6.3 必须满足的验收现象

1. 每次 episode done 后都有清晰的 `BEGIN → READY` 状态切换。
2. 即使 reset 后第一帧 camera-base offset 很大，该帧也不会进入 `pointgoal_step()`。
3. 只有 camera-base offset 连续 3 帧小于等于 0.05 m 后才恢复规划。
4. reset 前已经发出的旧 generation 请求即使返回，也只会被丢弃，不会进入 MINCO，更不会覆盖新 episode 输出。
5. `/navigator_reset_env` 与 `/pointgoal_step` 不再并发修改 `memory_queue`。
6. 新 episode 第一次 NavDP 推理的 RGB 历史由同一张首个稳定帧重复填充，不含旧 episode 图像，也不含黑色 padding。
7. `planning_output.is_planning` 不会因 stale 结果提前丢弃而永久保持 `True`。
8. 不出现锁死，主循环、HTTP 服务与 Ctrl+C 退出均正常。
9. 即使后续仍可能因候选本身失败，也不应再出现“新一轮开始后由于 reset 失步而整轮一直无轨迹、零动作直到超时”的现象。
10. 代码中未混入 MINCO、MPC、速度振荡和视频保存修复。

---

## 七、完成后回复格式

完成修改后请直接给出：

1. 修改了哪些文件；
2. 每个文件的核心改动；
3. reset 请求与旧规划请求现在如何排序；
4. stale generation 在哪两个阶段被检查并丢弃；
5. 运行了哪些检查，结果如何；
6. 若无法运行 Isaac Sim 完整测试，明确说明未运行，不要虚构结果。
