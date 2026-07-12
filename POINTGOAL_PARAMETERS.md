# PointGoal Wheeled 启动参数说明

本文说明 `eval_pointgoal_wheeled.py` 的启动参数、当前默认值和常用命令。命令默认在仓库根目录 `/home/alioth/NavDP` 下执行。

## 1. 默认启动

当前默认配置已经启用 MINCO 和耗时统计面板，轮速约束默认关闭：

```bash
python eval_pointgoal_wheeled.py
```

等价的关键默认配置为：

```text
场景：cluttered_easy，scene_index=0，scene_scale=1.0
环境数：1
演示回合数：10
MINCO：开启
耗时统计面板：开启
轮速约束：关闭
```

## 2. 参数清单

### 2.1 场景与评估

| 参数 | 类型 | 默认值 | 作用 |
|---|---:|---|---|
| `--scene_dir` | 字符串 | `/home/alioth/NavDP/assets/scenes/cluttered_easy` | 场景资源目录。 |
| `--scene_index` | 整数 | `0` | 选择场景目录中的场景编号。 |
| `--scene_scale` | 浮点数 | `1.0` | 场景缩放比例。通常保持 `1.0`，除非资源单位需要转换。 |
| `--num_envs` | 整数 | `1` | 并行仿真环境数量。增大后会提高显存、内存和规划负载。 |
| `--num_episodes` | 整数 | `10` | 本次评估运行的回合数，也就是演示轮数。 |
| `--stop_threshold` | 浮点数 | `-3.0` | 导航服务使用的停止判定阈值。 |
| `--port` | 整数 | `8888` | NavDP 导航服务端口，必须与服务端一致。 |

### 2.2 MPC 与底盘控制

| 参数 | 类型 | 默认值 | 单位 | 作用 |
|---|---:|---:|---|---|
| `--speed` | 浮点数 | `1.0` | m/s | MPC 最大前向线速度和期望速度上限。 |
| `--mpc_max_yaw_rate` | 浮点数 | `1.0` | rad/s | MPC 最大角速度。 |
| `--mpc_max_yaw_acc` | 浮点数 | `2.0` | rad/s² | MPC 相邻控制步允许的最大角加速度。 |
| `--mpc_max_wheel_speed` | 浮点数 | `0.0` | rad/s | 左右轮最大关节角速度。有限且 `>0` 时启用 MPC 轮速约束；`<=0` 时关闭。 |

注意：`--mpc_max_wheel_speed` 是轮子的角速度上限，不是机器人线速度。仓库没有给出可信的 Dingo 物理最大轮速，因此示例值只用于演示参数写法，实际值应以机器人或仿真执行器配置为准。

### 2.3 MINCO 轨迹优化

| 参数 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `--enable_minco` / `--no-enable_minco` | 布尔 | 开启 | 开启或关闭 MINCO 轨迹优化。 |
| `--minco_top_k` | 整数 | `1` | 从 NavDP 候选中送入 MINCO 尝试优化的候选数量。值越大，可能找到更优轨迹，但规划耗时通常更高。 |
| `--minco_safe_dist` | 浮点数 | `0.60` | 轨迹与障碍物要求的安全距离。单位与场景坐标一致，通常为米。 |
| `--minco_sample_dt` | 浮点数 | `0.05` | MINCO 输出轨迹的采样时间间隔，单位为秒。减小会增加样本数量和数据处理量。 |
| `--minco_max_vel` | 浮点数 | `1.0` | MINCO 位置轨迹的最大速度，通常为 m/s。 |
| `--minco_max_acc` | 浮点数 | `2.0` | MINCO 位置轨迹的最大加速度，通常为 m/s²。 |
| `--minco_max_iterations` | 整数 | `64` | MINCO 优化器最大迭代次数。 |

### 2.4 ESDF 地图

| 参数 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `--esdf_resolution` | 浮点数 | `0.05` | ESDF 栅格分辨率，通常为米/格。越小越精细，但内存与构建时间更高。 |
| `--esdf_padding` | 浮点数 | `1.0` | 在场景边界外为 ESDF 增加的范围。 |
| `--esdf_force_rebuild` | 开关 | 关闭 | 忽略已有缓存并强制重新构建 ESDF。使用时直接添加该参数。 |
| `--esdf_cache_name` | 字符串 | `esdf_2d.npz` | ESDF 缓存文件名。 |
| `--esdf_obstacle_min_height` | 浮点数 | `0.08` | 参与障碍栅格构建的最小高度。 |
| `--esdf_obstacle_max_height` | 浮点数 | `1.50` | 参与障碍栅格构建的最大高度。 |
| `--esdf_fill_footprint` | 整数布尔值 | `1` | 是否填充障碍物投影足迹；`1` 开启，`0` 关闭。 |
| `--esdf_footprint_inflate_cells` | 整数 | `1` | 障碍物足迹额外膨胀的栅格数量。 |

### 2.5 坐标系和耗时统计

| 参数 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `--use_robot_base_frame` | 整数布尔值 | `1` | `1` 表示规划轨迹按机器人基座位置修正，`0` 表示不使用该修正。 |
| `--timing_log_interval` | 整数 | `1` | 每隔多少个控制帧打印一次耗时信息。值为 `1` 时每帧打印。 |
| `--show_timing_overlay` / `--no-show_timing_overlay` | 布尔 | 开启 | 在输出画面中显示或隐藏耗时统计面板。 |

## 3. 布尔开关写法

MINCO 和耗时面板默认开启：

```bash
python eval_pointgoal_wheeled.py \
  --enable_minco \
  --show_timing_overlay
```

关闭它们：

```bash
python eval_pointgoal_wheeled.py \
  --no-enable_minco \
  --no-show_timing_overlay
```

`--esdf_force_rebuild` 是单向开关：不写表示关闭，写入命令表示开启。

## 4. 常用启动命令集

### 4.1 默认配置，不启用轮速约束

轮速约束默认值为 `0.0`，因此可以直接启动：

```bash
python eval_pointgoal_wheeled.py
```

显式写出关闭状态：

```bash
python eval_pointgoal_wheeled.py \
  --mpc_max_wheel_speed 0
```

### 4.2 开启轮速约束

下面用 `12.0 rad/s` 演示参数写法。该值不是仓库确认的 Dingo 物理极限，运行前应替换为执行器的真实上限：

```bash
python eval_pointgoal_wheeled.py \
  --mpc_max_wheel_speed 12.0
```

启用后，同一正数上限会同时用于 MPC 左右轮硬约束和下游 `DifferentialController` 轮速裁剪。

### 4.3 开启耗时统计

耗时面板当前默认开启。显式写法：

```bash
python eval_pointgoal_wheeled.py \
  --show_timing_overlay \
  --timing_log_interval 1
```

每 10 帧打印一次，同时保留画面面板：

```bash
python eval_pointgoal_wheeled.py \
  --show_timing_overlay \
  --timing_log_interval 10
```

关闭画面面板但仍保留终端耗时日志：

```bash
python eval_pointgoal_wheeled.py \
  --no-show_timing_overlay \
  --timing_log_interval 1
```

### 4.4 更换场景目录

切换到 `cluttered_hard` 场景目录：

```bash
python eval_pointgoal_wheeled.py \
  --scene_dir /home/alioth/NavDP/assets/scenes/cluttered_hard \
  --scene_index 0 \
  --scene_scale 1.0
```

选择同一目录中的第 3 个场景（编号从实际资源定义为准）：

```bash
python eval_pointgoal_wheeled.py \
  --scene_dir /home/alioth/NavDP/assets/scenes/cluttered_easy \
  --scene_index 2
```

### 4.5 修改演示轮数

运行 10 个回合：

```bash
python eval_pointgoal_wheeled.py \
  --num_episodes 10
```

使用 4 个并行环境运行 20 个回合：

```bash
python eval_pointgoal_wheeled.py \
  --num_envs 4 \
  --num_episodes 20
```

注意：增加并行环境数可能显著增加 GPU 显存、图像处理、NavDP 推理和 MINCO 优化负载。

### 4.6 完整组合示例

切换场景、运行 10 回合、开启轮速约束并每 5 帧打印一次耗时：

```bash
python eval_pointgoal_wheeled.py \
  --port 8888 \
  --scene_dir /home/alioth/NavDP/assets/scenes/cluttered_hard \
  --scene_index 0 \
  --scene_scale 1.0 \
  --num_envs 1 \
  --num_episodes 10 \
  --enable_minco \
  --mpc_max_wheel_speed 12.0 \
  --show_timing_overlay \
  --timing_log_interval 5
```

同样配置但不启用轮速约束：

```bash
python eval_pointgoal_wheeled.py \
  --scene_dir /home/alioth/NavDP/assets/scenes/cluttered_hard \
  --scene_index 0 \
  --num_episodes 10 \
  --mpc_max_wheel_speed 0 \
  --show_timing_overlay \
  --timing_log_interval 5
```

## 5. 常见注意事项

- 如果系统没有 `python` 命令，请将上述命令中的 `python` 改为 `python3` 或当前 Isaac Lab 环境的 Python 路径。
- `--port` 必须与 NavDP 服务端口一致，否则客户端无法正常连接。
- 修改 `--minco_safe_dist`、ESDF 高度范围或栅格分辨率后，必要时添加 `--esdf_force_rebuild`，避免继续读取不匹配的旧缓存。
- `--minco_top_k` 和 `--minco_max_iterations` 增大会提高计算量；先观察终端和画面中的耗时统计。
- 轮速约束开启后，如果给定上限过小，MPC 可用的线速度和角速度组合会明显收缩。
