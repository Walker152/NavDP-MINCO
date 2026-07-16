# NavDP–MINCO 全量实验运行与参数说明

本文对应 `configs/experiments/full_suite.json`。默认实验精确锁定 `cluttered_easy_0` 和 `cluttered_hard_0` 两个真实场景、三种方法、每组 10 个 episode，共 60 个 episode；所有 run 串行执行，同一时刻只有一个 Isaac 仿真。

## 一键运行

先进行零进程检查：

```bash
cd /home/alioth/NavDP
conda activate navdp
bash scripts/run_all_experiments.sh configs/experiments/full_suite.json \
  --backend isaac --dry-run
```

确认 `results/navdp_minco_full_real/dry_run_plan.json` 中 `run_count=6`、`started_processes=0` 后，运行全量实验：

```bash
bash scripts/run_all_experiments.sh configs/experiments/full_suite.json \
  --backend isaac --allow-real-simulation --resume
```

资源紧张时关闭视频：

```bash
bash scripts/run_all_experiments.sh configs/experiments/full_suite.json \
  --backend isaac --allow-real-simulation --resume --skip-video
```

失败修复后重试 FAILED run：

```bash
bash scripts/run_all_experiments.sh configs/experiments/full_suite.json \
  --backend isaac --allow-real-simulation --resume --retry-failed
```

## 终端进度显示

真实运行期间，主控每 0.5 秒检查当前 run 已落盘的 `episode_metrics.csv`。只有 episode 完成数发生变化时才输出新行，例如：

```text
[Progress] run 3/6 START | MINCO-COLD | cluttered_easy_0 | 10 episodes
[Progress] 20/60 episodes | 33.3% | run 3/6 | MINCO-COLD | cluttered_easy_0 | episode 0/10
[Progress] 23/60 episodes | 38.3% | run 3/6 | MINCO-COLD | cluttered_easy_0 | episode 3/10
[Progress] run 3/6 COMPLETE | MINCO-COLD | cluttered_easy_0
```

百分比以全 suite 的 60 个 episode 为分母。`--resume` 跳过已经完成且校验通过的 run 时，会显示 `SKIP_COMPLETE` 并把对应 episode 计入全局进度；失败会显示 `FAILED`，且不会虚报为 100%。

## 默认实验规模与顺序

每个场景使用原始 `pointgoal_start_goal_pairs.npy` 的第 0–9 行。RAW、MINCO-COLD、MINCO-HOT 共享 episode UID、起终点和 NavDP seed。

每条 run 配置均包含：

```json
"scene_ids": ["cluttered_easy_0", "cluttered_hard_0"]
```

因此即使 manifest 以后加入 easy_1～9 或 hard_1～9，本 suite 也不会运行它们。ESDF 使用场景内缓存 `esdf_2d_easy0_hard0_full60_v1.npz` 且 `force_rebuild=false`：COLD 在两个场景分别首次构建一次，HOT 复用对应缓存。

```text
RAW-SPARSE × 10 → RAW-DENSE × 10
→ MINCO-COLD-SPARSE × 10 → MINCO-COLD-DENSE × 10
→ MINCO-HOT-SPARSE × 10 → MINCO-HOT-DENSE × 10
```

## 每组自动记录的参数与资源

每个 run 目录包含：

- `run_config.json`：实际生效的 MINCO、ESDF、RAW MPC、MINCO MPC、视频和场景参数；
- `run_manifest.json`：完整 eval/server 命令、checkpoint SHA-256、Git 状态、Python/conda、CPU/GPU/内存信息；
- `esdf_runtime.json`：MINCO run 的实际 ESDF 原点、shape、XY 边界、高度边界和构建耗时；
- `resource_samples.csv`：NavDP 与 Isaac 进程树的 CPU、RSS、系统内存和 GPU 采样；
- `resource_summary.json`：UTC 开始/结束时间、持续时间和 CPU/RSS/GPU 峰值；
- `videos/*.video_complete.json`：FPS、CRF、scale、codec、像素格式、帧数和图像尺寸。

## 可配置参数

参数集中在 `configs/experiments/full_suite.json` 的 `parameters` 字段。修改参数时必须同时修改 `suite_id`，防止不同条件写入同一结果目录。

### MINCO

| JSON 参数 | 默认值 | 含义 |
|---|---:|---|
| `minco.top_k` | 2 | 送入 MINCO 的 NavDP 候选数量 |
| `minco.optimization_safe_distance_m` | 0.45 | 优化代价使用的 ESDF 安全距离 |
| `minco.validation_safe_distance_m` | 0.4 | C++/Python 轨迹验收与热启动历史检查距离 |
| `minco.sample_dt_s` | 0.05 | MINCO 轨迹时间采样间隔 |
| `minco.max_velocity_mps` | 1.0 | 优化器速度约束 |
| `minco.max_acceleration_mps2` | 1.0 | 优化器加速度约束 |
| `minco.max_iterations` | 64 | 最大优化迭代数 |
| `minco.penalty_weight_pos` | 10000 | 位置约束惩罚 |
| `minco.penalty_weight_vel` | 1000 | 速度约束惩罚 |
| `minco.penalty_weight_acc` | 1000 | 加速度约束惩罚 |
| `minco.penalty_weight_attractor` | 20 | 路径吸引项权重 |
| `minco.time_weight` | 0.1 | 总时间代价权重 |
| `minco.time_barrier_weight` | 10 | 时间 barrier 权重 |

### ESDF

| JSON 参数 | 默认值 | 含义 |
|---|---:|---|
| `esdf.resolution_m` | 0.05 | 栅格分辨率；越小越精细、内存越高 |
| `esdf.padding_m` | 1.0 | 场景 XY 外扩边界 |
| `esdf.cache_name` | `esdf_2d.npz` | ESDF 缓存名 |
| `esdf.force_rebuild` | false | 忽略缓存并重建 |
| `esdf.obstacle_min_height_m` | 0.08 | 障碍物最低高度 |
| `esdf.obstacle_max_height_m` | 1.5 | 障碍物最高高度 |
| `esdf.fill_footprint` | true | 是否闭运算并填充占据区 |
| `esdf.footprint_inflate_cells` | 1 | 占据区膨胀格数 |

改变 ESDF 分辨率、高度或膨胀参数时，建议同时设置新的 `cache_name` 并令 `force_rebuild=true`。

### MINCO 时域 MPC

| JSON 参数 | 默认值 | 含义 |
|---|---:|---|
| `minco_mpc.desired_v_mps` | 0.5 | 期望线速度 |
| `minco_mpc.w_max_radps` | 0.5 | 最大角速度 |
| `minco_mpc.max_acceleration_mps2` | 1.0 | 命令加速度约束 |
| `minco_mpc.max_yaw_acceleration_radps2` | 1.0 | 角加速度约束 |
| `minco_mpc.max_wheel_speed_radps` | null | 车轮角速度约束；null 表示关闭 |

MINCO MPC 的 `N=15`、`ref_gap=3`、Q/R 类权重、终端权重和 IPOPT 参数会完整记录在 `run_config.json`。这些参数当前作为控制器实现常量记录，不通过 suite JSON 修改。

RAW 必须保持原 NavDP 基线：`N=15`、`T=0.1`、`ref_gap=3`、`Q=[10,10,0]`、`R=[0.02,0.15]`、`v=w=0.5`。不要为包含 RAW 的正式对比修改 RAW 参数，否则不再是原生基线。

### 视频与场景

| JSON 参数 | 默认值 | 含义 |
|---|---:|---|
| `video.enabled` | true | 是否要求视频；CLI `--skip-video` 可临时关闭 |
| `video.fps` | 10 | 编码 FPS |
| `video.crf` | 23 | H.264 CRF；越大文件越小、画质越低 |
| `video.scale` | 1.0 | 输出图像缩放比例 |
| `scene.scale` | 1.0 | USD 场景缩放 |

## 按实验情景运行

以下情景应复制 `full_suite.json` 为新配置，并使用不同 `suite_id`。

### 只跑一个固定场景

在每条 `runs` 中使用精确 ID：

```json
"scene_ids": ["cluttered_easy_0"]
```

或：

```json
"scene_ids": ["cluttered_hard_0"]
```

运行：

```bash
bash scripts/run_all_experiments.sh <新配置.json> \
  --backend isaac --allow-real-simulation --resume
```

### 只跑一种方法

`runs` 只保留对应条目：

```json
{"experiment_id":"EXP-ALL_data_collection", "variant":"raw", "warm_start_mode":"cold"}
```

```json
{"experiment_id":"EXP-ALL_data_collection", "variant":"minco-cold", "warm_start_mode":"cold"}
```

```json
{"experiment_id":"EXP-ALL_data_collection", "variant":"minco-hot", "warm_start_mode":"gated"}
```

### 安全距离敏感性实验

分别创建不同安全距离的 suite，并同时明确
`minco.optimization_safe_distance_m` 与 `minco.validation_safe_distance_m`。
优化距离应不小于验收距离；每个 suite 使用独立 `suite_id`。RAW 数据可作为共同基线，
但正式统计必须校验配置一致性。

### ESDF 分辨率实验

建议比较 0.05 和 0.10 m：

```json
"esdf": {
  "resolution_m": 0.1,
  "padding_m": 1.0,
  "cache_name": "esdf_2d_res010.npz",
  "force_rebuild": true,
  "obstacle_min_height_m": 0.08,
  "obstacle_max_height_m": 1.5,
  "fill_footprint": true,
  "footprint_inflate_cells": 1
}
```

### MINCO 权重消融

每次只改变一个权重并保持其余参数不变。例如关闭 attractor：

```json
"penalty_weight_attractor": 0.0
```

降低位置惩罚：

```json
"penalty_weight_pos": 5000.0
```

提高时间代价：

```json
"time_weight": 0.5
```

### COLD 与 HOT 热启动对比

`runs` 只保留 MINCO-COLD 和 MINCO-HOT。两组仍共享相同 10 个 episode：

```json
"runs": [
  {"experiment_id":"EXP-04_warm_start", "variant":"minco-cold", "warm_start_mode":"cold"},
  {"experiment_id":"EXP-04_warm_start", "variant":"minco-hot", "warm_start_mode":"gated"}
]
```

## 注意事项

- 先 dry-run，再执行真实仿真。
- 不要让两个 suite 同时使用端口 8888。
- 资源不足时使用 `--skip-video`，不要并行运行多个 suite。
- 修改条件后必须更换 `suite_id`，必要时更换 ESDF cache 名。
- `--resume` 只跳过完成且校验通过的 run；FAILED run 需要 `--retry-failed`。
- mock、DRY_RUN 和 REAL 数据有强制来源标记，不能把 mock 数值作为算法结论。
