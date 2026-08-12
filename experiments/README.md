# NavDP–MINCO 实验工具链

全量运行、条件筛选和参数调整说明见 [README_PARAMETERS.md](README_PARAMETERS.md)。

本目录实现 `docs/experiments/01～12` 的实验架构，默认输出到仓库根目录 `results/`。同时提供 deterministic mock 后端和 fail-closed 的真实 Isaac/NavDP 编排；本地验收不会启动 Isaac Sim、NavDP 服务、CUDA、MPC 求解或 MINCO 优化。

## 目录

- `analyzers/`：安全、几何、MINCO 时域、标签、校验、单 run、配对和 suite 分析。
- `designers/`：scenario manifest 与 suite 配置校验、确定性 run 展开。
- `recorders/`：非 daemon 异步 CSV/NPZ 写入器和原子状态机。
- `simulators/`：mock、真实 Isaac/NavDP 命令适配与受控进程管理。
- `baselines/raw_navdp/`：RAW 原始坐标/参考/MPC 适配及来源证明。
- `integration/`：episode 注入、记录 hook、trace、视频和热启动生命周期。
- `orchestrators/`：运行、校验、分析、resume 闭环。
- `core/`：稳定 UID、数据模型、schema 和结果布局。
- `configs/`：可直接运行的 mock smoke 配置。
- `tests/`：不启动仿真的测试案例。

## 安全的本地命令

```bash
scripts/run_static_experiments.sh \
  --output results/navdp_minco_paper_local_verification_20260812
scripts/run_simulation_experiments.sh \
  --output results/navdp_minco_paper_local_verification_20260812 --resume
scripts/run_all_experiments.sh
```

默认的一键流程运行本地静态实验、mock 仿真以及 Isaac 动态实验的零进程
dry-run；不会启动 Isaac 或 NavDP 服务。只有下列显式参数才授权真实仿真：

```bash
scripts/run_all_experiments.sh --allow-real-simulation
scripts/run_all_experiments.sh --allow-real-simulation --full-suite
```

`--resume` 仅复用命令和输入哈希完全一致且已验证的 stage；`--retry-failed`
只重试失败 stage。结果根目录内按 `calibration/`、`static/`、`simulation/`、
`paper/`、`validation/` 和 `receipts/` 分层保存。

校验和单 run 分析：

```bash
python -m experiments validate <run_dir>
python -m experiments analyze-run <run_dir>
```

配对分析：

```bash
python -m experiments compare \
  --baseline <raw_or_cold_run> \
  --method <hot_or_minco_run> \
  --output results/<suite_id>/reports/<experiment>/<scene>/<comparison>
```

## 输出结构

```text
results/<suite_id>/
├── suite_config.json
├── experiments/<experiment_id>/<scene>/<variant>/<seed>/<run_id>/
│   ├── run_config.json
│   ├── run_status.json
│   ├── episode_metrics.csv
│   ├── plan_metrics.csv
│   ├── candidate_metrics.csv
│   ├── control_samples.csv
│   ├── timing_samples.csv
│   ├── events.csv
│   ├── traces/*.npz
│   ├── validation_report.{json,md}
│   └── analysis/{run_summary.csv,report.md}
└── reports/suite_report.md
```

mock 输出的每行都有 `data_source=SIMULATED`，suite 报告也有醒目标识。静态、
mock 和 dry-run 只能支持代码正确性、静态能力边界和运行就绪性，不能支持闭环碰撞率、
跟踪误差、Isaac 负载时延或动态性能结论；这些结论只允许来自校验通过的 `REAL` 数据。

## 环境边界

`navdp` 环境用于分析、mock、NavDP 服务和 RAW MPC；`isaaclab` 用于真实仿真入口。真实运行由两个独立 `conda run` 进程组成。本轮只运行 unittest、compileall、C++ build、mock 与 dry-run；GPU smoke 见 `docs/REAL_RUN_CHECKLIST.md`。

静态实验完整支持只使用 `navdp` 环境运行。脚本优先读取
`NAVDP_PYTHON`，否则使用 `/home/alioth/miniforge3/envs/navdp/bin/python`；不会静默退回缺少科研依赖的系统 Python。真实仿真可用
`ISAACLAB_PYTHON` 指定解释器，默认发现同一 Conda 根目录下的 `isaaclab` 环境：

```bash
export NAVDP_PYTHON=/home/alioth/miniforge3/envs/navdp/bin/python
export ISAACLAB_PYTHON=/home/alioth/miniforge3/envs/isaaclab/bin/python
scripts/run_static_experiments.sh
scripts/run_all_experiments.sh --allow-real-simulation
```

## 中文论文级视频证据包

每个静态轨迹 GIF、选定案例的 legacy/safe 配对 GIF，以及真实实验生成的
并排 MP4，均关联独立证据目录：

```text
<media_stem>_evidence/
├── caption_zh.md
├── caption_zh.txt
├── frame_metrics.csv
├── event_timeline.csv
├── video_manifest.json
├── evidence_manifest.json
├── validation.json
├── artifact_receipt.json
└── <media>.gif|mp4
```

Caption 固定说明对象、来源、方法、时间基准、样本分母、指标单位、事件定义、
同步误差、缺失/失败、证据边界和结论限制。CSV 中不可用指标保持空值并带
availability 状态，不用数值零代替。验证器实际解码媒体，并核验帧数、分辨率、
时间单调性、事件边界、UID 和所有 SHA-256。无真实 Isaac 数据时动态证据状态为
`PENDING_REAL_SIMULATION`，指标表仅有表头，禁止生成推测数值。
