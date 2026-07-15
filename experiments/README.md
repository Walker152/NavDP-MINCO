# NavDP–MINCO 实验工具链

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
conda activate navdp
python -m unittest discover -s experiments/tests -v
python -m experiments run-suite \
  --config experiments/configs/smoke_suite.json \
  --backend mock --resume
```

真实后端只做零进程 dry-run：

```bash
python -m experiments run-suite \
  --config experiments/configs/static_real_suite.json \
  --backend isaac --dry-run --skip-video
```

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

mock 输出的每行都有 `data_source=SIMULATED`，suite 报告也有醒目标识。不得用这些数值支持算法结论。

## 环境边界

`navdp` 环境用于分析、mock、NavDP 服务和 RAW MPC；`isaaclab` 用于真实仿真入口。真实运行由两个独立 `conda run` 进程组成。本轮只运行 unittest、compileall、C++ build、mock 与 dry-run；GPU smoke 见 `docs/REAL_RUN_CHECKLIST.md`。
