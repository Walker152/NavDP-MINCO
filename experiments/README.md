# NavDP–MINCO 实验工具链

本目录实现 `docs/experiments/01～03` 的独立实验架构。新增实验代码统一归档于此，默认输出到仓库根目录 `results/`。当前交付提供 CPU-only deterministic mock 后端，用于安全验证数据契约与分析流水线；它不会启动 Isaac Sim、NavDP 服务或 CUDA。

## 目录

- `analyzers/`：安全、几何、MINCO 时域、标签、校验、单 run、配对和 suite 分析。
- `designers/`：scenario manifest 与 suite 配置校验、确定性 run 展开。
- `recorders/`：非 daemon 异步 CSV/NPZ 写入器和原子状态机。
- `simulators/`：后端协议与 mock 后端；未来真实后端只在这里接入。
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

`navdp` 环境用于分析、mock 与未来端到端推理接入。`isaaclab` 环境仅用于未来真实仿真，本轮没有运行。当前 `navdp` 环境未安装 pytest，因此测试使用标准库 `unittest`，没有自动安装任何依赖。
