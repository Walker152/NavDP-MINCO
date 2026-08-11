# NavDP 一键科研实验流水线设计

## 1. 目标

建立一套不依赖已删除历史产物、从当前代码与版本化配置出发的实验流水线。流水线需要覆盖静态机器人标定、MINCO 静态轨迹优化、legacy 与 `safe_corridor_v1` 配对比较、初始状态能力边界扫描、动态 Best2/Worst2 准备、可选 Isaac 仿真、数据质量验证、论文级图表、报告和完整产物索引。

默认执行必须安全：完成所有本地静态实验、mock 冒烟和 Isaac dynamic dry-run，但不启动真实 Isaac 或 NavDP server。只有显式传入 `--allow-real-simulation` 才允许启动真实仿真。

## 2. 设计原则

1. Python 模块负责业务编排、状态、校验、统计和收据；Shell 只负责环境定位和用户入口。
2. 所有科研图表只读取当前运行生成的 CSV/JSON/NPZ，不允许硬编码实验结果。
3. 每个阶段具有独立输入、输出、状态和哈希，可单独运行，也可由总入口组合。
4. 结果目录默认不可覆盖。`--resume` 只跳过通过完整校验且输入哈希未变化的阶段。
5. synthetic、mock、dry-run 和 Isaac real 数据必须明确区分，不得混入同一性能结论。
6. validation 失败的阶段不得标记 COMPLETE；失败产物保留，用于审计和恢复。
7. 安全约束失败只能尝试下一候选、重新验证后 HOLD 或 STOP，不能回退到未通过安全检查的 RAW 路径。

## 3. 总体架构

### 3.1 Python 编排层

新增实验工作流模块，提供三个稳定子命令：

- `run-static-workflow`：本地标定、构建、静态 benchmark、能力边界、分析和论文产物。
- `run-simulation-workflow`：动态 materialization、dry-run，以及显式授权后的 dynamic pilot/full suite。
- `run-all-workflows`：依次调用前两者并生成顶层验收报告。

编排器只调用已有校准、静态、suite runner、analyzer 和 validator API，不通过解析终端文本判断成功。

### 3.2 Shell 入口层

提供：

- `scripts/run_static_experiments.sh`
- `scripts/run_simulation_experiments.sh`
- `scripts/run_all_experiments.sh`

Shell 入口从仓库位置解析绝对路径，优先使用 `NAVDP_PYTHON`，否则使用 `navdp` conda 环境，再回退到当前 Python。入口负责设置 `PYTHONPATH`、打印执行环境并将参数原样交给 Python CLI。

### 3.3 结果目录

每次新运行创建不可覆盖的版本目录：

```text
results/navdp_minco_paper_<UTC>/
  calibration/
  build/
  tests/
  static/
    legacy/
    safe_corridor_v1/
    comparison/
  boundary/
  dynamic_readiness/
  simulation/
    dynamic_pilot/
    full_suite/
  paper/
    figures/
    tables/
    captions/
    report.md
  validation/
  logs/
  experiment_receipt.json
  artifact_manifest.csv
  artifact_manifest.json
```

用户可用 `--output` 指定目录。未指定时使用 UTC 时间生成。`--resume` 要求目录内已有 workflow receipt，且配置、代码和关键输入哈希兼容。

## 4. 基础链路修复

### 4.1 evaluator 路径

Isaac backend、测试、自检脚本和文档统一解析 `run_scripts/eval_pointgoal_wheeled.py`。路径由仓库根目录构造，不依赖调用时工作目录。

### 4.2 标定参数单一来源

`configs/robots/dingo_calibration_v1.json` 是机器人几何的默认真值来源。默认 effective parameters 从标定配置获取：

- wheel radius、wheel base、max wheel speed；
- circumscribed radius；
- validation safe distance；
- optimization safe distance。

start exemption 是独立策略参数，不能假装来自标定。suite override 允许覆盖默认值，但必须记录 `defaults`、`overrides`、`effective` 三层，并校验：

- 数值有限；
- optimization distance 不小于 validation distance；
- validation distance 不小于机器人碰撞半径；
- wheel geometry 和限制为正；
- safe profile 不允许 null wheel limit。

### 4.3 suite 配置语义

CLI 显式参数优先于 suite JSON；未提供 CLI 参数时采用 suite 的 `resume`、`retry.failed` 和 `analysis.enabled`。run/suite receipt 保存解析前配置及最终行为，防止“写了但未生效”。

### 4.4 closure 去历史目录依赖

closure 接受本次 workflow root，不再硬编码日期目录。阶段缺失时：

- 被请求生成的阶段从配置执行；
- 未被请求但为下游必需的阶段给出明确错误；
- 不执行无条件 `copyfile`；
- legacy 和 safe baseline 均可从空目录重建。

## 5. 静态实验工作流

### 5.1 构建与算法门禁

执行 CMake configure/build，并运行 native compile/algorithm test。保存：

- CMake 命令和退出码；
- compiler、Python ABI、extension 路径及 SHA-256；
- stdout/stderr 日志；
- build receipt。

### 5.2 静态标定

运行 USD/config 静态提取、标定 profile 校验和可视化报告。动态未测字段保持 null，并标记 `READY_FOR_CALIBRATION_RUN`，不得生成模拟测量值。

### 5.3 legacy 与 safe benchmark

对同一 synthetic case catalogue 分别运行 legacy 和 `safe_corridor_v1`。每个 case 至少重复两次，校验状态、失败原因和数值指标的确定性。

静态 validator 必须复核：

- manifest/schema 版本；
- config、native extension、case metadata、case NPZ、ESDF 和产物哈希；
- case 完成集合与唯一性；
- CSV schema、行数、主键、有限值和数据来源；
- repeat count 与 `all_deterministic`；
- 图表和 backing metrics 一一对应。

### 5.4 初始条件能力边界

扫描轴由配置声明，至少包括：

- 初始速度大小和方向；
- 初始加速度方向；
- yaw error；
- yaw rate；
- 起点偏离 guide；
- 起点 clearance；
- guide/path 几何。

静态系统不能忠实表示的完整 hot-start history 单独记录为 `PENDING_DYNAMIC_VALIDATION`，不伪造历史轨迹。

输出完整候选排名、Best2/Worst2、各两项替补、排除原因、代码/config/calibration/case hash 和动态 materialization 参数。

## 6. MINCO 安全逻辑完善

1. corridor connectivity 使用几何胶囊真实交叠/共享覆盖判定，而不是依赖天然相同的折线端点。
2. corridor hard validator 限制 segment progression，避免从当前段任意跳至远端段。
3. adaptive validator 在达到最大深度但仍不满足空间/近边界细分条件时 fail closed，并提供独立 reason。
4. optimizer 输出分项 objective/penalty 诊断，并贯通 pybind、adapter、trace 和 CSV。
5. 增加完整 optimizer ESDF/corridor penalty 有限差分、时间梯度、采样边界、短段、重复点、急转和 corridor overlap 测试。

## 7. 动态仿真工作流

### 7.1 默认 dry-run

从本次 boundary selection 生成四个冻结 case、两个 profile、八条命令。dry-run 必须验证：

- 选择文件、policy、calibration、profile、case 和 scene hash；
- 每个 USD 障碍具有可验证的碰撞属性；
- point-goal 与 manifest 一致；
- evaluator 和 NavDP server 入口存在；
- run count 为 8；
- started processes 为 0。

### 7.2 frame sanity

真实运行开始规划前必须验证：

- robot 无初始 penetration；
-加载 calibration hash 与 receipt 一致；
- case/scene hash 一致；
- 初始线速度和角速度在容差内生效；
- 动态 ESDF clearance 与静态 case 在配置容差内一致；
- case UID 写入 episode metadata。

不能忠实 materialize 时，停止使用该 case，从已冻结替补中按顺序选择，并生成包含原因和新旧 hash 的 substitution receipt。开始任何正式 paired run 后不允许替换。

### 7.3 显式授权运行

`--allow-real-simulation` 允许依次执行 dynamic pilot；`--full-suite` 进一步执行完整 RAW/COLD/HOT suite。两者均保留独立输出和校验。

动态 pilot 的 paired profile 仅改变 constraint profile，warm start、checkpoint、speed、MPC、Top-K、scene 和 seeds 固定。

### 7.4 机器真值与控制记录

从环境 termination terms、contact/collision signal 和可用的传感器数据记录：

- canonical termination 与 raw terms；
- contact detected；
- collision object；
- impact force（传感器可用时）；
- termination frame、plan 和 planning cycle；
- actual wheel speeds、limit 和 saturation；
- HOLD/STOP duration；
- executed clearance、reference age 和 stale。

硬件/仿真接口确实不提供的字段保持空值，并在 data-quality report 中说明来源不可用；不得由视频或字符串猜测冲击力/碰撞对象。

## 8. 科研统计和图表

### 8.1 通用规范

每张图输出：

- 300 DPI PNG；
- 矢量 PDF；
- backing CSV 或 JSON；
- caption Markdown；
- figure receipt，包含输入文件 SHA-256、生成器版本、样本量、单位和数据来源。

颜色、方法顺序、字体、线型和缺失值样式统一。失败 case 保留在离散结果分母中；只对成功轨迹定义的连续指标明确条件分母。

### 8.2 静态图表

至少生成：

- legacy/safe 轨迹、guide、corridor、footprint 和 ESDF overlay；
- clearance–arc length；
- velocity/acceleration/jerk/yaw-rate–time；
- 单因素能力边界曲线；
- 两个实际扫描因素构成的二维 heatmap；
- failure reason stack；
- runtime/clearance/guide-deviation Pareto；
- legacy→safe transition matrix；
- 完整 ranking table；
- Best2/Worst2 paired case cards 和动画。

### 8.3 动态图表

只有 REAL 数据存在且通过 validator 时生成：

- paired success/SPL/duration/path length；
- tracking error 与 executed clearance；
- collision/contact/termination；
- planning latency/deadline/reference age；
- HOLD/STOP/stale timeline；
- wheel saturation；
- static prediction vs dynamic outcome；
- legacy/safe synchronized video。

二项指标使用 Wilson 区间；paired 差值优先使用 episode/case 粒度 bootstrap 区间。规划 cycle 不作为独立实验样本。样本不足时只报告描述统计和原始点，不生成夸大的显著性结论。

## 9. 顶层一键流程

默认 `scripts/run_all_experiments.sh` 执行：

```text
preflight
→ build/test
→ static calibration
→ legacy benchmark
→ safe benchmark
→ static paired analysis
→ boundary scan and selection
→ mock suite
→ dynamic materialization/dry-run
→ global validation
→ paper figures/tables/report
→ artifact manifest and final receipt
```

传入 `--allow-real-simulation` 后增加 dynamic pilot。只有同时传入 `--full-suite` 才增加完整 RAW/COLD/HOT 仿真，避免一次授权意外扩大运行规模。

## 10. 状态、恢复和错误处理

每个阶段状态为 `PENDING`、`RUNNING`、`COMPLETE`、`FAILED` 或 `READY_FOR_REAL_RUN`。阶段 COMPLETE 必须同时满足：进程退出成功、产物完整、validator 通过、输入哈希与 receipt 相符。

`--resume` 行为：

- COMPLETE 且 receipt 兼容：跳过；
- FAILED：默认停止，传入 `--retry-failed` 才重跑；
- RUNNING：标记 INTERRUPTED 后重跑；
- 输入 hash 变化：拒绝在原目录续跑，要求新输出版本。

脚本退出码非零表示顶层流程未通过；最终报告列出失败阶段、原因、日志和精确恢复命令。

## 11. 测试策略

所有行为变更使用测试驱动开发。测试层次：

1. 单元测试：路径解析、参数合并/校验、receipt、统计、图表 backing data。
2. 静态集成测试：从空临时目录运行 legacy/safe 小型 benchmark 和比较。
3. workflow 测试：mock backend 与 dynamic dry-run，验证零进程和 resume。
4. native 测试：C++ corridor、adaptive validator、梯度和 pybind diagnostics。
5. 完整回归：`conda run -n navdp python -m unittest discover -s experiments/tests -p 'test_*.py'`。

测试不得依赖已删除的历史 results。需要历史 trace 语义时，在测试临时目录生成最小、自描述、带 hash 的 fixture。

## 12. 验收标准

1. 三个 Shell 入口可从任意工作目录调用。
2. 默认总入口在本地完成静态全流程、mock 和 dynamic dry-run，不启动真实仿真。
3. 静态工作流从空结果目录完整生成 benchmark、边界扫描、全部图表、报告和 manifest。
4. 仿真工作流 dry-run 生成八条可执行命令且 started processes 为 0。
5. 真实仿真只有双重显式授权条件满足时执行对应范围。
6. 完整 Python 测试、C++ 构建和 native 算法测试通过。
7. 所有科研图表数据驱动，PNG/PDF/backing data/caption/receipt 完整。
8. 顶层 validator 能发现被修改或缺失的输入、表格、图表和收据。
9. 工作流不依赖 V3 或 `navdp_minco_longterm_20260726` 历史结果目录。
10. 最终 report 明确区分已证明结论、静态证据、mock/dry-run 证据和仍需真实仿真的结论。
