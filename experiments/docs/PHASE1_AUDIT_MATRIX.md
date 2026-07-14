# 阶段一实验 Pipeline 代码审计矩阵

审计基线：commit `bd3cba4`；审计依据为实际 Python 源码与 `results/navdp_minco_mock_smoke/`，不采用 README 自述作为完成证据。

| 功能 | 计划要求 | 当前文件/产物 | 审计状态 | 缺口与阶段一动作 |
|---|---|---|---|---|
| 统一身份与目录 | 稳定 UID、分层 run | `core/models.py`, `core/layout.py` | 部分实现 | 保留并扩展 suite 状态、scenario manifest 快照 |
| Planning cycle | 每次触发一行，含失败/HOLD/STOP/stale | 无 | 未实现 | 新增 schema、mock 情况、校验和汇总 |
| 有效轨迹表 | `plan_metrics` 只存发布轨迹 | `core/schemas.py`, mock | 部分实现 | validator 强制 published/stale 语义 |
| Trace | versioned NPZ + metadata JSON | mock 写单个简化 NPZ | 未达标 | 新增 `trace_schema.py`、`trace_writer.py` |
| 指标边界 | 几何/时序/yaw wrap/命令/clearance/deadline | `analyzers/metrics.py` | 部分实现 | 补轨迹对齐与运行指标 |
| 统计 | bootstrap/Wilcoxon/McNemar/Wilson/小样本 | `paired.py` 仅均值/中位数 | 未实现 | 新增 `statistics.py` 并报告方法 |
| 绘图 | 07 所列 EXP-01～08 核心图 | 无 | 未实现 | 新增 `visualizers/`，空数据记录 skip |
| 主表 | 五张核心表及扩展表 | 无 | 未实现 | 新增 `result_tables.py`，CSV + Markdown |
| 案例/失败 | 自动选择、索引与报告 | 无 | 未实现 | 新增 selector 与 failure analyzer |
| Run 分析 | summary/data quality/report | 仅基础 summary/report | 部分实现 | 增加 quality 和 cycle 指标 |
| Suite 分析 | EXP-00～08、五表、索引、中性报告 | 仅 3 行报告 | 未实现 | 统一报告生成器和 artifact manifest |
| Mock 覆盖 | 2 场景、3 组、8 情况、碰撞/超时等 | 2 场景、3 组、全成功 | 未达标 | 新 manifest 与确定性 event profiles |
| Resume | COMPLETE 重新校验后跳过 | `suite_runner.py` | 基础实现 | 增加 ANALYZING/suite 状态与分析恢复 |
| 一键入口 | shell → run/validate/analyze/report | 无 | 未实现 | 新增 `scripts/run_all_experiments.sh` |
| Artifact manifest | JSON+CSV、hash/size/source | 无 | 未实现 | 遍历最终产物并自校验 |
| 测试矩阵 | 每个 requirement 映射 test method | 文档表，无 CSV | 未达标 | 生成 `test_case_matrix.csv` |

阶段一不得修改：`eval_*`、NavDP server/policy、MINCO C++、MPC 和 `navdp_raw/`。
