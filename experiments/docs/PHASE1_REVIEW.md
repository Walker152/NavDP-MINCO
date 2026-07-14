# 阶段一完成复检

复检基线：阶段一工作树；禁止项检查对象包括 `eval_*`、NavDP server/policy、MINCO C++、MPC 和 `navdp_raw/`。

## 06 总纲复检

- `planning_cycles.csv`、versioned trace、六张既有 CSV、状态与分层目录均已落盘。
- mock suite 为 2 场景、3 variants、8 scenario types；RAW/COLD/HOT 共享 episode UID。
- `suite_status.json`、`scenario_manifest.json`、run validation/analysis、EXP-00～08 和总报告均存在。
- 阶段一未修改真实主体文件；证据由提交前 `git diff --name-only` 再确认。

结论：06 中阶段一范围达标；真实 backend/RAW 属于阶段二，未提前混入。

## 07 报告与产物复检

- Suite report 包含配置/来源、覆盖与质量、核心结果、EXP 摘要、失败、边界和索引。
- 每个 EXP 目录包含 `report.md`、`summary_metrics.csv`、`paired_metrics.csv`、`plots/`、`tables/`。
- 七张核心 CSV 和 Markdown 表均含 n、区间、baseline delta、relative change 和 data_source 字段。
- 63 张核心 PNG 非空；每张由单独 figure 生成，标题含 n，带 `SIMULATED DATA` 水印；`plot_index.csv` 记录状态。
- 自动生成 representative cases、failure index/report；mock 结果不作算法优劣结论。

结论：07 列出的 EXP-01～08 核心图名均有产物；动态 plan/episode 图使用确定性 `mock` 标识实例化。

## 08 Pipeline 复检

- 审计矩阵在 `PHASE1_AUDIT_MATRIX.md`；不是依据 README 自述。
- cycles 的成功、失败、HOLD_LAST、STOP 分母由 mock 覆盖，plan table 只写 published 轨迹。
- Trace 每 cycle 一份 NPZ 与 metadata JSON；实际结果为 24/24，artifact 校验无错误。
- 轨迹 elapsed shift、异频率、yaw wrap、旧轨迹结束、命令变化和 deadline 有单测。
- bootstrap、Wilson、McNemar、Wilcoxon/小样本规则均返回 method，依赖不可用时不安装。
- 一键 shell 串联 run、validation、run analysis、suite reports 和 artifact manifest。

结论：08 的阶段一功能已实现；测试矩阵位于 `experiments/test_case_matrix.csv`。

## 10 编排中的阶段一复检

- Suite 支持稳定 run hash、两场景、三 variants；同一数据采集 run 供 EXP-00～08 分析，不为每张图重复运行。
- COMPLETE run 重新校验后 resume；实际第二次运行 `completed=0, skipped=6, failed=0`。
- artifact JSON/CSV 包含 path/hash/size/source，校验结果 `[]`。
- Shell 为薄包装，接受 backend/resume/retry/dry-run/skip-video/analysis-only 参数。

结论：mock 编排达标；isaac dry-run、安全门和进程命令属于阶段二。

## 11 Pipeline 清单复检

- Pipeline 清单中的 cycles、trace、OOB、时间加权、统计、表、图、案例、失败、artifact、水印、report 全部有测试或产物证据。
- `test_case_matrix.csv` 映射 TC-01～TC-28 到真实 unittest method。
- 未启动 Isaac、NavDP server、CUDA、真实 MINCO 或 MPC。

## 验收结果

```text
unittest: 26 tests, 0 failures, PASS
compileall: PASS
mock first run: completed=6 skipped=0 failed=0
mock resume: completed=0 skipped=6 failed=0
plots: 63 PNG, zero-size=0
core CSV tables: 7
traces: 24 NPZ + 24 metadata JSON
artifact validation errors: 0
temporary files: 0
```
