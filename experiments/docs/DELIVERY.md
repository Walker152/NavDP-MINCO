# 交付说明

## 已交付

- 面向 EXP-00～08 的分层结果架构和统一 CSV schema；
- 稳定 manifest/episode UID、suite 校验和 deterministic run 展开；
- ESDF 安全、几何曲率、MINCO 时域、跟踪距离和 situation 标签基础分析器；
- 非 daemon 异步 CSV/NPZ writer、原子 JSON/NPZ 和 run 状态机；
- mock 后端、端到端编排、逐 run 校验、resume、run/suite 报告；
- episode 配对分析和关键配置一致性检查；
- RAW 原始 MPC/几何的真实控制循环分流、真实 pointgoal manifest/episode 注入、双环境进程编排；
- 71 个非仿真测试、测试案例、测试记录和 deterministic mock 示例结果。

## 示例交付

- Suite 报告：`results/navdp_minco_mock_smoke/reports/suite_report.md`
- Run 示例：`results/navdp_minco_mock_smoke/experiments/EXP-01_raw_profile/SPARSE/raw/0/run_e84ab08a14dc/`
- 配对示例：`results/navdp_minco_mock_smoke/reports/EXP-06_navigation/SPARSE/raw_vs_hot/`

## 运行边界

eval、NavDP seed 接口、C++ 热启动诊断和真实后端已完成必要接入，但没有在本地启动任何真实进程。真实性能结论仍依赖 `REAL_RUN_CHECKLIST.md` 的受控 GPU smoke。

## Schema 摘要

所有表共享 suite/experiment/run/variant/scene/seed/episode 身份字段。plan 保存 RAW/MINCO、安全、热启动、优化与规划耗时；candidate 保存 Top-K 诊断；control 保存状态、参考、命令和误差；timing 使用长表；events 保存主次失败原因；episode 保存闭环结果与计数。mock/真实来源通过 `data_source` 强制区分。
