# 交付说明

## 已交付

- 面向 EXP-00～08 的分层结果架构和统一 CSV schema；
- 稳定 manifest/episode UID、suite 校验和 deterministic run 展开；
- ESDF 安全、几何曲率、MINCO 时域、跟踪距离和 situation 标签基础分析器；
- 非 daemon 异步 CSV/NPZ writer、原子 JSON/NPZ 和 run 状态机；
- mock 后端、端到端编排、逐 run 校验、resume、run/suite 报告；
- episode 配对分析和关键配置一致性检查；
- 15 个非仿真测试、测试案例、测试记录和真实生成的最小示例结果。

## 示例交付

- Suite 报告：`results/navdp_minco_mock_smoke/reports/suite_report.md`
- Run 示例：`results/navdp_minco_mock_smoke/experiments/EXP-01_raw_profile/SPARSE/raw/0/run_e84ab08a14dc/`
- 配对示例：`results/navdp_minco_mock_smoke/reports/EXP-06_navigation/SPARSE/raw_vs_hot/`

## 有意未改动

未修改 eval 脚本、NavDP server、policy、C++ MINCO 和 Isaac 配置。原因是用户禁止本地真实仿真，而这些改造若没有真实 smoke 验证会扩大回归风险。后续按 `REAL_BACKEND_INTEGRATION.md` 通过单一 adapter 接入。

## Schema 摘要

所有表共享 suite/experiment/run/variant/scene/seed/episode 身份字段。plan 保存 RAW/MINCO、安全、热启动、优化与规划耗时；candidate 保存 Top-K 诊断；control 保存状态、参考、命令和误差；timing 使用长表；events 保存主次失败原因；episode 保存闭环结果与计数。mock/真实来源通过 `data_source` 强制区分。
