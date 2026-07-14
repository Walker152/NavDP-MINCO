# Evaluation Pipeline and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development and execute inline task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 06/07/08/10 中属于阶段一的全部数据、分析、绘图、报告、校验和一键 mock 流水线要求。

**Architecture:** 保持现有端口与适配器边界。记录契约集中在 `core/`，mock 只产生确定性事件，分析器只读取落盘产物，visualizer 只消费表与 trace，orchestrator 串联校验、分析和 artifact 索引。

**Tech Stack:** Python 3.10、NumPy、Matplotlib、标准库 csv/json/hashlib/unittest、可选 SciPy fallback。

---

### Task 1: Planning cycle 与 trace 契约

**Files:** `experiments/core/schemas.py`, `experiments/core/trace_schema.py`, `experiments/recorders/trace_writer.py`, `experiments/tests/test_pipeline_contracts.py`

- [ ] 写失败测试：cycle schema 含全部必需字段；trace 写出 NPZ 与 metadata，shape/dtype/unit/version 可校验。
- [ ] 运行该测试，确认因模块/字段缺失失败。
- [ ] 实现 schema、writer 和 validator；再运行至通过。

### Task 2: 指标与统计

**Files:** `experiments/analyzers/metrics.py`, `experiments/analyzers/statistics.py`, `experiments/tests/test_advanced_metrics.py`

- [ ] 写失败测试：elapsed shift、异频率、yaw wrap、旧轨迹结束、command delta、deadline、bootstrap、Wilson、McNemar 和小样本规则。
- [ ] 运行并确认 RED；实现最小数值逻辑；运行至 GREEN。

### Task 3: 完整 deterministic mock 与校验

**Files:** `experiments/configs/mock_full_scenarios.json`, `experiments/configs/mock_full_suite.json`, `experiments/simulators/mock_backend.py`, `experiments/analyzers/validator.py`, `experiments/tests/test_mock_coverage.py`

- [ ] 写失败测试验证八类场景、三 variant、collision/timeout、修复/拒绝、HOT 接受/拒绝和 cycle 分母。
- [ ] 实现 profiles、trace 和严格语义校验；运行至通过。

### Task 4: 绘图、表格、案例和报告

**Files:** `experiments/visualizers/*.py`, `experiments/analyzers/result_tables.py`, `case_selector.py`, `failure_cases.py`, `artifact_manifest.py`, `suite_analysis.py`, `experiments/tests/test_reporting.py`

- [ ] 写失败测试，要求 EXP-00～08 目录、07 核心图名或 skip reason、七张 CSV/Markdown 表、representative/failure、suite report 和 hash manifest。
- [ ] 实现共享单图 helper 与各 EXP 规格；运行至通过。

### Task 5: 编排、CLI 与一键脚本

**Files:** `experiments/orchestrators/suite_runner.py`, `experiments/cli/main.py`, `scripts/run_all_experiments.sh`, `experiments/tests/test_full_pipeline.py`

- [ ] 写失败测试：一键 mock、resume、analysis-only 和参数透传。
- [ ] 实现 shell 薄包装与完整分析生命周期；运行至通过。

### Task 6: 阶段一复检与提交

**Files:** `experiments/docs/PHASE1_REVIEW.md`, `experiments/test_case_matrix.csv`

- [ ] 运行 unittest、compileall、CLI help 和一键 mock。
- [ ] 逐条核对 06、07、08、10、11 的 Pipeline 项，记录文件/测试/产物证据和 skipped plot。
- [ ] 确认未改真实主体与 `navdp_raw/`，提交 `feat(experiments): complete evaluation pipeline and reporting`。
