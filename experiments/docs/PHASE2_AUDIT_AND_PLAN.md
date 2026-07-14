# RAW Baseline and Real Backend Static Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development and execute inline task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不启动 Isaac/NavDP/CUDA/MINCO/MPC 的条件下，完成 RAW 等价封装、真实 backend dry-run、安全门、eval hook、视频、server/seed 与 warm-start proposal 静态适配。

**Architecture:** 原始 `navdp_raw/` 始终只读，通过带 hash 的纯 NumPy adapter 复用其几何参考逻辑。真实 backend 仅构造命令并 lazy import；eval hook、视频和 proposal store 是可独立 fake-event 测试的边界组件。真实主体只增加参数与 hook，不改变算法默认值。

**Tech Stack:** Python 3.10、NumPy、AST/hashlib/unittest、C++ 静态签名检查；不执行重资源组件。

---

## RAW 审计结论

- 文档写作 `NAVDP/navdp_raw`，仓库实际只读路径为 `navdp_raw/`。
- Top-1 路径：`navdp_raw/eval_pointgoal_wheeled.py:80-96`，直接将 NavDP Top-1 相机路径变换到世界坐标后构造原 `MPC_Controller`。
- 原 MPC：`navdp_raw/utils_tasks/tracking_utils.py`，默认 `N=15, desired_v=0.5, v_max=0.5, w_max=0.5, ref_gap=3, T=0.1`；Q=`diag(10,10,0)`、R=`diag(.02,.15)`；控制约束 `v∈[0,v_max]`, `w∈[-w_max,w_max]`。
- 轨迹加密 ratio=50；reference 依据最近点和 `desired_v*ref_gap*T` 弧长选择，末端不足时重复终点。
- success/SPL/distance：原 eval 的 episode done 分支，SPL 为 `clip(euclidean/trajectory_length,0,1)*success`。
- 源 SHA-256：eval `1239da...8f39`，tracking `003e60...159a`，basic `c83cf8...9650`，visualization `641740...9516`。

### Task 1: RAW provenance 与数值等价

**Files:** `experiments/baselines/raw_navdp/{controller.py,trajectory_adapter.py,metrics_adapter.py,equivalence.py,provenance.json,PROVENANCE.md}`, `experiments/tests/test_raw_baseline.py`

- [ ] 写失败测试：源 hash、参数、参考点、Top-1 变换、SPL 等价。
- [ ] 实现无 CasADi 求解的参考逻辑封装和 provenance 校验；运行至通过。

### Task 2: Isaac backend、安全门与 dry-run

**Files:** `experiments/simulators/isaac_navdp_backend.py`, `experiments/configs/static_real_suite.json`, `experiments/orchestrators/suite_runner.py`, `experiments/cli/main.py`, `experiments/tests/test_real_backend_static.py`

- [ ] 写失败测试：lazy import、三 variant 命令差异、seed/headless/video 参数、dry-run 不调用 subprocess、无 allow 时拒绝。
- [ ] 实现 build/validate/run 安全门并运行至通过。

### Task 3: Eval hook 与视频

**Files:** `experiments/integration/eval_hooks.py`, `experiments/recorders/video_recorder.py`, `eval_pointgoal_wheeled.py`, `experiments/tests/test_eval_hooks_video.py`

- [ ] 写 fake event 和合成帧失败测试：READY→cycle→published→control→done、stale/HOLD/STOP、reset、稳定 episode 文件名和 metadata。
- [ ] 实现 bridge/recorder，向 eval 增加 CLI 和最小 hook；不执行 eval。

### Task 4: NavDP server 与 seed 静态接口

**Files:** `baselines/navdp/navdp_server.py`, `baselines/navdp/policy_agent.py`, `baselines/navdp/policy_network.py`, `experiments/tests/test_navdp_static_interfaces.py`

- [ ] 写 AST/文本失败测试：health schema、CLI、默认禁视频、显式 generator/seed。
- [ ] 最小修改接口，不改变采样分布/步数；仅 compileall/AST 验证。

### Task 5: Preview/Commit 生命周期

**Files:** `experiments/integration/warm_start.py`, `utils_tasks/navdp_minco_adapter.py`, MINCO header/cpp/pybind（仅在现有接口可安全扩展时）, `experiments/tests/test_warm_start_lifecycle.py`

- [ ] 写失败测试：preview 不污染、同 snapshot、unsafe/stale/未选拒绝 commit、reset 清空、速度错误回 COLD。
- [ ] 实现 proposal store 与 adapter 诊断；执行 Python 测试和 C++ 静态签名检查，不运行 optimizer。

### Task 6: 阶段二复检与提交

**Files:** `experiments/docs/PHASE2_REVIEW.md`, `experiments/test_case_matrix.csv`

- [ ] 运行全部 unittest、compileall、mock resume、isaac dry-run；核对 09/10/11 每项证据。
- [ ] 比较 `navdp_raw/` tree hash，证明未修改；确认未启动进程。
- [ ] 提交 `feat(experiments): integrate NavDP raw and MINCO experiment adapters`。
