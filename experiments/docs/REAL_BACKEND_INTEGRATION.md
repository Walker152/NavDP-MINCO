# 真实 NavDP / Isaac Lab 后端接入说明

## 接入边界

新增 `experiments/simulators/isaac_navdp_backend.py` 并实现 `ExperimentBackend.run(run, episodes, writer)`。编排器、schema、校验器和分析器无需导入 Isaac 模块。真实命令必须显式选择未来的 `isaac` 后端；当前 CLI 只接受 `mock`，因此不会误启动仿真。

## eval hook 顺序

1. episode reset 完成后建立稳定 `episode_uid`，并调用 `processor.reset_history(env_id=0)`。
2. NavDP Top-K 产生后写 candidate；RAW Top-1 同时计算几何和统一 ESDF 指标。
3. MINCO 候选使用同一 committed history snapshot 做 preview，未选择候选不得污染历史。
4. 只有 C++ 验证、Python ESDF 验证、episode generation 和 stale 检查全部通过且准备发布给 MPC 的 proposal 才写 plan。
5. 真正发布后才 commit history；discard 其他 proposal。HOLD_LAST 只写 event，不计为新优化成功；STOP 不生成伪 plan。
6. MPC 命令产生后、`env.step` 前写 control。reset/wait/stop 无参考时误差写 NaN。
7. episode 结束写 episode 汇总，writer close 后进入校验；校验通过才标 COMPLETE。

## 热启动 Preview / Commit 数据流

```text
committed safe history snapshot
  → Top-K optimize_preview（共享 snapshot）
  → C++ validation
  → Python ESDF validation
  → select proposal
  → generation/stale check
  → publish MPC
  → commit_history(selected)
  → discard remaining proposals
```

历史必须包含 committed plan UID、episode generation、发布时间、位置/yaw 轨迹和安全验证标记。速度误差超阈值必须返回 COLD_START。

## 环境命令模板（未在本地执行）

分析/推理侧：

```bash
conda activate navdp
python -m experiments validate <run_dir>
```

未来真实仿真侧：

```bash
conda activate isaaclab
python -m experiments run-suite --config <real_suite.json> --backend isaac --resume
```

上面的 `isaac` 后端当前故意未开放，需完成 adapter 和小规模人工批准的 smoke 后再启用。

## 当前接口尚不能可靠获得的字段

在真实 hook 完成前，`history_*`、proposal/commit 诊断、轮速约束、精确 observation-to-command、视频编码完整性和部分 C++ phase timing 不应伪造。记录器应留空，pandas/分析读取时解释为 NaN；报告列出缺失率。
