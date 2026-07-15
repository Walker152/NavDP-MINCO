# 阶段二复检（已被 2026-07-15 重新审计更新）

旧版曾错误声称 RAW 不运行原 MPC 也满足要求。当前有效结论与证据以 `REAUDIT_06_12_MATRIX.md` 和 `FINAL_AUDIT.md` 为准。

## 09 RAW 与真实主体适配复检

- `navdp_raw/` 被当作只读来源；`provenance.json` 与 `PROVENANCE.md` 固定四个源 hash、原 MPC 参数、Q/R、定速和适配边界。
- RAW adapter 已包含原 CasADi MPC 的可执行最小复刻，并在 eval 控制循环由 factory 明确选择；静态测试不构造或求解它。
- `IsaacNavDPBackend` 无 Isaac/torch 顶层 import；`run()` 无显式 allow 时 fail closed。
- eval 已增加 15 项实验 CLI、`num_envs=1` 门、动态 headless、视频/可视化独立分支和 monitor hook。
- hook 顺序覆盖 episode start、planning cycle、有效 plan、control、episode done、reset/finally；stale/HOLD/STOP 不写新 plan。
- `EpisodeVideoRecorder` 用合成帧生成可读取 MP4 和 complete metadata。
- NavDP server 提供 `/health`、run/seed/output/video/log CLI，服务端视频默认关闭；policy 的 5 个 diffusion `randn` 均使用显式 generator。
- Preview/Commit 在 Python lifecycle、adapter、C++ pipeline 和 pybind 均有接口；preview 不改 history，commit 发生在 Python ESDF、generation/stale 和最终发布之后；reset 清理；速度误差大回 COLD。

结论：09 静态适配完成；真实运行、性能和硬件行为仍须 GPU smoke。

## 10 一键编排复检

- `static_real_suite.json` 使用真实 USD/hash 与原 pointgoal NPY 行，产生 DENSE/SPARSE × RAW/COLD/HOT 共 6 条 eval 命令和 6 条 NavDP server 命令。
- RAW 含 `--no-enable_minco`，COLD 为 `--enable_minco --warm-start-mode cold`，HOT 为 `--enable_minco --warm-start-mode gated`。
- 三组共享 manifest、episode UID、seed、NavDP seed 和 headless/video/monitor 参数。
- `--backend isaac --dry-run` 输出 `started_processes=0`；未带 dry-run 或 allow 时拒绝。
- Shell 仍是薄包装；mock resume 未受阶段二影响。

结论：10 的静态真实编排、安全门与三组命令达标；真实进程组清理只可在未来受控 smoke 验证。

## 11 静态验收复检

- 未启动 Isaac Sim/Lab、NavDP server、CUDA、真实 MINCO optimizer、真实 MPC 或相机。
- 允许项已执行：unittest、compileall、AST、fake event、dry-run、合成 MP4、C++ build（未运行 compile-test/optimizer）。
- `navdp_raw/` hash 由测试重新计算且无 diff。
- C++ target `minco_processor`, `minco_processor_compile_test`, `_minco_processor` 编译通过，仅验证构建。
- 测试矩阵扩展至 TC-46，覆盖 11 指定的 Pipeline 和主体适配类别。

## 阶段二验收结果

```text
unittest: 71 tests, PASS
compileall: PASS
C++ build: PASS
mock resume: PASS
isaac dry-run: 6 commands, started_processes=0
synthetic MP4: readable, 4 frames, metadata complete
RAW provenance errors: 0
```
