# NavDP–MINCO 两阶段实验交付总体审计

审计依据：`docs/experiments/12_Codex_总执行提示词_两阶段串联.md`，并逐项回查 06–11。重新审计日期：2026-07-15。

> 更正：2026-07-14 版本把“存在 adapter/命令”误判为 RAW 与真实 episode 已接入。该结论撤销；本文件以下内容为重新实现后的证据。此次修改按用户要求保持未提交。

## 1. 阶段与提交边界

- 阶段一提交：`22e4b9d feat(experiments): complete evaluation pipeline and reporting`。
- 本轮补全不提交，不产生新的提交号。
- 阶段一完成并通过 `PHASE1_REVIEW.md` 后才进入阶段二；阶段二逐项复检见 `PHASE2_REVIEW.md`。
- 结果目录 `/results/` 已被 `.gitignore` 忽略，代码、配置、测试和审计文档位于版本控制范围内。

## 2. 需求覆盖结论

| 范围 | 审计证据 | 结论 |
|---|---|---|
| 06–08 数据与报告 | 实际 CSV 驱动表图、严格 schema/validator、resume、mock E2E | PASS（管线） |
| 09 RAW/主体接入 | eval 实际 factory 分流；RAW 原 Top-1 几何 + 原 reference + 原 CasADi MPC；MINCO 使用当前时域 MPC | PASS（静态） |
| 10 编排 | 真实 USD/hash/NPY episode 注入；DENSE/SPARSE × RAW/COLD/HOT；双 conda 进程及定向清理 | PASS（dry-run） |
| 11 验收 | 71 unittest、compileall、C++ build、RAW 数值/来源、真实命令零进程 | PASS（允许范围） |

## 3. 最终门禁记录

```text
MPLCONFIGDIR=/tmp/matplotlib conda run -n navdp python -m unittest discover -s experiments/tests -v
Ran 71 tests in 40.308s — OK

conda run -n navdp python -m compileall -q experiments utils_tasks/navdp_minco_adapter.py eval_pointgoal_wheeled.py baselines/navdp
PASS

cmake --build minco_processor/build -j2
minco_processor / minco_processor_compile_test / minco_processor_py — Built (未执行测试二进制或优化器)

MPLCONFIGDIR=/tmp/matplotlib conda run -n navdp bash scripts/run_all_experiments.sh experiments/configs/mock_full_suite.json --backend mock --resume
completed=0, skipped=6, failed=0

conda run -n navdp bash scripts/run_all_experiments.sh experiments/configs/static_real_suite.json --backend isaac --dry-run
run_count=6, eval_commands=6, server_commands=6, started_processes=0

verify_provenance(Path('.'))
[]
```

## 4. 安全与真实性声明

- 本地未运行 Isaac Sim/Lab、NavDP 端到端推理服务、CUDA 仿真、真实 MINCO 优化或真实 MPC。
- 仅执行 mock、静态接口测试、合成视频、Python 编译和 C++ 构建；dry-run 只生成命令。
- `navdp_raw/` 保持只读且 git diff 为空；四个来源文件的 SHA-256 与 provenance 完全一致。
- mock 结果仅证明实验管线、统计、图表、报告和恢复机制可用，不冒充真实机器人性能数据。

## 5. 真实运行前的剩余外部验证

真实 GPU 环境仍需按 `REAL_RUN_CHECKLIST.md` 完成小规模 smoke，确认 Isaac 相机/物理、NavDP checkpoint、真实 MINCO/ESDF 数值行为、进程清理和硬件性能。该项依赖用户明确授权及可用 GPU，不属于本次“禁止实际仿真”的本地验收范围。

总体结论：06–12 所需实验骨架已连通，静态与模拟门禁通过。它足以启动配对实验，但在完成受控 GPU smoke 前不得宣称 RAW 与 MINCO 的真实性能对比已经完成。
