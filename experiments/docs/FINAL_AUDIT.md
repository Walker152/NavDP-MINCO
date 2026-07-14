# NavDP–MINCO 两阶段实验交付总体审计

审计依据：`docs/experiments/12_Codex_总执行提示词_两阶段串联.md`，并逐项回查 06、07、08、09、10、11。审计日期：2026-07-14。

## 1. 阶段与提交边界

- 阶段一提交：`22e4b9d feat(experiments): complete evaluation pipeline and reporting`。
- 阶段二按要求使用提交信息：`feat(experiments): integrate NavDP raw and MINCO experiment adapters`。
- 阶段一完成并通过 `PHASE1_REVIEW.md` 后才进入阶段二；阶段二逐项复检见 `PHASE2_REVIEW.md`。
- 结果目录 `/results/` 已被 `.gitignore` 忽略，代码、配置、测试和审计文档位于版本控制范围内。

## 2. 需求覆盖结论

| 范围 | 审计证据 | 结论 |
|---|---|---|
| 06 数据与指标 | versioned CSV/NPZ、planning cycle、时间对齐、deadline、统计检验 | PASS |
| 07 可视化与表格 | 63 个核心 PNG、7 类核心 CSV/Markdown 表、案例与失败索引 | PASS |
| 08 全量 mock | 2 场景、3 variants、8 情形、6 runs；resume 跳过 6/6 | PASS |
| 09 RAW/主体接入 | RAW hash/provenance、reference/坐标/指标等价测试、eval hooks、视频、seed、HOT lifecycle | PASS（静态） |
| 10 编排 | DENSE/SPARSE × RAW/COLD/HOT 六命令；Isaac backend fail-closed；dry-run 零进程 | PASS（静态） |
| 11 验收 | 45 unittest、compileall、C++ build、合成 MP4、RAW hash、mock resume | PASS |

## 3. 最终门禁记录

```text
MPLCONFIGDIR=/tmp/matplotlib conda run -n navdp python -m unittest discover -s experiments/tests -v
Ran 45 tests in 15.270s — OK

conda run -n navdp python -m compileall -q experiments utils_tasks/navdp_minco_adapter.py eval_pointgoal_wheeled.py baselines/navdp
PASS

cmake --build minco_processor/build -j2
minco_processor / minco_processor_compile_test / minco_processor_py — Built (未执行测试二进制或优化器)

MPLCONFIGDIR=/tmp/matplotlib conda run -n navdp bash scripts/run_all_experiments.sh experiments/configs/mock_full_suite.json --backend mock --resume
completed=0, skipped=6, failed=0

conda run -n navdp bash scripts/run_all_experiments.sh experiments/configs/static_real_suite.json --backend isaac --dry-run
run_count=6, started_processes=0, failed=0

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

总体结论：两阶段要求的代码、配置、测试、mock 结果、报告与审计文档已交付；静态与模拟门禁全部通过，真实运行边界已明确标注。
