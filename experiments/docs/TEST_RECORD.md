# 测试记录

测试日期：2026-07-15（Asia/Shanghai）
工作目录：`/home/alioth/NavDP`  
原则：未启动 Isaac Sim / Isaac Lab / NavDP server，未进行 CUDA 推理。

## RED 记录

最初执行 `conda run -n navdp pytest -q experiments/tests`：失败，原因是 navdp 环境没有 pytest。没有安装依赖，测试改用标准库 unittest。

执行 `conda run -n navdp python -m unittest discover -s experiments/tests -v`：5 个模块因 `experiments.core/analyzers/designers/recorders/orchestrators` 尚不存在而失败，证明首批测试先于实现。配对分析测试也先因 `experiments.analyzers.paired` 不存在而失败。

## GREEN 记录

最终单元与 mock 集成测试：

```text
命令: conda run -n navdp python -m unittest discover -s experiments/tests -v
结果: Ran 71 tests in 40.308s — OK
```

mock 端到端首次运行：

```text
命令: conda run -n navdp python -m experiments run-suite --config experiments/configs/smoke_suite.json --backend mock --resume
结果: {"completed": 6, "skipped": 0, "failed": 0}
```

resume 再运行：

```text
结果: {"completed": 0, "skipped": 6, "failed": 0}
```

编译检查：

```text
命令: conda run -n navdp python -m compileall -q experiments
结果: exit 0
```

实际生成的 SPARSE RAW vs HOT 配对示例：共同 episode 1 个，三个示例 delta 均为 0；这是 deterministic mock 的预期，仅证明配对链路可运行。

重新审计补充：C++ `minco_processor`、compile test 目标和 Python 扩展完成编译但未执行；真实配置 dry-run 生成 6 个 eval 与 6 个 server 命令且 `started_processes=0`；RAW provenance 校验返回 `[]`，`navdp_raw/` 无 diff。

## 未执行

- `conda activate isaaclab` 及任何真实仿真；
- headless 摄像机和 MP4 完整性；
- NavDP 模型服务与推理；
- C++ MINCO 热启动 Preview/Commit 的运行时行为（仅编译与静态生命周期测试）；
- MPC 闭环与性能基准。

这些项目明确保留到真实后端接入后，在受控 GPU 机器上执行。
