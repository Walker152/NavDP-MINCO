# AutoDL 自检修复脚本设计

## 目标

新增一个可重复、幂等、默认执行安全修复的一键脚本，用于在 AutoDL 容器中快速诊断并修复 NavDP、IsaacLab、CUDA、Vulkan、运行时代码和实验启动链路中已经确认的问题。脚本必须在不启动真实实验的前提下给出明确的最终 PASS/FAIL 结论和可审计报告。

## 非目标与安全边界

- 不安装、卸载或清理宿主机 NVIDIA 驱动包。
- 不删除 `/etc/vulkan/icd.d` 或 `/usr/share/vulkan/icd.d` 中的 ICD 文件。
- 不覆盖仓库运行时代码，不自动解压更新包。
- 不删除实验结果、checkpoint、场景资源或日志。
- 不启动带 `--allow-real-simulation` 的真实评测。
- 默认直接终止严格匹配本仓库的残留 Isaac、Kit、eval、NavDP server 和相关 conda wrapper 进程，但必须排除脚本自身、祖先进程和不含仓库/IsaacLab 路径的同名系统进程。

## 方案

采用独立纯 Bash 入口：

```text
scripts/autodl_self_check_repair.sh
```

原因：

- Conda 或 Python 环境损坏时仍可运行前置诊断。
- 与现有 `scripts/setup_autodl.sh` 风格一致。
- 可通过 PATH 注入伪命令进行无 GPU 集成测试。
- JSON 和 Python 语义检查通过选定 Conda 环境内的 Python 完成，无需额外生产 helper。

新增对应集成测试：

```text
tests/scripts/test_autodl_self_check_repair.sh
```

README 增加命令入口和退出码说明。

## 命令行接口

默认调用：

```bash
bash scripts/autodl_self_check_repair.sh
```

默认执行可逆修复、残留进程清理、CUDA 检查、Isaac smoke、实验 dry-run 和报告生成。

支持参数：

- `--check-only`：只检查，不写运行环境文件、不修改 shell 启动文件、不终止进程。
- `--skip-smoke`：跳过 Isaac headless smoke。
- `--skip-dry-run`：跳过实验命令 dry-run。
- `--config PATH`：指定 suite 配置，默认 `configs/experiments/full_suite.json`。
- `--report-dir PATH`：指定报告目录，默认 `$REPO_ROOT/results/autodl_self_check`。
- `--smoke-timeout SECONDS`：设置 Isaac smoke 超时，默认 180 秒。
- `-h`、`--help`：显示帮助。

保留 `--kill-stale` 作为兼容性显式开关，但默认模式已经执行同样的残留进程清理；在 `--check-only` 下该开关无效且报告为只读。

环境变量：

- `AUTODL_WORK_DIR`：默认 `/root/autodl-tmp/navdp`，不可用时退回 `$HOME/.navdp-autodl`。
- `CONDA_BIN`：默认按 `/root/miniconda3/bin/conda`、PATH 中的 `conda` 顺序发现。
- `NAVDP_ENV_NAME`：默认 `navdp`。
- `ISAACLAB_ENV_NAME`：默认 `isaaclab`。
- `ISAACLAB_DIR`：默认 `$AUTODL_WORK_DIR/IsaacLab`。
- `CONDA_ENVS_PATH`：默认 `$AUTODL_WORK_DIR/conda/envs`。
- `NAVDP_RUNTIME_ENV_FILE`：默认 `$HOME/.config/navdp/autodl-runtime.env`。
- `NAVDP_SKIP_BASHRC_UPDATE=1`：允许写运行环境文件但不更新 `~/.bashrc`。

## 执行流程

### 1. 基础前置检查

检查 Linux、仓库根目录、可写报告目录以及以下命令：

- `bash`
- `awk`
- `grep`
- `sed`
- `find`
- `ps`
- `pgrep`
- `nvidia-smi`
- `vulkaninfo`
- `timeout`
- `sha256sum`
- Conda

检查 NavDP 和 IsaacLab Conda 环境、`isaaclab.sh`、`create_empty.py`、suite 配置、`eval_pointgoal_wheeled.py` 和实验后端文件。

缺少安装级组件时给出指向 `scripts/setup_autodl.sh` 的修复命令，不尝试隐式重装。

### 2. 残留进程识别与清理

从 `ps -eo pid=,ppid=,pgid=,sid=,etimes=,args=` 读取进程快照。候选命令必须匹配至少一个运行入口：

- `eval_pointgoal_wheeled.py`
- `omni.kit`
- `isaac-sim`
- IsaacLab 路径下的 Python/Kit
- `baselines/navdp/navdp_server.py`
- 包含上述入口的 `conda run`

并且命令行必须包含规范化后的 `REPO_ROOT` 或 `ISAACLAB_DIR`。以下 PID 必须排除：

- 当前脚本 PID `$$`
- 当前脚本父 PID 链
- PID 1
- 不含目标路径的其他用户进程

默认模式对候选进程按 PGID 去重，先发送 `SIGTERM`，等待最多 5 秒，再对仍存活 PID 发送 `SIGKILL`。仅向其成员全部属于候选集合的进程组发送组信号；否则逐 PID 发送，避免影响共享 shell。

`--check-only` 只记录候选，不发送信号。

### 3. NVIDIA 与 Vulkan ICD 自检修复

首先记录：

- `CUDA_VISIBLE_DEVICES`
- `NVIDIA_VISIBLE_DEVICES`
- `NVIDIA_DRIVER_CAPABILITIES`
- `nvidia-smi` 的 GPU 名称、驱动、显存、利用率和计算进程
- 原始 `vulkaninfo --summary`

从以下目录发现 NVIDIA ICD JSON：

- `/etc/vulkan/icd.d`
- `/usr/share/vulkan/icd.d`

对每个候选分别设置：

```bash
VK_ICD_FILENAMES=<candidate>
VK_DRIVER_FILES=<candidate>
```

运行限定超时的 `vulkaninfo --summary`。合格候选必须：

- 命令成功；
- 恰好枚举一个 NVIDIA 物理设备；
- 不重复枚举同一 NVIDIA `deviceName`；
- 不把 llvmpipe 当成选中设备。

优先选择 `/etc/vulkan/icd.d` 中的合格候选，其次选择 `/usr/share/vulkan/icd.d`。如果没有合格候选，脚本失败并保留全部探测输出。

默认模式原子写入：

```text
$NAVDP_RUNTIME_ENV_FILE
```

文件导出 `VK_ICD_FILENAMES`、`VK_DRIVER_FILES`、`CONDA_ENVS_PATH` 和已解析的运行目录。脚本使用带起止标记的幂等区块让 `~/.bashrc` source 该文件；修改前创建一次备份。`NAVDP_SKIP_BASHRC_UPDATE=1` 时跳过 bashrc。

脚本自身立即导出同样变量，使后续 Conda、Isaac 和 dry-run 检查使用修复后的环境。

### 4. CUDA 与 Conda 环境检查

在 `isaaclab` 环境运行 Python，断言：

- `torch.cuda.is_available()` 为真；
- `torch.cuda.device_count()` 至少为 1；
- device 0 名称包含 `NVIDIA`；
- 能创建 CUDA tensor 并完成同步。

在 `navdp` 环境检查 `torch`、`flask`、`diffusers` 可导入。检查失败时保留 Python 输出并给出对应环境修复提示。

### 5. 运行时代码一致性

静态检查 eval parser 和 `IsaacNavDPBackend.build_command()` 的关键契约：

- `minco_start_validation_exemption_radius`
- `minco_penalty_weight_attractor`
- `navdp_seeds`
- `raw_controller`
- `experiment_variant`

在 `isaaclab` 环境调用 eval 的 `--help`，确认 parser 在创建 `AppLauncher` 前正常接受帮助路径，并且帮助文本包含关键参数。

脚本不覆盖代码。发现不一致时报告“远端运行时版本混用”，打印当前文件 SHA-256，并指示重新同步仓库或受信任更新包。

### 6. Isaac headless smoke

运行：

```bash
conda run --no-capture-output -n isaaclab \
  bash "$ISAACLAB_DIR/isaaclab.sh" -p \
  "$ISAACLAB_DIR/source/standalone/tutorials/00_sim/create_empty.py" \
  --headless
```

输出写入独立日志。由于教程可能持续循环，以下条件判定成功：

- 日志出现 `app ready` 或 `Simulation App Startup Complete`；
- GPU 表中 RTX/NVIDIA 设备为 Active；
- 未出现致命模式：
  - `Multiple Installable Client Drivers`
  - `Failed to create any GPU`
  - `GPU Foundation is not initialized`
  - `no suitable CUDA GPU`
  - `Fatal`
  - `Segmentation`

检测到成功标记后主动终止 smoke 进程组并视为成功；超时且没有成功标记则失败。默认显示器缺失和 crash reporter 警告不作为失败。

### 7. 实验 dry-run 与参数契约

通过 NavDP 环境运行：

```bash
python -m experiments run-suite \
  --config "$CONFIG" \
  --backend isaac \
  --dry-run \
  --skip-video
```

定位 suite 对应的 `dry_run_plan.json` 并验证：

- `started_processes == 0`；
- `run_count` 大于 0，且与 commands/server_commands 数量一致；
- 每条 eval 命令包含 `conda run --no-capture-output`、`isaaclab.sh -p`；
- 关键参数和值是两个独立 argv 元素；
- 不存在 `attractor20.0` 一类参数粘连；
- `--navdp-seeds` 数量等于 `--num_episodes`；
- `--episode-uids` 数量等于 `--num_episodes`；
- RAW 使用 `--raw-controller original-navdp-mpc` 和 `--no-enable_minco`；
- MINCO 使用 `--raw-controller disabled` 和 `--enable_minco`；
- server/eval 端口一致。

dry-run 只生成配置和计划文件，不启动 NavDP 或 Isaac 进程。

### 8. 历史日志诊断

扫描结果目录中最新的：

- `isaac_eval.stderr.log`
- `isaac_eval.stdout.log`
- `run_status.json`

分类报告：

- 重复 Vulkan ICD；
- GPU Foundation 初始化失败；
- argparse 参数不兼容；
- CUDA OOM；
- segmentation/fatal；
- stale/failed/running run；
- 已达到 `app ready`、`FrameCheck`、`ControlRef` 或 `EpisodeDone`。

历史错误不单独导致当前检查失败；当前 Vulkan、CUDA、smoke、代码契约或 dry-run 失败才决定失败状态。对于历史 `FAILED` run，输出正式重试命令应包含 `--resume --retry-failed --allow-real-simulation`，但不自动执行。

### 9. 报告与退出码

每次运行创建：

```text
<report-dir>/<UTC timestamp>/
├── summary.txt
├── environment.txt
├── process-scan.txt
├── vulkan-before.txt
├── vulkan-probes/
├── torch-cuda.txt
├── runtime-contract.txt
├── isaac-smoke.log
├── experiment-dry-run.txt
└── historical-diagnostics.txt
```

终端每阶段输出 `[PASS]`、`[REPAIRED]`、`[WARN]` 或 `[FAIL]`。最终汇总包含检查数、修复数、警告数、失败数、选中的 ICD 和报告目录。

退出码：

- `0`：所有必需检查通过，允许存在非阻塞警告；
- `1`：一个或多个必需检查仍失败；
- `2`：命令行参数错误；
- `130`：用户中断。

## 测试设计

Shell 集成测试通过临时 PATH 注入伪命令，不依赖真实 GPU 或 Conda。至少覆盖：

1. `--help` 和未知参数。
2. 健康单 ICD 路径。
3. 原始 Vulkan 重复枚举同一 4090，脚本选择单一 `/etc` ICD。
4. `/etc` 候选失败，回退 `/usr/share`。
5. 无可用 NVIDIA ICD 时明确失败。
6. `--check-only` 不写文件、不改 bashrc、不发 kill。
7. 默认模式原子写环境文件，bashrc 区块不重复。
8. 默认模式清理严格匹配的残留进程，同时保留不相关进程。
9. PyTorch CUDA 不可用时失败。
10. Isaac smoke 出现 `app ready` 时通过并清理 smoke 进程。
11. Isaac smoke 出现 GPU Foundation 致命错误时失败。
12. eval/backend 参数缺失时报告版本混用。
13. dry-run argv 正确时通过。
14. seed 数量错误、参数粘连或 started_processes 非零时失败。
15. 历史错误被分类但不覆盖当前健康结论。
16. 重复执行不重复写 bashrc 管理区块。

测试还运行 `bash -n`；环境有 `shellcheck` 时运行 ShellCheck。

## 验收标准

- 在无 GPU 的测试环境中，全部伪工具集成测试通过。
- 在真实 AutoDL 环境中，能够复现并修复重复 NVIDIA ICD，限定后只枚举一个 RTX 4090。
- PyTorch CUDA 检查通过。
- Isaac smoke 到达 `app ready`，且没有 GPU Foundation 致命错误。
- dry-run 启动进程数为零且完整参数契约通过。
- 默认清理匹配的残留进程，不误杀不相关进程。
- 二次执行不产生重复配置或 bashrc 条目。
- 报告足以区分驱动、环境、代码版本、参数拼装、残留进程和历史失败。
