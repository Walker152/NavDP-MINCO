# AutoDL 严格历史配置设计

## 背景

现有 `scripts/autodl_self_check_repair.sh` 在检测不到
`/root/autodl-tmp` 时会退回 `$HOME/.navdp-autodl`。这个回退便于本机验证，
但可能让本机 IsaacLab、Conda 和 GPU 检查结果被误认为目标 AutoDL 容器的
状态。

本设计以 2026-07-17 留存的 AutoDL 运行记录为唯一生产配置依据：

- 仓库：`/root/NavDP`
- 大文件工作目录：`/root/autodl-tmp/navdp`
- IsaacLab：`/root/autodl-tmp/navdp/IsaacLab`
- Conda：`/root/miniconda3/bin/conda`
- Conda environments：`/root/autodl-tmp/navdp/conda/envs`
- 目标运行记录中的 GPU：NVIDIA GeForce RTX 4090
- 目标运行记录中的驱动：560.35.03

GPU 型号和驱动版本属于诊断证据，不作为可写设置，也不作为永久硬编码的
版本锁。脚本必须读取并记录当前 AutoDL 容器的实际值。

## 目标

生产执行默认采用严格 AutoDL 配置，不再根据执行机器的 `$HOME`、本机
IsaacLab 或本机 Conda 安装推导目标路径。路径不满足历史配置时立即失败，
并明确报告当前路径与期望路径；不得把本机成功结果作为 AutoDL 通过结论。

## 严格生产配置

非测试模式下，脚本固定使用：

```text
REPO_ROOT=/root/NavDP
AUTODL_WORK_DIR=/root/autodl-tmp/navdp
ISAACLAB_DIR=/root/autodl-tmp/navdp/IsaacLab
CONDA_BIN=/root/miniconda3/bin/conda
CONDA_ENVS_PATH=/root/autodl-tmp/navdp/conda/envs
NAVDP_RUNTIME_ENV_FILE=/root/.config/navdp/autodl-runtime.env
BASHRC=/root/.bashrc
```

脚本文件通过真实路径解析后的仓库根必须等于 `/root/NavDP`。生产模式不接受
环境变量把上述路径改为其他机器路径，也不提供隐式 `$HOME/.navdp-autodl`
回退。若需要检查不同 AutoDL 镜像，应先形成新的明确配置规格，而不是复用
本机探测结果。

`AUTODL_REPAIR_TESTING=1` 是唯一例外。测试模式继续允许夹具仓库、临时 HOME、
伪 Conda 和临时 IsaacLab 路径，以便无 GPU 集成测试覆盖生产控制流。脚本
必须拒绝普通生产会话设置 `AUTODL_REPAIR_TESTING=1` 后调用真实信号后端；
现有测试模式安全约束保持不变。

## 环境修复边界

脚本可以写入并持久化：

```bash
export VK_ICD_FILENAMES=<逐项探测通过的单一 NVIDIA ICD>
export VK_DRIVER_FILES=<同一 NVIDIA ICD>
export CONDA_ENVS_PATH=/root/autodl-tmp/navdp/conda/envs
export AUTODL_WORK_DIR=/root/autodl-tmp/navdp
export ISAACLAB_DIR=/root/autodl-tmp/navdp/IsaacLab
export NAVDP_REPO_ROOT=/root/NavDP
export CONDA_BIN=/root/miniconda3/bin/conda
```

NVIDIA ICD 路径不能根据本机结果写死。脚本仍依次探测
`/etc/vulkan/icd.d` 和 `/usr/share/vulkan/icd.d` 中的 NVIDIA manifest，
只接受能成功枚举恰好一个 NVIDIA 物理设备且不包含 llvmpipe 的候选。

以下变量不由容器内脚本自动赋值：

- `CUDA_VISIBLE_DEVICES`
- `NVIDIA_VISIBLE_DEVICES`
- `NVIDIA_DRIVER_CAPABILITIES`
- `CUDA_HOME`

历史记录证明前三项未设置时，`nvidia-smi`、PyTorch CUDA 和限定单 ICD 后的
Isaac smoke 均可成功。尤其是 `NVIDIA_VISIBLE_DEVICES` 与
`NVIDIA_DRIVER_CAPABILITIES` 属于容器启动阶段配置；在容器内补写无法补挂载
缺失的设备或驱动能力。脚本只记录这些变量，并以实际 CUDA、Vulkan 和 Isaac
smoke 结果判定健康状态。

## 历史故障对应关系

脚本必须继续覆盖以下历史故障，不把它们混同为 GPU 资源不足：

1. 原始 `vulkaninfo` 同时枚举重复 RTX 4090 和 llvmpipe：通过单 ICD 探测与
   `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` 修复。
2. `Multiple Installable Client Drivers`、`Failed to create any GPU`、
   `GPU Foundation is not initialized` 和 `no suitable CUDA GPU`：由 Isaac
   smoke 致命模式检测。
3. `--minco_start_validation_exemption_radius` 不被远端 parser 接受：判定为
   `/root/NavDP` 运行时代码版本混用，脚本不得覆盖源码。
4. `--minco_penalty_weight_attractor20.0` 参数粘连：由 dry-run argv 契约拒绝。
5. `--navdp-seeds` 数量少于 `--num_episodes`：由 dry-run seed 数量契约拒绝。
6. PID 1 接管的遗留 Isaac/NavDP 进程：继续按现有严格身份、年龄和进程树规则
   清理，不扩大匹配范围。
7. `OmniGraphSettings::getCudaDeviceOrdinal` 回退到 GPU0：若 GPU 表显示唯一
   NVIDIA Active 设备、smoke 到达 `app ready` 且无致命 GPU 模式，则分类为
   非阻塞警告；否则随当前 smoke 失败。

## 输出与结论

`environment.txt` 和 `summary.txt` 必须增加：

- `Profile: autodl-strict-history`
- 每个固定生产路径的期望值和实际值
- 当前 NVIDIA GPU、驱动和选中的 ICD
- 非 AutoDL 路径拒绝原因

只有严格 AutoDL 前置检查、CUDA、Vulkan、Isaac smoke、运行时代码契约和
dry-run 全部通过，才能输出 AutoDL PASS。本机或测试夹具通过时，输出必须
明确标记 `testing`，不能声称目标 AutoDL 已通过。

## 测试

在现有 Shell 集成测试中新增失败优先的回归场景：

1. 非测试模式从非 `/root/NavDP` 仓库执行时，前置检查失败且不探测本机
   IsaacLab。
2. 非测试模式即使设置本机 `AUTODL_WORK_DIR`、`ISAACLAB_DIR` 或
   `CONDA_ENVS_PATH`，也拒绝路径覆盖。
3. 生产默认值固定为历史 AutoDL 路径，不再出现 `$HOME/.navdp-autodl`。
4. 测试模式仍能使用临时夹具路径并运行完整集成测试。
5. 生成的 runtime env 包含历史固定路径和 `CONDA_BIN`。
6. `CUDA_VISIBLE_DEVICES`、`NVIDIA_VISIBLE_DEVICES`、
   `NVIDIA_DRIVER_CAPABILITIES` 和 `CUDA_HOME` 不被脚本写入。
7. 测试模式 summary 明确标记 `testing`，生产 summary 标记
   `autodl-strict-history`。

全部既有进程安全、Vulkan、CUDA、source contract、smoke、dry-run、并发写入、
回滚和幂等测试必须继续通过。

## 验收标准

- 默认生产执行只针对历史 AutoDL 目录布局。
- 缺少任何固定安装组件时失败，不退回本机目录。
- 不用本机 GPU、IsaacLab 或 Conda 成功替代 AutoDL 验证。
- 运行环境文件只持久化容器内确实可修复的路径和单一 Vulkan ICD。
- 历史中的重复 ICD、代码版本混用、参数粘连、seed 数量和遗留进程均有明确
  独立诊断。
- 无 GPU 测试夹具仍可完整验证控制流，且不会执行真实信号或生产写入。
