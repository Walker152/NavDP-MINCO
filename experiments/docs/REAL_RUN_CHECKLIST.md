# 真实 GPU Smoke 检查清单

> 本清单仅供获得明确授权后的真实环境验证；本次交付未执行以下步骤。

## 前置条件

- [ ] GPU、显存和磁盘空间满足 Isaac Lab 与 NavDP checkpoint 要求。
- [ ] `conda activate navdp` 可导入 NavDP 依赖并找到 checkpoint。
- [ ] `conda activate isaaclab` 可启动 Isaac Lab，且相机扩展可用。
- [ ] 使用独立结果目录，确认 `/results/` 不进入 Git。
- [ ] 首次 smoke 仅 1 scene、1 episode、`num_envs=1`，保存运行配置与日志。

## 分层验证

- [ ] RAW：确认 `/health`、固定 seed、Top-1 坐标变换、MPC 输出及 episode 完成。
- [ ] MINCO-COLD：确认 ESDF、Python/C++ 校验、轨迹发布、失败回退和 trace 字段。
- [ ] MINCO-HOT：确认首周期 COLD，后续满足门控时 HOT；preview 不污染 history，只有已发布轨迹 commit。
- [ ] 人工制造 stale/unsafe/optimizer failure，确认 HOLD/STOP 且不写入伪 plan。
- [ ] 检查 headless MP4 可解码、帧数/FPS/episode UID metadata 完整。
- [ ] 检查 Ctrl-C、超时及异常退出后 NavDP/Isaac 子进程均被清理。

## 配对与结果验收

- [ ] RAW/COLD/HOT 使用相同 scene、episode UID、seed、初始状态及安全配置。
- [ ] `artifact_manifest.json` 校验通过，CSV/NPZ/video/report 无缺失或重复 episode。
- [ ] 报告明确区分真实结果与 mock 结果，并记录硬件、checkpoint、Git commit 和环境版本。
- [ ] 先审阅 smoke 结果，再决定是否扩大 episode 数；不得直接启动全量仿真。
