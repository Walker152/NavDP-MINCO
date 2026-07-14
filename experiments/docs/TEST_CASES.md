# 测试案例

| ID | 层级 | 行为 | 验收 |
|---|---|---|---|
| TC-01 | Core | episode UID 与 variant/run 无关 | 相同场景起终点和 seed 得到相同 UID |
| TC-02 | Layout | 结果按实验/场景/方法/seed/run 分类 | 路径各层完整 |
| TC-03 | Metrics | 直线含重复点和 NaN | 长度正确、曲率接近 0、不崩溃 |
| TC-04 | Safety | ESDF 双线性查询与 OOB | 中心插值正确，OOB 计入 unsafe 分母 |
| TC-05 | Tracking | 点到折线距离 | 距离、段号和投影比例正确 |
| TC-06 | Temporal | MINCO 非均匀时间采样 | mean/RMS 使用时间权重 |
| TC-07 | Labels | OOB 和阈值边界 | OOB=RAW_UNSAFE，边界=HIGH_TURN |
| TC-08 | Writer | 异步 CSV 与 NPZ | close 后行数完整，NPZ 可读，无 tmp |
| TC-09 | Writer | schema 多余字段 | submit 立即抛 ValueError |
| TC-10 | Lifecycle | 非法状态跳转 | RUNNING 不能直接 COMPLETE |
| TC-11 | Manifest | 场景排序与 UID | scene_id 确定性排序，跨场景 UID 不同 |
| TC-12 | Suite | HOT 配 cold 模式 | 配置加载时拒绝 |
| TC-13 | Pairing | episode_uid 交集配对 | 仅共同 UID 进入 delta |
| TC-14 | Pairing | safe_dist 不一致 | 在统计前拒绝 |
| TC-15 | E2E | mock suite | 生成六张 CSV、trace、校验和报告 |
| TC-16 | Resume | 再次运行 COMPLETE suite | 重新校验后跳过所有 run |

真实仿真、视频、NavDP server、C++ Preview/Commit 和 MPC 只能在资源允许且明确授权的环境中另做测试，本地测试套件不包含这些入口。
