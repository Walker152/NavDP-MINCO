# Codex 修复任务：MINCO 候选轨迹逐级回退，避免 Top-1 失败直接停车

## 一、任务目标

当前 PointGoal 评估脚本默认：

```python
parser.add_argument("--minco_top_k", type=int, default=1)
```

而 `NavDPMincoAdapter.optimize_candidates()` 已经能够按照 NavDP critic 分数对候选轨迹排序，并逐条调用 MINCO。

现有默认配置只尝试 critic 排名第一的候选。只要该候选出现以下任一情况：

```text
invalid_candidate
MINCO optimizer failed
invalid_waypoints
PY_ESDF_UNSAFE
```

本周期就会返回 `MINCO_FAIL`。如果当前 episode 尚无有效缓存，则评估端进入：

```text
MINCO_STOP
```

机器人持续零动作，容易使整轮 episode 超时失败。

本阶段只修复**候选轨迹容错**：

```text
按 critic 排名尝试 Top-K
→ 前一候选失败时继续尝试下一候选
→ 找到第一条完整通过 MINCO 和 Python ESDF 验证的候选后立即采用
→ 仅当 Top-K 全部失败时才返回 MINCO_FAIL
```

目标是避免“Top-1 恰好不可优化”导致整轮停车，同时控制额外规划耗时。

---

## 二、修改范围

主要修改：

```text
eval_pointgoal_wheeled.py
utils_tasks/navdp_minco_adapter.py
```

不要修改：

```text
navdp_server.py
policy_agent.py
tracking_utils.py
differential_drive_utils.py
differential_controller.py
minco_pybind.cpp
minco_pipeline.cpp
sim_esdf_builder.py
esdf_query_utils.py
visualization_utils.py
```

---

## 三、严格约束

本轮不得修改：

1. episode reset 状态机、generation 检查和 HTTP 锁；
2. 视频写入逻辑；
3. MINCO 优化器内部参数和代价函数；
4. `safe_dist`、`max_vel`、`max_acc`、`max_iterations`；
5. C++ validation 和 Python ESDF 二次验证条件；
6. HOLD_LAST 与 STOP 的现有语义；
7. MPC 模型、权重、速度参考和控制约束；
8. 速度振荡问题；
9. NavDP 网络、critic 输出和候选生成方式；
10. success、SPL、distance 指标；
11. 不增加并行 MINCO processor；
12. 不引入候选间 hot start；
13. 不执行 Git 提交。

坚持最小改动。

---

## 四、当前问题定位

### 4.1 评估默认只尝试一个候选

将：

```python
parser.add_argument("--minco_top_k", type=int, default=1)
```

修改为：

```python
parser.add_argument("--minco_top_k", type=int, default=4)
```

命令行仍允许用户覆盖。

### 4.2 适配器当前会遍历 Top-K 后再按 objective 选最优

当前逻辑大致为：

```python
for selected_idx in order[:self.top_k]:
    ...
    if candidate success:
        if best is None or self._is_better(...):
            best = scored
```

这意味着当 `top_k=4` 时，即使第一条已经成功，仍会继续优化其余候选。单个 MINCO 已可能耗时几十到一百多毫秒，全部尝试会使正常规划周期耗时成倍增加。

本次改成**按 critic 排名的逐级失败回退**：

```text
rank 0 成功 → 立即采用并停止
rank 0 失败 → 尝试 rank 1
rank 1 成功 → 立即采用并停止
...
Top-K 全部失败 → 返回原有 MINCO_FAIL
```

不要在本阶段重新设计“critic + objective”的复合评分。

NavDP critic 排名负责候选优先级；MINCO 和 ESDF 负责可执行性筛选。

---

## 五、具体实现要求

### 5.1 规范化 top_k

在 `NavDPMincoAdapter.__init__()` 中，将：

```python
self.top_k = int(top_k)
```

改为：

```python
self.top_k = max(1, int(top_k))
```

避免误配置为 0 或负数时完全不尝试候选。

不要设定超过候选总数的强制上限；实际切片自然由候选数量限制。

### 5.2 保持 critic 排序方式不变

继续使用现有：

```python
order = self._candidate_order(
    critic_values[env_idx],
    len(candidates_world[env_idx]),
)
```

排序要求保持：

- critic 越大，优先级越高；
- 非有限 critic 值按最低优先级处理；
- 不改变 NavDP 输出数组的内容；
- 不使用 objective 预排序。

### 5.3 改为第一条完整有效候选立即成功

在候选循环中，只有候选依次通过以下全部检查后，才能停止：

```text
_as_guide_path() 有效
processor.optimize() 未抛异常
result["success"] 为 True
waypoints 维度和数量有效
Python ESDF 查询为有限值
py_min_esdf > safe_dist
```

构造 `scored` 后，直接：

```python
best = scored
break
```

删除或停止使用：

```python
if best is None or self._is_better(scored, best):
    best = scored
```

`_is_better()` 若只在这里使用，可以删除；若其他代码还使用，则保留但本流程不再调用。不要为了删除一个小函数扩大改动。

### 5.4 所有失败类型都进入下一候选

以下失败必须只记录原因，然后 `continue`：

```text
invalid_candidate
processor.optimize exception
result success=False
invalid_waypoints
PY_ESDF_UNSAFE
```

不得因为某一候选：

- validation 失败；
- ESDF 不安全；
- 数值异常；
- waypoint 退化；

直接结束整个环境的候选搜索。

### 5.5 invalid_candidate 也写入 candidate_timings

当前 `invalid_candidate` 只写入 `failures`，没有进入 `candidate_timings`。

为其补充统一记录，例如：

```python
candidate_timings.append({
    "candidate_rank": int(rank),
    "selected_index": int(selected_idx),
    "python_call_ms": 0.0,
    "cpp_optimize_time_ms": float("nan"),
    "success": False,
    "objective": float("inf"),
    "min_esdf": float("nan"),
    "failure_reason": "invalid_candidate",
})
```

其他候选记录也补充：

```python
"candidate_rank": int(rank)
```

候选循环建议改为：

```python
candidate_indices = order[:min(self.top_k, len(order))]
for rank, selected_idx in enumerate(candidate_indices):
    ...
```

不要新增复杂统计类。

### 5.6 结果中增加最小必要诊断字段

成功和失败结果均增加：

```python
"configured_top_k": int(self.top_k)
"attempted_candidate_count": int(len(candidate_timings))
"attempted_candidate_indices": [...]
"selected_candidate_rank": ...
```

成功时：

```python
selected_candidate_rank = 对应 rank
```

全失败时：

```python
selected_candidate_rank = -1
```

`attempted_candidate_indices` 从 `candidate_timings` 提取即可。

不要修改已有字段名称，确保评估脚本和 timing overlay 仍可读取：

```text
success
waypoints
samples
selected_index
objective
min_esdf
py_min_esdf
failure_reason
candidate_timings
```

### 5.7 日志应体现逐级回退

成功日志补充：

```text
selected_rank=<rank>
attempted=<count>/<configured_top_k>
```

示例：

```text
[NavDP-Minco] env=0 status=MINCO_OK selected_idx=5 selected_rank=2 attempted=3/4 ...
```

失败日志补充：

```text
attempted=<count>/<configured_top_k>
```

示例：

```text
[NavDP-Minco] env=0 status=MINCO_FAIL attempted=4/4 fallback_mode=HOLD_LAST_OR_STOP reason=...
```

每个候选失败不需要额外逐条 `print`，避免终端刷屏；完整原因保留在：

```python
candidate_timings
failure_reason
```

### 5.8 failure_reason 保留全部 Top-K 结果的可诊断性

当前失败结果只拼接：

```python
failures[-3:]
```

当 `top_k=4` 时可能丢掉第一条失败原因。

改为保留本轮所有实际尝试候选的简洁原因：

```python
reason = "; ".join(failures)
```

由于默认最多 4 条，不会造成不可接受的日志长度。

如果候选数量为 0，则仍使用：

```text
NO_VALID_CANDIDATE
```

### 5.9 HOLD_LAST 和 STOP 不变

适配器只负责：

```text
Top-K 至少一条成功 → success=True
Top-K 全失败       → success=False
```

评估脚本继续保持现有行为：

```text
存在有效且未过期 cache → MINCO_HOLD_LAST
无有效 cache            → MINCO_STOP
```

不得延长 cache 生命周期，不得跨 episode 复用 cache，不得用 raw NavDP 轨迹替代 MINCO 轨迹。

### 5.10 不跨候选复用优化状态

每个候选仍以当前机器人状态和对应 guide path 单独调用：

```python
self.processor.optimize(...)
```

不要新增：

- 上一候选的控制点作为下一候选初值；
- 上一候选的时间分配作为下一候选初值；
- processor 池；
- 多线程并行优化。

现阶段保持现有 COLD_START 行为。

---

## 六、适配器预期流程

修改后的单环境流程应为：

```text
critic 排序
    ↓
取前 min(top_k, candidate_count) 个索引
    ↓
rank 0:
    guide path 非法 → 记录并继续
    MINCO 失败      → 记录并继续
    waypoint 非法   → 记录并继续
    Python ESDF 不安全 → 记录并继续
    全部通过        → 立即返回该候选
    ↓
rank 1 / rank 2 / rank 3 同理
    ↓
全部失败
    ↓
返回 MINCO_FAIL
    ↓
评估端自行决定 HOLD_LAST 或 STOP
```

---

## 七、禁止的错误实现

不要采用以下方案：

### 7.1 不要无条件优化全部 Top-4

错误：

```python
for candidate in top4:
    optimize all
choose minimum objective
```

这会让正常规划耗时接近四倍。

### 7.2 不要只检查 candidate 格式后选第一条

MINCO success、waypoint 有效性和 Python ESDF 验证必须全部通过。

### 7.3 不要放松安全判定

禁止改成：

```python
py_min_esdf >= safe_dist - epsilon
```

保持现有严格条件：

```python
py_min_esdf > safe_dist
```

### 7.4 不要在全部失败后直接执行 raw NavDP

全部失败仍进入原有 HOLD_LAST 或 STOP。

### 7.5 不要修改 critic 分数

不得归一化、重标定或人为混入 MINCO objective。

---

## 八、验收要求

### 8.1 静态检查

执行：

```bash
python -m py_compile eval_pointgoal_wheeled.py
python -m py_compile utils_tasks/navdp_minco_adapter.py
```

### 8.2 最小逻辑验证

构造或临时 mock 以下情况，验证循环行为：

#### 情况 A：第一候选成功

```text
rank 0 success
```

要求：

```text
attempted_candidate_count=1
selected_candidate_rank=0
不调用 rank 1~3
```

#### 情况 B：第一候选 invalid，第二候选成功

要求：

```text
attempted_candidate_count=2
selected_candidate_rank=1
selected_index 为第二候选原始索引
```

#### 情况 C：前两候选 validation 失败，第三候选成功

要求：

```text
attempted_candidate_count=3
selected_candidate_rank=2
返回 success=True
```

#### 情况 D：Top-4 全部失败

要求：

```text
success=False
selected_candidate_rank=-1
attempted_candidate_count=4
failure_reason 包含四条候选的失败摘要
fallback_mode=HOLD_LAST_OR_STOP
```

#### 情况 E：只有两个候选

即使配置：

```text
top_k=4
```

也只能尝试两条，不能越界。

#### 情况 F：top_k 配置为 0

适配器内部应规范化为 1。

### 8.3 仿真日志验证

使用：

```bash
python eval_pointgoal_wheeled.py \
  --port 8888 \
  --scene_dir /home/alioth/NavDP/assets/scenes/cluttered_easy \
  --scene_index 0 \
  --scene_scale 1.0 \
  --num_envs 1 \
  --num_episodes 10 \
  --enable_minco \
  --minco_top_k 4 \
  --minco_safe_dist 0.60 \
  --minco_sample_dt 0.05 \
  --minco_max_vel 1.0 \
  --minco_max_acc 2.0 \
  --minco_max_iterations 64
```

重点确认：

1. 第一候选成功时，`attempted=1/4`，正常周期耗时不会乘以四；
2. 第一候选失败、后续候选成功时，出现 `selected_rank>0`；
3. 只有实际尝试的 Top-K 全失败才出现 `MINCO_FAIL`；
4. `MINCO_STOP` 次数应低于 Top-1 版本；
5. HOLD_LAST 逻辑未改变；
6. 轨迹仍通过 C++ validation 和 Python ESDF 双重检查；
7. 不出现候选索引越界；
8. reset、视频、MPC 和指标行为无变化。

---

## 九、完成后报告

修改完成后只报告：

1. 修改了哪些文件；
2. 默认 Top-K 从多少改为多少；
3. 是否改成“第一条完整有效候选立即采用”；
4. 新增了哪些诊断字段；
5. 静态检查结果；
6. 是否进行了 mock 或仿真验证；
7. 仿真中出现过多少次 `selected_rank>0`；
8. Top-K 全失败次数与 `MINCO_STOP` 次数。

不要输出大段计划复述，不要提交代码。
