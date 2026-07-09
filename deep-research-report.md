# NavDP 动力学轨迹平滑方案分析报告

## 执行摘要

结合 NavDP 仓库的静态分析、论文中的数据生成方式，以及你前面对“先做一层优化防护、后续再把运动学/动力学代价前置到训练”的思路，我的结论是：**一个月内最合理、风险最低、对现有系统侵入最小的方案，不是改训练，而是在 NavDP 已经输出 `all_trajectory` 和 `all_values` 之后、当前 `MPC_Controller` 之前，插入一个“意图保持的稀疏控制点动力学平滑器”**。这样既能复用仓库已经具备的扩散候选生成、critic 排序、HTTP 解耦接口和异步规划/跟随框架，又能显式补上当前 pipeline 中缺失的运行期曲率、角速度、加速度、最小安全间距与速度剖面约束。仓库当前确实已经具备：扩散式候选轨迹生成、goal-agnostic critic 做候选筛选、异步 MPC 跟随、差速轮速映射，但**没有**一个独立的运行时轨迹几何/动力学平滑层。citeturn36view0turn9view0turn9view1turn17view0turn13view3

更具体地说，当前 point-goal 路径链路是：`client_utils.py` 通过 HTTP 向 `navdp_server.py` 发送 RGB-D 和 goal，服务端调用 `policy_agent.py`，后者使用 `policy_network.py` 生成 `all_trajectory`、`all_values` 和 top-trajectory；`eval_pointgoal_wheeled.py` 再把最优轨迹变换到世界系，直接交给 `MPC_Controller` 追踪。这个 follower 使用的是固定时间步长的单轨/独轮模型 MPC，代价里只有位置跟踪项与控制输入项，没有障碍 clearance、曲率软约束、加速度/jerk 约束，也没有“意图保持”的后处理。citeturn16view0turn16view1turn34view0turn34view1turn17view0turn17view1turn13view4

因此，建议的主落地方向是：**对 top-K 候选轨迹做轻量级二阶段优化**。第一阶段在几何层面对稀疏控制点做 ESDF/局部距离场 + 意图吸引 + 曲率平滑优化；第二阶段在固定几何下做速度剖面整形，使其满足差速/类车底盘的角速度、横向加速度、线加速度与 jerk 约束。默认建议 K=2、控制点数 M=6 或 8、优化迭代 8–15 次、规划线程预算控制在 5–10 ms/条候选轨迹。这样既保留 NavDP 原有“端到端决策意图”，又把“动力学可行性与过渡安全性”前置到了执行前的最后一道防线。作为长期方向，则可以沿着你提出的路线，把曲率/控制努力/clearance 代价前置到专家轨迹构造与监督标签中，和论文中用 ESDF 对 A* 路径做 waypoint refine 再 cubic spline 平滑的做法是一致的。citeturn21view0turn35academia0

## 仓库静态分析

从目录层面看，和本方案最相关的模块集中在 `baselines/navdp`、`utils_tasks`、`wheeled_robots/controllers` 和 `configs/robots`。`baselines/navdp` 下包含 `navdp_server.py`、`policy_agent.py`、`policy_backbone.py`、`policy_network.py`；`utils_tasks/client_utils.py` 负责 HTTP 客户端调用；`utils_tasks/tracking_utils.py` 提供当前的 `PlanningInput/PlanningOutput` 数据结构和 `MPC_Controller`；`wheeled_robots/controllers/differential_controller.py` 负责把 \([v,\omega]\) 映射成左右轮角速度；`configs/robots/dingo_config.py` 给出了 Dingo 轮半径和轮距等现成参数。README 还明确说明该 benchmark 采用 **导航方法与评测解耦的 HTTP API**，并且实现了**轨迹规划与轨迹跟随完全异步**的框架，这正好为插入一个后端平滑器创造了天然接入点。citeturn24view0turn15view0turn15view1turn11view0turn20view0turn36view0

基于这些源码，当前最有价值、最建议直接复用的模块可以概括如下。下表是对仓库现状的工程化归纳，而不是新增设计。citeturn24view0turn15view0turn15view1turn11view0turn20view0

| 模块 | 当前职责 | 对平滑方案的复用价值 | 建议 |
|---|---|---|---|
| `policy_network.py` | 生成 16 条候选轨迹并用 critic 排序 | 直接提供 top-K 候选，不必重写 planner | **强复用** |
| `policy_agent.py` | 预处理 RGB-D、维护时序记忆、输出 top1 / all candidates | 可作为 smoother 的上游输入适配层 | **强复用** |
| `navdp_server.py` | 暴露 `/pointgoal_step` 等 HTTP 接口 | 无需改协议，只扩展返回或在客户端后处理 | **强复用** |
| `client_utils.py` | 发送请求并接收 `trajectory/all_trajectory/all_values` | 直接拿到 top-K 候选与分数 | **强复用** |
| `tracking_utils.py` 的 `PlanningInput/Output` | 异步规划线程共享数据 | 适合新增 `smoothed_trajectory_world` 等字段 | **强复用** |
| `tracking_utils.py` 的 `MPC_Controller` | 当前轨迹跟随器 | 可先保留，只替换其参考轨迹输入 | **中等复用** |
| `differential_controller.py` | \([v,\omega]\) 到轮速映射和限幅 | 用于把动力学约束收敛到执行器空间 | **强复用** |
| `dingo_config.py` | Dingo 轮半径、轮距、相机参数 | 可作为默认参数源 | **强复用** |

更关键的是数据流。`navigator_reset()` 向 `/navigator_reset` 发送相机内参、`stop_threshold` 和 batch size；`pointgoal_step()` 把 batched RGB、Depth 和 `{goal_x, goal_y}` 发给 `/pointgoal_step`，响应里直接返回三项：`trajectory`、`all_trajectory`、`all_values`。这意味着后处理器根本不需要侵入模型内部，只要在 client 侧或 planning thread 内部消费这三个量即可。citeturn15view0turn16view0turn16view1turn16view2turn34view0turn34view2

在模型内部，`NavDP_Agent` 初始化时设置 `image_size=224`、`memory_size=8`、`predict_size=24`；图像做 resize+pad 到 224，深度值被裁到 \(0.1\sim5.0\) m，point-goal 被剪裁到 \([-10,10]\) 且前向 \(x\) 被裁到 \([0,10]\)。时序上，agent 用 `memory_queue` 累积最近 8 帧图像，作为 RGB-D backbone 的时间上下文输入。对我们来说，这意味着 smoother 可以把 NavDP 的输出看成一个**24 点、约 2–3 s 局部时域的短时参考路径**，不建议在一个月方案里改这一时域配置。citeturn31view0turn30view3

候选轨迹生成与 critic 逻辑也比较清晰。`NavDP_Policy` 的 point-goal 分支把 point-goal 编码成一个线性 token，与 RGB-D backbone token 一起进入基于 DDPM scheduler 的扩散去噪过程；默认 `sample_num=16`，每次生成 16 条长度为 `predict_size=24` 的动作序列，随后通过 `torch.cumsum(naction / 4.0, dim=1)` 转成轨迹点列。critic 由 `predict_critic()` 完成，它对预测轨迹做 token embedding，再和 **goal-agnostic** 的 RGB-D 条件一起送入 decoder，最后输出 scalar critic 值；排序后取 top-2 为 `positive_trajectory`、bottom-2 为 `negative_trajectory`，而 `policy_agent.py` 最终只返回 `good_trajectory[:,0]` 作为执行轨迹。这个设计很适合我们做 **top-K 后优化**：模型已经替我们提供了多样性和初筛，我们不需要自己再做候选采样。citeturn29view2turn29view0turn9view0turn9view1turn6view3

从感知 backbone 看，仓库把 RGB 和 depth 都送入 `DepthAnythingV2`，再通过 transformer decoder 压成 memory token；image-goal 与 pixel-goal 也分别有专门 backbone。这个实现说明 NavDP 的本体是**局部视觉驱动的轨迹提议器**，不是一个显式地图规划器。因此“额外增加一个显式的局部距离场/ESDF 代价层”，不会和当前模型冲突，反而正好补足 runtime safety margin 的缺口。citeturn26view1turn26view2turn28view0

## 动力学平滑问题定义

当前 benchmark 里的跟随链路对动力学的处理相对粗糙：`planning_thread()` 得到 NavDP 的一条局部轨迹后，将其投到世界系，直接用 `MPC_Controller` 构造参考轨迹；该 controller 先线性 densify 参考路径，再在 \(N=15\)、\(T=0.1\) s 的独轮模型上求解控制，状态代价矩阵 \(Q=\mathrm{diag}(10,10,0)\)，控制代价 \(R=\mathrm{diag}(0.02,0.15)\)，硬约束只有 \(v\in[0,v_{\max}],\;\omega\in[-\omega_{\max},\omega_{\max}]\)。同时，差速控制器只做 \([v,\omega]\) 到左右轮速的解析映射和上下限裁剪。换言之，**当前系统已经有“跟随控制”，但没有“面向执行器的路径几何整形与速度剖面整形”**。citeturn17view0turn17view1turn13view3turn13view2turn13view4

因此，建议把“动力学平滑”严格定义为一个**局部、短时域、保意图的轨迹几何+时间参数化问题**。给定 NavDP 的原始离散轨迹 \(\hat{\mathbf p}_{0:N-1}\)，其中 \(N=24\)，构造一条由稀疏控制点参数化的平滑曲线 \(\mathbf p(s)\)，并求出对应的速度剖面 \(v(s)\)，使得它在尽量贴近原始意图的同时，满足底盘的非完整约束、执行器速度约束、曲率与加速度约束、以及最小 clearance 约束。这样做和论文中“先用 ESDF+A* 生成路径，再对 waypoint 做 obstacle-aware refine，最后 cubic spline 平滑”的专家轨迹生成方式是一致的，只是把那套思想从离线数据生成搬到了在线局部后处理。citeturn21view0

对**差速/独轮底盘**，建议采用如下离散模型作为约束基础，与仓库现有 `DifferentialController` 和 `MPC_Controller` 完全一致：

\[
x_{i+1}=x_i+v_i\cos\theta_i\Delta t,\qquad
y_{i+1}=y_i+v_i\sin\theta_i\Delta t
\]

\[
\theta_{i+1}=\theta_i+\omega_i\Delta t,\qquad
v_{i+1}=v_i+a_i\Delta t,\qquad
\omega_{i+1}=\omega_i+\alpha_i\Delta t
\]

并保留轮速映射：

\[
\omega_R=\frac{2v+\omega b}{2r},\qquad
\omega_L=\frac{2v-\omega b}{2r}
\]

其中 \(r\) 是轮半径，\(b\) 是轮距。仓库里的 Dingo 参数分别为 \(r=0.0591\) m、\(b=0.22616\) m，可直接作为 wheeled baseline 的默认实参。citeturn13view2turn20view0

对**类车/单轨底盘**，建议采用标准运动学 bicycle model：

\[
x_{i+1}=x_i+v_i\cos\theta_i\Delta t,\qquad
y_{i+1}=y_i+v_i\sin\theta_i\Delta t
\]

\[
\theta_{i+1}=\theta_i+\frac{v_i}{L}\tan\delta_i\Delta t,\qquad
v_{i+1}=v_i+a_i\Delta t,\qquad
\delta_{i+1}=\delta_i+u_{\delta,i}\Delta t
\]

其中 \(L\) 为轴距，\(\delta\) 为前轮转角，\(|\kappa|=\frac{|\tan\delta|}{L}\)。在自动驾驶和车辆轨迹规划文献里，使用 kinematic bicycle model 作为局部轨迹规划基础是标准做法。citeturn38academia1

在路径几何上，我建议把轨迹写成**稀疏控制点 B-spline 或 Catmull–Rom spline**，优化变量只放在中间控制点上，而不是直接优化全部 24 个 waypoint。设控制点集合为 \(\mathbf c_0,\dots,\mathbf c_{M-1}\)，其中 \(M=6\) 或 \(8\)，则评价时把它重采样成 \(N_e=32\) 或 \(40\) 个稠密点评估曲率、clearance 和速度剖面。这样做能显著减少变量维度，也符合“稀疏参数化、稠密约束评估”的经典工程路线。citeturn35academia0

建议采用如下总目标函数：

\[
J = 
w_{\text{intent}}J_{\text{intent}}+
w_{\text{cp}}J_{\text{cp}}+
w_{\text{clear}}J_{\text{clear}}+
w_{\kappa}J_{\kappa}+
w_{\Delta^2}J_{\Delta^2}+
w_{\Delta^3}J_{\Delta^3}+
w_{\text{end}}J_{\text{end}}+
w_vJ_v+
w_aJ_a+
w_jJ_j
\]

其中最重要的三项分别是：

\[
J_{\text{intent}}
=
\sum_i
\left(
w_{\text{lat}}\big((\mathbf p_i-\hat{\mathbf p}_i)^\top \hat{\mathbf n}_i\big)^2
+
w_{\text{lon}}\big((\mathbf p_i-\hat{\mathbf p}_i)^\top \hat{\mathbf t}_i\big)^2
\right),\quad
w_{\text{lat}} > w_{\text{lon}}
\]

这意味着对原轨迹的**横向偏离惩罚更重、纵向重参数化惩罚更轻**，从而更稳地保持 NavDP 已选中的“绕障侧”和局部 homotopy。

\[
J_{\text{cp}}=\sum_{j\in\mathcal A}\|\mathbf c_j-\hat{\mathbf p}_{a_j}\|^2
\]

这就是你最开始提到的“对路径施加控制点吸引代价”，可以作为最轻量版本。

\[
J_{\text{clear}}=\sum_i \left[\max(0,d_{\text{safe}}-d(\mathbf p_i))\right]^2
\]

其中 \(d(\cdot)\) 来自局部 ESDF/EDT；如果当前位置到障碍物的 clearance 小于安全间距 \(d_{\text{safe}}\)，代价迅速增加。这个项既和论文中用 ESDF refine waypoint 的做法一致，也最能直接解决“waypoint 本身不安全、序列中间过渡有风险”的问题。citeturn21view0

曲率和控制约束建议这样落地。先由稠密曲线求几何曲率：

\[
\kappa_i=
\frac{x'_i y''_i-y'_i x''_i}
{\left((x'_i)^2+(y'_i)^2\right)^{3/2}+\varepsilon}
\]

然后对差速底盘施加

\[
|\kappa_i|\le \kappa_{\max},\qquad
|\omega_i|=|v_i\kappa_i|\le \omega_{\max},\qquad
|a_{\text{lat},i}|=v_i^2|\kappa_i|\le a_{\text{lat},\max}
\]

对类车底盘施加

\[
|\kappa_i|\le \frac{\tan \delta_{\max}}{L},\qquad
|\dot{\delta}_i|\le \dot{\delta}_{\max},\qquad
|a_i|\le a_{\max}
\]

这样一来，曲率几何约束、速度约束和执行器约束就贯通起来了。对差速底盘，还可以进一步检查 \(|\omega_L|,|\omega_R|\le \omega_{\text{wheel},\max}\) 作为最终执行器可达性验收。citeturn13view2turn17view0turn38academia1

如果底盘参数**未指定**，我建议先采用下列默认范围；若使用仓库中的 Dingo，则直接读取 repo 参数。下表是推荐的工程默认值，而不是仓库现有设定。  

| 底盘 | 参数 | 未指定时推荐默认值范围 | 备注 |
|---|---|---:|---|
| 差速/独轮 | \(v_{\max}\) | 0.4–0.8 m/s | 室内服务机器人常用 |
| 差速/独轮 | \(\omega_{\max}\) | 0.6–1.2 rad/s | 若沿用现有 MPC，可先取 0.8–1.0 |
| 差速/独轮 | \(a_{\max}\) | 0.4–0.8 m/s² | 控制更平顺 |
| 差速/独轮 | \(\alpha_{\max}\) | 1.0–2.5 rad/s² | 对急转抑制明显 |
| 差速/独轮 | \(a_{\text{lat},\max}\) | 0.4–0.8 m/s² | 用于曲率限速 |
| 差速/独轮 | \(d_{\text{safe}}\) | footprint 外扩后再加 0.10–0.20 m | 与 clearance 代价配合 |
| 类车/单轨 | \(L\) | 0.25–1.20 m | 未指定时写明“未指定” |
| 类车/单轨 | \(\delta_{\max}\) | 0.35–0.60 rad | 决定 \(\kappa_{\max}\) |
| 类车/单轨 | \(\dot{\delta}_{\max}\) | 0.5–1.5 rad/s | 保障转向平滑 |
| 类车/单轨 | \(a_{\max}\) | 0.5–1.5 m/s² | 室内偏小，室外可放宽 |

权重建议则如下，按“先意图与安全，后舒适与速度”的优先级设置即可。这些是建议值，用来给原型提供一个稳妥初始点。  

| 代价项 | 符号 | 建议初值 | 说明 |
|---|---|---:|---|
| 意图保持 | \(w_{\text{intent}}\) | 10.0 | 主约束 |
| 控制点吸引 | \(w_{\text{cp}}\) | 3.0 | 轻量保护 |
| 安全间距 | \(w_{\text{clear}}\) | 20.0–40.0 | 窄通道建议更高 |
| 曲率软约束 | \(w_{\kappa}\) | 8.0 | 与速度剖面联动 |
| 二阶差分平滑 | \(w_{\Delta^2}\) | 2.0 | 抑制折线感 |
| 三阶差分平滑 | \(w_{\Delta^3}\) | 0.5 | 抑制 jerk |
| 终点保持 | \(w_{\text{end}}\) | 10.0 | 高于中间点 |
| 速度跟踪 | \(w_v\) | 1.0 | 不宜过高 |
| 加速度惩罚 | \(w_a\) | 0.2 | timing-stage |
| jerk 惩罚 | \(w_j\) | 0.05 | timing-stage |

## 稀疏控制点优化器实现

我建议采用**两阶段实现**，而不是一次性做全状态联合 NLP。原因很简单：你现在要的是“一月可落地”，不是“最优但复杂”的学术系统。两阶段版本更稳、调参更容易、也更符合现有 NavDP + MPC 的模块边界。第一阶段只优化几何控制点；第二阶段在固定几何下做速度剖面整形。这样把变量维度从“整条时序轨迹的全部状态+控制”降到了“6–8 个二维控制点 + 32 个标量速度”，对实时性非常友好。稀疏参数化、稠密约束评估本身也是经典高效轨迹优化框架的通用思想。citeturn35academia0

变量建议设置为：

\[
\mathcal X_{\text{geom}}=\{\Delta \mathbf c_1,\dots,\Delta \mathbf c_{M-2}\},\qquad
\mathbf c_0=\mathbf p_{\text{robot}},\quad
\mathbf c_{M-1}=\hat{\mathbf p}_{N-1}\text{ 或软约束}
\]

也就是**只优化中间控制点偏移量**。其中起点固定为当前机器人投影点，终点建议在第一版里用高权重软约束，而不是死固定。死固定虽然简单，但在局部极窄通道里容易让优化器失去可行域弹性；高权重软约束更稳。采样控制点时不要按索引等间隔，而要按**弧长等间隔**从 NavDP 原始轨迹中取点，这样不会在前半段过密、后半段过疏。citeturn9view1turn31view0

初始化建议遵循“尽量不改变 NavDP 意图”的原则。最稳的初始化是：先把 top-K 中每条候选轨迹做一次弧长重参数化；再选 \(M\) 个控制点；然后在每个控制点周围做一次**很小尺度的 ESDF 梯度上升预偏移**，例如最多 5 cm–10 cm，只用于把显然贴边的控制点轻轻推离障碍；最后再进入主优化。这个预偏移能明显改善优化初值，特别是在原始 waypoint 已经接近障碍边界时。论文的数据生成阶段本来就在做“greedy search 以增大离障碍物距离”的 refine，所以这个 runtime 预处理是合理的。citeturn21view0

优化器选择上，原型阶段建议优先用 **SLSQP 或 L-BFGS-B**。如果约束大多通过 soft penalty 写进目标函数，那么 L-BFGS-B 就够了，速度通常更快；如果你希望把少量硬约束（比如起点固定、控制点边界、终点落在局部 corridor 内）显式写出来，则用 SLSQP 更直接。因为变量量级很小，我不建议第一版直接上 CasADi 全量重写；CasADi 更适合第二阶段你要把当前 `MPC_Controller` 一并升级时再统一。若你希望尽可能贴现有 PyTorch 依赖，也可以用 `torch.autograd + torch.optim.LBFGS` 做纯张量版实现。citeturn17view0turn35academia0

迭代数建议非常克制。默认设置可以是：**每条候选 8 次主迭代，必要时最多 15 次**；一旦连续两次目标改善小于阈值，就提前退出。经验上，对于 \(M=6\)、\(N_e=32\) 的情形，这样已经足够把明显的折线尖角、贴障风险和不合理曲率拉回来。考虑到仓库已有异步 planning thread，建议把后处理预算控制在 **5–10 ms/trajectory**；若 K=2，则整个平滑预算目标定在 10–20 ms 内，避免把规划线程频率拖得太低。citeturn14view3turn36view0

速度剖面的实现应该比几何优化更简单。固定一条平滑曲线后，沿弧长计算 \(\kappa_i\)，再构造曲率限速和 clearance 限速：

\[
v_{\kappa,i}=\sqrt{\frac{a_{\text{lat},\max}}{|\kappa_i|+\varepsilon}},\qquad
v_{\omega,i}=\frac{\omega_{\max}}{|\kappa_i|+\varepsilon}
\]

\[
v_{\text{clear},i}=v_{\min} + k_d \cdot \max(0, d(\mathbf p_i)-d_{\text{safe}})
\]

然后取

\[
v^{\text{raw}}_i=\min(v_{\max},v_{\kappa,i},v_{\omega,i},v_{\text{clear},i})
\]

再做一次前向/后向 pass，分别处理加速与减速约束：

\[
v_i \leftarrow \min\left(v_i,\sqrt{v_{i-1}^2+2a_{\max}\Delta s_i}\right)
\]

\[
v_i \leftarrow \min\left(v_i,\sqrt{v_{i+1}^2+2a_{\text{dec},\max}\Delta s_i}\right)
\]

这样就能在几乎不增加计算量的情况下得到一个足够平顺的局部速度参考。若当前 `MPC_Controller` 暂时不支持显式速度参考，可以先把 `desired_v` 设置为参考速度前几步的均值或最小值，作为过渡版本。citeturn17view0turn13view4

失败回退策略必须明确，否则实时系统就会在极端场景里不稳定。建议采用四级回退：第一，若优化正常收敛且 clearance 过检，就输出平滑轨迹；第二，若优化未收敛但原始 top-1 满足基本 clearance 和曲率阈值，就回退到 NavDP 原始 top-1；第三，若 top-1 不可行但 top-K 中存在可行候选，则切换到可行候选；第四，若所有候选都失败，则保持当前仓库逻辑，减速或零速。仓库评测脚本本来就有 “No trajectory available, using zero action” 的保底分支，所以这条回退链和现有机制是兼容的。citeturn13view5

## 与现有 NavDP pipeline 集成

接入点非常明确：**就在 `eval_pointgoal_wheeled.py` 的 `planning_thread()` 里，`pointgoal_step()` 返回候选之后、实例化 `MPC_Controller` 之前**。当前代码已经拿到了 `trajectory_points_camera`、`all_trajectories_camera`、`all_values_camera`，并把最优轨迹逐点变换到世界系；你要做的只是同步把 top-K 的候选也变换到世界系，然后把它们交给 `DynSmoothOptimizer`，最后把输出的 `smoothed_trajectory_world` 传入 `MPC_Controller`。这比改 server、更改训练代码或修改 NavDP 模型结构，都要轻量得多。citeturn14view3turn13view4turn16view0turn16view1

若按仓库已有数据结构改造，建议沿用 `PlanningInput` 和 `PlanningOutput`，只在 `PlanningOutput` 增加少数字段，例如：`smoothed_trajectory_world`、`selected_candidate_idx`、`optimizer_cost`、`optimizer_status`、`clearance_min`。这样不会破坏异步线程的结构，也利于可视化和离线诊断。`PlanningOutput` 当前已经保存 `trajectory_points_world`、`all_trajectories_world`、`all_values_camera` 和 planning 状态，这正是天然可扩展的地方。citeturn17view0

推荐的集成流程如下：

```mermaid
flowchart LR
    A[RGB-D + Goal] --> B[pointgoal_step]
    B --> C[NavDP all_trajectory + all_values]
    C --> D[取 top-K 候选]
    D --> E[相机系转世界系]
    E --> F[局部距离场/ESDF 构建]
    F --> G[稀疏控制点几何优化]
    G --> H[速度剖面整形]
    H --> I[可行性检查与回退]
    I --> J[MPC_Controller]
    J --> K[DifferentialController]
    K --> L[左右轮速度]
```

top-K 策略不建议一步到位做太大。仓库当前默认 `sample_num=16`，但实际返回给上层执行的是 top-1，内部保留了 top-2 positive trajectory。工程上最划算的方案通常是“**从 16 条生成候选里先按 critic 取 top-2，再做后优化重排**”。这样能显著减少“critic 排名略有误差导致的执行失败”，但不会把优化计算量拉爆。citeturn8view1turn9view1turn6view3

下面这个 top-K 选择表是我的建议值。  

| 策略 | 说明 | 计算开销 | 风险 | 结论 |
|---|---|---:|---|---|
| K=1 | 只优化当前 top-1 | 最低 | critic 误排时无纠错能力 | 不推荐作为默认 |
| K=2 | 优化 top-2，按后验可行性+critic 复排 | 低 | 很少错过次优可行路 | **推荐默认** |
| K=4 | 优化 top-4 | 中 | 稳健但更耗时 | 复杂场景可开 |
| K=8 | 大范围重排 | 高 | 实时性风险大 | 一月方案不建议 |

候选的最终打分建议不要只看 critic，也不要只看优化后代价。更合理的是：

\[
S_k = \lambda_c \,\widetilde{V}_k - \lambda_o J_k^{\text{post}} - \lambda_f \mathbb{I}[\text{infeasible}]
\]

其中 \(\widetilde{V}_k\) 是归一化 critic 值，\(J_k^{\text{post}}\) 是后优化总代价。这样可以保留 NavDP 学到的语义/避障意图，又把显式动力学可行性纳入最终排序。

接口层面，建议尽量**不改 HTTP 协议**。因为 `client_utils.py` 已经返回 `trajectory, all_trajectory, all_value`，足够完成 top-K 平滑。第一版可以完全在 `planning_thread()` 内部处理；如果后面需要把后处理下沉到 server 侧，再考虑为 JSON 增加可选字段 `smoothed_trajectory` 与 `selected_candidate_idx`。这符合 README 中“框架与评测解耦”的设计原则。citeturn16view1turn36view0

## 实验验证计划与指标

验证建议分三层推进：**仿真、rosbag、实物**。仿真层直接复用仓库已有 IsaacSim + IsaacLab benchmark；rosbag 层重点验证深度输入波动、局部距离场构造和 planner 实时性；实物层只做最后一周的小范围走廊、桌椅、窄门、行人干扰验证。README 明确说明 benchmark 已支持 point-goal/image-goal/no-goal，且使用了高保真 IsaacSim/IsaacLab 评测框架；现有 `eval_pointgoal_wheeled.py` 也已经会输出 success、SPL 和 distance。也就是说，性能指标基线仓库已经准备好了。citeturn36view0turn14view4

我建议的消融组应至少包含以下六组，才能把“平滑是否真的有效”说清楚。  
第一组，**Baseline**：原始 NavDP + 当前 MPC。  
第二组，**K1-Geom**：只对 top-1 做几何平滑，不做速度剖面。  
第三组，**K1-Geom+Time**：top-1 做完整两阶段平滑。  
第四组，**K2-Geom+Time**：top-2 平滑后重排。  
第五组，**K2-Geom+Time-NoClear**：去掉 ESDF/clearance 项，验证安全间距项的贡献。  
第六组，**K2-Geom+Time-NoIntent**：去掉意图保持项，验证是否会更容易换边或丢失 NavDP 语义决策。

指标要分成三类。**性能类**沿用 success、SPL、distance，并补充 time-to-goal；**安全类**增加最小 clearance、碰撞率、贴障时间占比、曲率超限率、轮速超限率；**实时类**记录 pointgoal_step 延迟、平滑器耗时、MPC 求解耗时、整条 planning thread 周期的 p50/p95/p99。当前脚本已经打印 MPC 求解耗时并且在 episode 结束时记录 success、SPL 和 distance，这些可以继续复用。citeturn13view4turn14view4

测试用例建议不要只看平均指标，而要覆盖“最能暴露后处理价值”的场景：  
走廊贴边、单次急转 90°、S 型穿行、窄门通行、桌腿/椅腿密集、动态行人横穿、终点就在障碍边缘、低纹理深度噪声、相机轻微抖动、短视距盲转角。特别是“窄门 + 急转 + 原轨迹贴边”的组合，是最容易体现控制点吸引+clearance 优化价值的 case。  

若做 rosbag 复现，建议至少准备两类 bag：一类是静态狭窄环境，主要验证 clearance 和轮速平顺性；另一类是动态障碍短时遮挡场景，主要观察 smoother 是否会过度偏离原始意图。对实物验证，则优先看三项：是否减少“先冲向障碍再被 MPC 拉回”的过渡风险；是否降低急转时轮速尖峰；是否在窄门前出现更少的左右横跳。

## 风险、应对措施与代码目录建议

最大的技术风险不是“优化算不动”，而是**优化做得太强，反而破坏 NavDP 已经学到的局部决策意图**。这在视觉导航里很常见：模型原本选了右侧绕行，后处理为了追求 clearance 或更小曲率把它拉到了左侧，结果在遮挡结构或动态人群里直接换了 homotopy。应对方式就是前面定义的**各向异性意图代价**：横向偏差权重大于纵向偏差；并且只对 top-K 中 critic 已经认可的候选做局部微调，而不是从头重规划。

第二个风险是**局部距离场质量不足**。NavDP 运行时本身没有显式全局地图，若你只用单帧深度构造 2D EDT，在玻璃、反光、遮挡或黑色地面场景里会有空洞。应对方式是分级实现：原型期用“单帧深度 + 最近几帧局部融合”的轻量 EDT；若在 Isaac 中可直接访问更稳定的碰撞/占据信息，则优先使用仿真侧局部占据层。并且要允许 smoother 在“距离场不可用”时自动退化成纯意图+曲率平滑，而不是整体失效。

第三个风险是**和现有 MPC 的职责冲突**。如果后处理已经给出一条几何与速度都较强约束的轨迹，而当前 `MPC_Controller` 仍然使用固定 `desired_v` 和纯位置跟踪，可能出现“上游平滑想减速，下游控制还在追速度”的不一致。短期应对是把 `desired_v` 改为平滑后前几步参考速度的统计量；中期再把 MPC 的参考扩展到 \((x,y,\theta,v)\) 或至少增加速度参考项。

第四个风险是**实时性抖动**。top-K、ESDF、优化器、MPC 全部加起来，如果不做预算，很容易在复杂场景里出现 planning thread 周期抖动。应对措施是严格设置：K 默认 2，控制点默认 6，最大迭代 15，超过预算直接提前终止并回退原始轨迹；同时在日志中永久记录 `optimizer_status` 和耗时分布，把“质量退化”做成可诊断问题，而不是隐性问题。

推荐的代码目录结构如下。它尽量不改动原有模块，只新增一个 `postprocess` 包，并在评测脚本中接入。

```text
NavDP/
├── baselines/
│   └── navdp/
├── utils_tasks/
│   ├── client_utils.py
│   ├── tracking_utils.py
│   └── postprocess/
│       ├── __init__.py
│       ├── dyn_smoother.py          # 主优化器
│       ├── local_esdf.py            # 深度/占据到 EDT/ESDF
│       ├── spline_utils.py          # 控制点采样、B-spline 重采样、曲率
│       ├── speed_profile.py         # 曲率限速 + 前后向 pass
│       ├── candidate_ranker.py      # critic + post-cost 复排
│       └── fallback.py              # 失败回退策略
├── tests/
│   ├── test_spline_utils.py
│   ├── test_curvature_limits.py
│   ├── test_speed_profile.py
│   ├── test_candidate_ranker.py
│   ├── test_dyn_smoother_smoke.py
│   └── test_planning_thread_integration.py
└── eval_pointgoal_wheeled.py
```

单元测试要点建议至少覆盖五类。  
其一，**几何正确性**：控制点到 spline 重采样后，起点终点是否符合约束，弧长是否单调。  
其二，**曲率与动力学约束**：给定高曲率轨迹，速度 profile 是否能把 \(v^2\kappa\) 压到阈值以下，差速轮速是否不超限。  
其三，**意图保持**：构造左右绕障两条候选，验证 smoother 不会无故换边。  
其四，**fallback**：故意喂入不可行候选或坏距离场，确认系统能稳定回退而不是抛异常。  
其五，**集成烟测**：mock 一个 `pointgoal_step` 返回值，保证 `planning_thread()` 从 top-K 平滑到 `MPC_Controller` 的整条链路能跑通。

## 可直接复制的 Python ROS2 伪代码与四周计划

下面给出一个不超过 200 行的 Python/ROS2 风格伪代码。它的目标不是替代你最终实现，而是把“top-K 取候选、构建局部距离场、控制点优化、速度整形、回退到现有 MPC”的关键接口一次说明白。

```python
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt

@dataclass
class SmoothConfig:
    top_k: int = 2
    num_ctrl: int = 6
    num_eval: int = 32
    dt: float = 0.1
    v_max: float = 0.6
    w_max: float = 1.0
    a_max: float = 0.6
    a_dec_max: float = 0.8
    a_lat_max: float = 0.6
    d_safe: float = 0.20
    intent_w: float = 10.0
    cp_w: float = 3.0
    clear_w: float = 30.0
    curve_w: float = 8.0
    smooth2_w: float = 2.0
    smooth3_w: float = 0.5
    end_w: float = 10.0
    max_iter: int = 12

class DynSmoothOptimizer:
    def __init__(self, cfg: SmoothConfig):
        self.cfg = cfg

    def select_topk(self, all_traj, all_values):
        idx = np.argsort(-all_values)[: self.cfg.top_k]
        return all_traj[idx], all_values[idx], idx

    def build_local_esdf(self, occ_mask, resolution):
        # occ_mask: 1 obstacle, 0 free
        free_dist = distance_transform_edt(1 - occ_mask) * resolution
        obs_dist = distance_transform_edt(occ_mask) * resolution
        return free_dist - obs_dist  # simple signed distance proxy

    def sample_ctrl_points(self, traj_xy):
        s = self.arc_length(traj_xy)
        s_query = np.linspace(0.0, s[-1], self.cfg.num_ctrl)
        ctrl = np.zeros((self.cfg.num_ctrl, 2))
        for j, sq in enumerate(s_query):
            i = np.searchsorted(s, sq)
            i = np.clip(i, 0, len(traj_xy) - 1)
            ctrl[j] = traj_xy[i]
        return ctrl

    def spline_resample(self, ctrl):
        t = np.linspace(0.0, 1.0, len(ctrl))
        tq = np.linspace(0.0, 1.0, self.cfg.num_eval)
        sx = CubicSpline(t, ctrl[:, 0], bc_type="clamped")
        sy = CubicSpline(t, ctrl[:, 1], bc_type="clamped")
        x = sx(tq); y = sy(tq)
        dx = sx(tq, 1); dy = sy(tq, 1)
        ddx = sx(tq, 2); ddy = sy(tq, 2)
        pts = np.stack([x, y], axis=1)
        curv = (dx * ddy - dy * ddx) / (np.power(dx * dx + dy * dy, 1.5) + 1e-6)
        return pts, curv

    def arc_length(self, pts):
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(ds)])

    def tangent_normal(self, traj_xy):
        d = np.gradient(traj_xy, axis=0)
        t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        n = np.stack([-t[:, 1], t[:, 0]], axis=1)
        return t, n

    def esdf_query(self, pts, esdf, origin_xy, resolution):
        ij = np.round((pts - origin_xy) / resolution).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, esdf.shape[1] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, esdf.shape[0] - 1)
        # assume row-major map: y, x
        return esdf[ij[:, 1], ij[:, 0]]

    def optimize_geometry(self, raw_xy, start_xy, esdf, origin_xy, resolution):
        ctrl0 = self.sample_ctrl_points(raw_xy)
        ctrl0[0] = start_xy
        raw_eval = self.resample_for_intent(raw_xy)
        t_hat, n_hat = self.tangent_normal(raw_eval)

        x0 = ctrl0[1:-1].reshape(-1)

        def unpack(z):
            ctrl = ctrl0.copy()
            ctrl[1:-1] = z.reshape(-1, 2)
            ctrl[0] = start_xy
            return ctrl

        def objective(z):
            ctrl = unpack(z)
            pts, curv = self.spline_resample(ctrl)
            d = self.esdf_query(pts, esdf, origin_xy, resolution)
            lat = np.sum(((pts - raw_eval) * n_hat).sum(axis=1) ** 2)
            lon = np.sum(((pts - raw_eval) * t_hat).sum(axis=1) ** 2)
            intent = lat + 0.2 * lon
            cp = np.sum((ctrl - self.sample_ctrl_points(raw_xy)) ** 2)
            clear = np.sum(np.maximum(0.0, self.cfg.d_safe - d) ** 2)
            curve = np.sum(np.maximum(0.0, np.abs(curv) - 1e-6) ** 2)
            diff2 = np.sum(np.diff(pts, n=2, axis=0) ** 2)
            diff3 = np.sum(np.diff(pts, n=3, axis=0) ** 2)
            end_cost = np.sum((pts[-1] - raw_xy[-1]) ** 2)
            return (
                self.cfg.intent_w * intent +
                self.cfg.cp_w * cp +
                self.cfg.clear_w * clear +
                self.cfg.curve_w * curve +
                self.cfg.smooth2_w * diff2 +
                self.cfg.smooth3_w * diff3 +
                self.cfg.end_w * end_cost
            )

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.cfg.max_iter, "ftol": 1e-4},
        )
        ctrl = unpack(res.x)
        pts, curv = self.spline_resample(ctrl)
        return pts, curv, res

    def resample_for_intent(self, raw_xy):
        s = self.arc_length(raw_xy)
        sq = np.linspace(0.0, s[-1], self.cfg.num_eval)
        x = np.interp(sq, s, raw_xy[:, 0])
        y = np.interp(sq, s, raw_xy[:, 1])
        return np.stack([x, y], axis=1)

    def build_speed_profile(self, pts, curv, clearances):
        s = self.arc_length(pts)
        ds = np.diff(s, prepend=s[0] + 1e-6)
        v_curve = np.sqrt(np.maximum(1e-6, self.cfg.a_lat_max / (np.abs(curv) + 1e-6)))
        v_w = self.cfg.w_max / (np.abs(curv) + 1e-6)
        v_clear = np.clip(0.10 + 1.5 * np.maximum(0.0, clearances - self.cfg.d_safe), 0.10, self.cfg.v_max)
        v = np.minimum.reduce([np.full_like(v_curve, self.cfg.v_max), v_curve, v_w, v_clear])

        # forward pass
        for i in range(1, len(v)):
            v[i] = min(v[i], np.sqrt(max(1e-6, v[i-1] ** 2 + 2 * self.cfg.a_max * ds[i])))
        # backward pass
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(max(1e-6, v[i+1] ** 2 + 2 * self.cfg.a_dec_max * ds[i+1])))
        return v

    def post_rank(self, critic_value, clear_min, curvature_violation, optim_ok):
        infeasible = (clear_min < self.cfg.d_safe * 0.5) or curvature_violation or (not optim_ok)
        return critic_value - 10.0 * float(infeasible) + 0.5 * clear_min

    def smooth_topk(self, all_traj_world, all_values, start_xy, esdf, origin_xy, resolution):
        cands, vals, idxs = self.select_topk(all_traj_world, all_values)
        best = None
        best_score = -1e9

        for k in range(len(cands)):
            raw_xy = cands[k][:, :2]
            pts, curv, res = self.optimize_geometry(raw_xy, start_xy, esdf, origin_xy, resolution)
            clear = self.esdf_query(pts, esdf, origin_xy, resolution)
            v = self.build_speed_profile(pts, curv, clear)
            curv_bad = np.any(v * np.abs(curv) > self.cfg.w_max + 1e-3)
            score = self.post_rank(vals[k], clear.min(), curv_bad, res.success)
            if score > best_score:
                best_score = score
                best = {
                    "traj_xy": pts,
                    "speed": v,
                    "candidate_idx": int(idxs[k]),
                    "critic": float(vals[k]),
                    "clear_min": float(clear.min()),
                    "optim_success": bool(res.success),
                }

        return best

# ------------ integration in planning_thread ------------
def planning_step_with_smoother(pointgoal_step_fn, smoother, goal, image, depth,
                                camera_pos, camera_rot, robot_xy, local_occ, map_origin, map_res):
    traj_cam, all_traj_cam, all_values = pointgoal_step_fn(goal, image, depth)
    # transform top-K candidates from camera to world
    all_traj_world = []
    for cand in all_traj_cam[0]:  # one env example
        pts = []
        for p in cand:
            pw = camera_pos + camera_rot @ np.array([p[0], p[1], 0.0])
            pts.append([pw[0], pw[1], p[2]])
        all_traj_world.append(np.array(pts))
    all_traj_world = np.array(all_traj_world)

    esdf = smoother.build_local_esdf(local_occ, map_res)
    best = smoother.smooth_topk(
        all_traj_world=all_traj_world,
        all_values=all_values[0],
        start_xy=np.array(robot_xy),
        esdf=esdf,
        origin_xy=np.array(map_origin),
        resolution=map_res,
    )

    if best is None:
        # fallback to original top-1
        return traj_cam, None

    # feed best["traj_xy"] into MPC reference, optional desired_v = np.mean(best["speed"][:5])
    return best["traj_xy"], best
```

最后给出一个尽量务实的四周落地计划。这个计划按“先打通、再稳住、再验证”的顺序安排，更适合你现在的目标。

| 周次 | 目标 | 具体任务 | 交付物 |
|---|---|---|---|
| 第一周 | 打通链路 | 在 `planning_thread()` 中拿到 top-K；补齐相机系到世界系的候选转换；实现局部 EDT/ESDF；加入日志与可视化 | 能显示 top-K + clearance 热力图的调试版 |
| 第二周 | 做出几何平滑 | 完成稀疏控制点采样、意图代价、clearance 代价、曲率/光滑度代价；实现 K=1/K=2 原型与 fallback | 几何平滑版，可在 Isaac 中稳定跑 |
| 第三周 | 加入动力学 | 完成速度 profile、轮速/角速度/横向加速度限制；与现有 MPC 对齐；做参数扫表 | 完整两阶段平滑版，含耗时统计 |
| 第四周 | 系统验证 | 完成仿真消融、rosbag 回放、少量实物测试；固化默认参数；补单元测试和 README | 可提交的实验报告与合入代码 |

从长期看，我赞同你前面提出的“把运动学/动力学代价前置到训练端”的方向，但**一月交付版本**不建议把精力放在改训练。更合理的路径是：先把本文这个 post-optimizer 跑通，用它沉淀出最有效的曲率、clearance、速度和意图保持代价；随后再把这些代价迁移到专家轨迹生成与监督标签优化里，让模型天然学会“更可执行的局部轨迹”。论文本来就已经在专家数据生成中使用 ESDF refine + spline smoothing，这条路线和 NavDP 的原始设计并不冲突，而是顺着它往前推进。citeturn21view0turn35academia0