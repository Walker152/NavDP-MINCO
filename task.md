# Codex 改造任务：修复 MINCO yaw 真实状态传递，保持世界系速度输入不变

## 任务背景

当前 NavDP → MINCO → MPC 链路中：

1. MINCO 是通用质点轨迹优化器，起始速度应继续使用机器人实际反馈的世界系速度：

   ```text
   [vx_world, vy_world, 0]

MPC 是差速机器人运动学控制器，实际线速度继续使用世界系速度在机器人车头方向上的投影：

v_forward = vx_world * cos(yaw) + vy_world * sin(yaw)

当前 eval_pointgoal_wheeled.py 已经把实际世界系线速度和实际 yaw rate 放入 MINCO 的 states：

vel = robot_lin_vel_w[idx].copy()
vel[2] = 0.0

yaw_rate = float(robot_ang_vel_w[idx])
当前真正的问题在 C++ 接口和 yaw 优化内部：
minco_pybind.cpp 中通过 (void)yaw_rate 丢弃了实际角速度；
MincoPipeline::optimizeYaw() 没有接收实际 yaw rate；
当平面速度大于阈值时，yaw 初值被速度方向覆盖，而不是优先使用真实反馈 yaw；
导致机器人正在转弯时，新生成的 yaw 轨迹仍可能从零角速度或错误姿态开始。

本轮只修复真实 yaw/yaw rate 状态传递，不修改 MINCO 的质点模型，不修改 MPC 模型，不进行曲率约束改造。

不要先输出计划文档，直接检查仓库实际代码并完成修改、编译检查和必要验证。禁止执行任何 git 提交、推送、重置或分支操作。

一、总体要求

采用最小改动原则。

本轮必须保持：

MINCO 起始速度 = 实际反馈世界系速度 [vx_world, vy_world, 0]
MPC 实际线速度 = 世界系速度在车头方向上的投影 v_forward
MPC 实际角速度 = root_ang_vel_w 的 z 分量

禁止将 MINCO 世界系速度改造成：

vx = v_forward * cos(yaw)
vy = v_forward * sin(yaw)

这种重构会主动删除实际反馈中的横向速度，不属于本轮需求。

禁止修改：

MINCO 位置轨迹优化模型；
MINCO 速度和加速度代价；
MPC 状态方程；
MPC 参考轨迹生成；
曲率限速；
ESDF；
NavDP 候选轨迹；
fallback、hold-last 和 episode reset 逻辑。
二、确认 Python 侧世界系速度保持不变

重点检查：

eval_pointgoal_wheeled.py
utils_tasks/basic_utils.py
utils_tasks/navdp_minco_adapter.py
2.1 MINCO 状态构造

保持现有语义：

vel = (
    robot_lin_vel_w[idx].copy()
    if robot_lin_vel_w is not None
    else np.zeros(3, dtype=np.float64)
)
vel[2] = 0.0

这里的 robot_lin_vel_w 必须继续来自：

robot.data.root_lin_vel_w[:, :3]

它是实际反馈的世界系速度，不是控制命令。

只允许补充必要的有限值保护，例如：

if not np.all(np.isfinite(vel)):
    vel = np.zeros(3, dtype=np.float64)

不得做车体前向投影后再重构世界速度。

2.2 yaw 和 yaw rate

保持：

state_yaw = float(robot_yaw_w[idx])
yaw_rate = float(robot_ang_vel_w[idx])

其中 robot_ang_vel_w 在主控制循环中应继续取：

root_ang_vel_w[:, 2]

只允许增加有限值保护：

if not np.isfinite(state_yaw):
    # 使用现有相机朝向或其他已有可靠姿态作为兜底
    ...

if not np.isfinite(yaw_rate):
    yaw_rate = 0.0

不要增加滤波、限幅或新的状态估计器。

2.3 MPC 反馈保持不变

确认 MPC 实际线速度继续由：

compute_forward_velocity(robot_lin_vel_w_xy, robot_yaw_w)

计算，实际角速度继续使用：

root_ang_vel_w[:, 2]

本轮不修改这一逻辑。

三、修复 pybind 中 yaw rate 被丢弃的问题

重点修改：

minco_pybind.cpp

当前存在：

(void)yaw_rate;

删除该语句。

把输入的 yaw rate 写入 MINCO 请求当前状态：

request.current.yaw = std::isfinite(yaw) ? yaw : 0.0;
request.current.yaw_rate = std::isfinite(yaw_rate) ? yaw_rate : 0.0;

如果 request.current 对应的状态结构当前没有 yaw_rate 字段，在其真实定义所在的头文件中增加：

double yaw_rate{0.0};

要求：

字段放在当前状态结构中，与 position、velocity、acceleration、yaw 同级；
不新建另一套 yaw 状态结构；
不通过全局变量、静态变量或额外缓存传递；
Python 接口参数名和调用方式保持兼容；
合法的 yaw_rate 必须完整传递到 pipeline。
四、将真实 yaw rate 接入 MINCO yaw 初始边界

重点修改：

minco_pipeline.cpp
对应的 minco_pipeline.hpp

当前 optimizeYaw() 的参数中只有：

current_yaw
goal_yaw

扩展为接收：

double current_yaw,
double current_yaw_rate,
double goal_yaw

修改声明、定义和所有调用点，保持签名完全一致。

在调用 yaw 优化时传入：

request.current.yaw,
request.current.yaw_rate,
goal_yaw

不得在调用前把 yaw rate 置零。

五、修复 yaw 初始状态构造

当前逻辑在平面速度较大时使用：

atan2(vy, vx)

覆盖当前真实 yaw。

修改为：

Eigen::Vector4d init_yaw_state = Eigen::Vector4d::Zero();
Eigen::Vector4d goal_yaw_state = Eigen::Vector4d::Zero();

double init_yaw = current_yaw;

if (!std::isfinite(init_yaw)) {
  const Eigen::Vector2d vel_xy = start_state.col(1).head<2>();
  if (vel_xy.allFinite() && vel_xy.norm() > 0.1) {
    init_yaw = std::atan2(vel_xy.y(), vel_xy.x());
  } else {
    init_yaw = 0.0;
  }
}

init_yaw_state(0) = init_yaw;
init_yaw_state(1) =
  std::isfinite(current_yaw_rate) ? current_yaw_rate : 0.0;
init_yaw_state(2) = 0.0;
init_yaw_state(3) = 0.0;

核心语义必须是：

真实反馈 yaw 有效时：
    yaw 初值始终使用真实 yaw

真实反馈 yaw 无效时：
    才允许使用位置轨迹初始速度方向作为兜底

真实反馈 yaw rate 有效时：
    yaw_dot 初值使用真实 yaw rate

禁止继续采用：

speed > threshold ? atan2(vy, vx) : current_yaw

因为速度方向是质点轨迹切向，不等同于机器人实际姿态测量。

六、保持现有 yaw 插值结构

当前 yaw 优化使用五阶插值时已经支持：

init_state3 << yaw, yaw_rate, yaw_acc;

因此不要重写 YawTrajOpt 的插值算法。

只需确认：

init_yaw_state(0) -> 实际 yaw
init_yaw_state(1) -> 实际 yaw rate
init_yaw_state(2) -> 0

能够原样进入五阶 minimum-jerk yaw 插值。

不要在 yaw_traj_opt.cpp 中：

用位置速度重新覆盖 yaw；
把 yaw rate 再次清零；
新增额外 yaw 代价；
修改 waypoint 分配；
修改 yaw 最大速度判定；
修改时间分配。

如果无需修改该文件，应保持不动。

七、goal yaw 与角度连续性

保留当前 goal yaw 的处理方式：

const double yaw_err =
  std::atan2(
    std::sin(goal_yaw - init_yaw_state(0)),
    std::cos(goal_yaw - init_yaw_state(0))
  );

goal_yaw_state(0) = init_yaw_state(0) + yaw_err;

终点 yaw rate 和更高阶导数保持现有语义，不在本轮增加新的终点角速度策略。

八、yaw 优化失败时的回退

保留当前 yaw 优化失败后的安全回退结构。

回退轨迹的 yaw 应继续使用有效的真实：

request.current.yaw

若其无效则使用零或现有可靠兜底值。

本轮不要求在 yaw 优化失败时保持当前非零 yaw rate，因为常量 yaw fallback 本身无法表达连续旋转。不要为了这一点新增复杂回退轨迹。

九、禁止的改动

本轮禁止：

将 MINCO 的世界系速度替换为 v_forward 重构速度；
删除实际反馈中的横向速度；
修改 compute_forward_velocity()；
修改 MPC 模型或代价函数；
修改 MINCO 位置 PVA 优化；
增加差速约束或曲率约束；
增加轮速约束；
增加滤波器或状态观测器；
修改 yaw 最大角速度参数；
增加高频调试输出；
进行无关重构、重命名或格式化整个文件。
十、验证要求
10.1 静态检查

确认代码中不再存在：

(void)yaw_rate;

确认 yaw_rate 的完整链路为：

Isaac Sim root_ang_vel_w[:, 2]
→ planning_input.robot_ang_vel_w
→ states["yaw_rate"]
→ NavDPMincoAdapter
→ MincoProcessor.optimize(yaw_rate=...)
→ request.current.yaw_rate
→ MincoPipeline::optimizeYaw()
→ init_yaw_state(1)
→ YawTrajOpt 五阶插值
→ samples[:, 14]
10.2 世界系速度检查

确认 MINCO 速度链路仍为：

Isaac Sim root_lin_vel_w[:, :3]
→ planning_input.robot_lin_vel_w
→ states["velocity"]
→ request.current.velocity
→ MINCO start_state

不得出现：

v_forward → cos/sin → MINCO velocity
10.3 行为验证

至少验证三种初始状态：

静止
v = 0
w = 0

预期：

MINCO 起始位置速度为零；
yaw 起点等于当前机器人 yaw；
yaw_dot 起点为零。
直行
v > 0
w ≈ 0

预期：

MINCO 位置起始速度等于实际世界系反馈；
yaw 起点等于实际机器人 yaw；
不因速度方向存在轻微偏差而覆盖真实 yaw；
yaw_dot 起点接近零。
正在转弯
v > 0
w != 0

预期：

MINCO 位置起始速度仍等于实际世界系反馈；
yaw 起点等于实际机器人 yaw；
yaw 轨迹初始导数继承实际反馈 yaw rate；
输出 samples 的首段 yaw_dot 不再无条件从零开始。

允许插值器和离散采样带来小量数值误差，不要求首个离散样本与输入值逐位完全相等，但必须保持边界条件语义一致。

10.4 编译和语法检查

完成：

C++ 相关目标编译；
pybind 模块编译；
Python 文件语法检查；
调用参数数量和名称检查；
对应头文件声明与实现签名检查。

不要仅修改源码而不编译接口。

十一、最终汇报

完成后只汇报：

修改了哪些文件；
yaw_rate 现在经过了哪些接口；
MINCO 世界系速度为何保持不变；
yaw 初始角和初始角速度现在如何构造；
完成了哪些编译和运行检查；
是否仍存在失败项。

不要生成新的设计文档，不要执行任何 git 操作。