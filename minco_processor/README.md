# minco_processor

`minco_processor` is a pure C++ algorithm library for NavDP trajectory post-processing.
It no longer builds as a ROS 2/Nav2 plugin and does not depend on `rclcpp`, Nav2,
ROS messages, TF, pluginlib, or ROGMap.

## Retained Scope

- MINCO trajectory primitives and polynomial trajectory data structures.
- Initial path state construction through waypoint, start state, and end state inputs.
- Time allocation and warm-start support inside `MincoOptimizer`.
- Joint optimization of segment times and intermediate control points.
- ESDF-based position penalty through `minco_processor::EsdfMapInterface`.
- Trajectory safety checking by sampling optimized trajectories against ESDF distance.
- Yaw trajectory optimization.

Removed ROS/Nav2 surfaces include the global planner plugin, FSM, visualizer,
SMAC/Nav2 costmap search, TF adapters, ROS publishers/subscribers, and message output.

## Build

```bash
cmake -S minco_processor -B minco_processor/build
cmake --build minco_processor/build
```

Optional install:

```bash
cmake --install minco_processor/build --prefix /tmp/minco_processor_install
```

The installed package exports `minco_processor::minco_processor` for CMake consumers:

```cmake
find_package(minco_processor REQUIRED)
target_link_libraries(your_target PRIVATE minco_processor::minco_processor)
```

## ESDF Interface

Consumers provide map access by implementing:

```cpp
#include "minco_processor/esdf_map.hpp"

class MyEsdfMap : public minco_processor::EsdfMapInterface
{
public:
  minco_processor::QueryResult query(const Eigen::Vector3d & pos) const override;
};
```

`QueryResult::distance` is the signed clearance used by optimization and safety
checks. Positive values are free-space clearance, negative values indicate the
query point is inside an obstacle, and `gradient` should point toward increasing
clearance when available.

## Core Headers

- `traj_opt/minco_optimizer.hpp`: MINCO joint time/control-point optimizer.
- `traj_opt/yaw_traj_opt.h`: yaw trajectory optimization.
- `minco_processor/esdf_map.hpp`: pure ESDF query interface.
- `minco_core/components/trajectory_safety_checker.hpp`: trajectory ESDF sampling checker.

## Compile Contract

`tests/test_pure_algorithm_compile.cpp` is built as `minco_processor_compile_test`.
It verifies that the retained public headers can be included and linked without
ROS/Nav2 dependencies.
