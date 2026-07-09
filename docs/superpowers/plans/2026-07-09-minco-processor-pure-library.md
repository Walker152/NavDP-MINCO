# Minco Processor Pure Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `minco_processor` into a ROS-free C++ algorithm library for MINCO trajectory optimization and ESDF queries.

**Architecture:** Keep the numerical trajectory stack (`utils`, `traj_opt`, selected corridor/safety helpers) and replace `rog_map`/Nav2 map access with a small pure C++ ESDF query interface. Build and install only the pure headers and library target through standard CMake.

**Tech Stack:** C++17, CMake, Eigen3, OpenMP, yaml-cpp, local MINCO/L-BFGS utilities.

---

### Task 1: Compile Contract

**Files:**
- Create: `minco_processor/tests/test_pure_algorithm_compile.cpp`
- Modify: `minco_processor/CMakeLists.txt`

- [ ] Add a minimal executable that includes the intended public pure headers and constructs the optimizer/config objects.
- [ ] Run CMake/build and verify it fails before production changes because ROS/ROGMap headers are still required.
- [ ] Keep the executable as a compile contract after the refactor.

### Task 2: Pure ESDF Interface

**Files:**
- Create: `minco_processor/include/minco_processor/esdf_map.hpp`
- Modify: `minco_processor/include/traj_opt/minco_optimizer.hpp`
- Modify: `minco_processor/src/traj_opt/minco_optimizer.cpp`
- Modify: `minco_processor/include/minco_core/corridor_generator.hpp`
- Modify: `minco_processor/src/minco_core/corridor_generator.cpp`
- Modify: `minco_processor/include/minco_core/components/trajectory_safety_checker.hpp`
- Modify: `minco_processor/src/minco_core/components/trajectory_safety_checker.cpp`

- [ ] Define `QueryStatus`, `QueryResult`, and `EsdfMapInterface`.
- [ ] Replace `rog_map::MapQueryInterface` pointers with `minco_processor::EsdfMapInterface`.
- [ ] Remove ROS logging and Nav2 cost constants from safety checks.

### Task 3: Pure CMake Target

**Files:**
- Modify: `minco_processor/CMakeLists.txt`
- Modify: `minco_processor/package.xml`

- [ ] Remove `ament_cmake`, Nav2, ROS message, pluginlib, TF, and `rog_map` dependencies.
- [ ] Build only `src/utils/*.cpp`, `src/traj_opt/*.cpp`, `src/minco_core/corridor_generator.cpp`, and `src/minco_core/components/trajectory_safety_checker.cpp`.
- [ ] Install only pure public headers.

### Task 4: Verification

**Files:**
- Build directory: `minco_processor/build`

- [ ] Configure with `cmake -S minco_processor -B minco_processor/build`.
- [ ] Build with `cmake --build minco_processor/build`.
- [ ] Fix compile errors without reintroducing ROS/Nav2 dependencies.
