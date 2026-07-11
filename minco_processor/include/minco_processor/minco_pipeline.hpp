#pragma once

#include <Eigen/Core>

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "data_structure/base/trajectory.h"
#include "minco_core/components/trajectory_safety_checker.hpp"
#include "minco_processor/esdf_map.hpp"
#include "traj_opt/minco_optimizer.hpp"
#include "traj_opt/yaw_traj_opt.h"
#include "utils/header/eigen_alias.hpp"

namespace minco_processor {

class MincoPipeline
{
public:
  enum class PlanningState
  {
    kColdStart,
    kHotStart,
  };

  struct Config
  {
    Config();

    minco_planner::MincoOptimizer::Config optimizer;
    double lookahead_dist{5.0};
    double traj_goal_tolerance{0.3};
    double safety_sample_dt{0.05};
    double validation_sample_dt{0.05};
    double sample_dt{0.05};
    double validation_dynamic_scale{1.5};
    double start_projection_margin{0.05};
    int max_sparse_waypoints{8};
    bool enable_yaw_opt{true};
  };

  struct State
  {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    double yaw{0.0};
    double yaw_rate{0.0};
  };

  struct Request
  {
    std::vector<Eigen::Vector3d> guide_path;
    State current;
    bool has_terminal_goal{false};
    Eigen::Vector3d terminal_goal{Eigen::Vector3d::Zero()};
    double goal_yaw{std::numeric_limits<double>::quiet_NaN()};
    double now{0.0};
  };

  struct TrajectorySample
  {
    double t{0.0};

    Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vel{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acc{Eigen::Vector3d::Zero()};
    Eigen::Vector3d jerk{Eigen::Vector3d::Zero()};

    double yaw{0.0};
    double yaw_dot{0.0};
  };

  struct Result
  {
    bool success{false};
    std::string failure_reason{"NOT_RUN"};
    PlanningState planning_state{PlanningState::kColdStart};
    std::vector<Eigen::Vector3d> dense_path;
    std::vector<Eigen::Vector3d> sparse_waypoints;
    Eigen::Matrix3d start_state{Eigen::Matrix3d::Zero()};
    Eigen::Matrix3d end_state{Eigen::Matrix3d::Zero()};
    super_utils::vec_Vec3f initial_points;
    super_utils::VecDf initial_times;
    super_utils::VecDf local_vmaxs;
    geometry_utils::Trajectory trajectory;
    geometry_utils::Trajectory yaw_trajectory;
    std::vector<TrajectorySample> samples;
    double objective{std::numeric_limits<double>::infinity()};
    int optimizer_return_code{0};
    std::map<std::string, double> timing_ms;
    int dense_path_size{0};
    int sparse_waypoint_size{0};
    bool local_end_is_goal{false};
    int mandatory_corner_count{0};
    int optimizer_iteration_count{0};
  };

  MincoPipeline();
  explicit MincoPipeline(const Config & config);

  void setConfig(const Config & config);
  void setMap(std::shared_ptr<EsdfMapInterface> map);
  void resetHistory();

  Result optimize(const Request & request);

private:
  std::vector<Eigen::Vector3d> extractLocalPath(
    const std::vector<Eigen::Vector3d> & guide_path, const Eigen::Vector3d & current_position) const;
  std::vector<Eigen::Vector3d> sparsifyPath(
    const std::vector<Eigen::Vector3d> & dense_path, bool local_end_is_goal, int * mandatory_corner_count) const;
  bool isLineFree(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2) const;

  PlanningState determinePlanningState(
    const State & current, const std::vector<Eigen::Vector3d> & sparse_path, double now) const;
  void prepareColdStart(const State & current, Eigen::Matrix3d & start_state) const;
  void prepareHotStart(const State & current, double sample_t, Eigen::Matrix3d & start_state) const;
  void allocatePathTime(const std::vector<Eigen::Vector3d> & sparse_path,
    const Eigen::Matrix3d & start_state,
    bool local_end_is_goal,
    PlanningState state,
    bool has_shifted_seed,
    const super_utils::vec_Vec3f & shifted_waypoints,
    const super_utils::VecDf & shifted_durations,
    super_utils::vec_Vec3f & init_ps,
    super_utils::VecDf & init_ts,
    super_utils::VecDf & local_vmaxs) const;
  bool validateTrajectory(
    const geometry_utils::Trajectory & trajectory, const Eigen::Vector3d & expected_end_pos) const;
  bool optimizeYaw(const Eigen::Matrix3d & start_state,
    const geometry_utils::Trajectory & pos_traj,
    geometry_utils::Trajectory & out_yaw_traj,
    PlanningState state,
    double current_yaw,
    double current_yaw_rate,
    double goal_yaw,
    double now) const;
  std::vector<TrajectorySample> sampleTrajectory(const geometry_utils::Trajectory & pos_traj,
    const geometry_utils::Trajectory & yaw_traj,
    double dt,
    double fallback_yaw) const;

  Config config_;
  std::shared_ptr<EsdfMapInterface> map_;
  std::unique_ptr<minco_planner::MincoOptimizer> optimizer_;
  std::unique_ptr<traj_opt::YawTrajOpt> yaw_optimizer_;
  std::unique_ptr<minco_planner::TrajectorySafetyChecker> safety_checker_;

  geometry_utils::Trajectory last_traj_;
  geometry_utils::Trajectory last_yaw_traj_;
  bool has_last_traj_{false};
  bool has_last_yaw_traj_{false};
};

}  // namespace minco_processor
