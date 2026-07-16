#ifndef MINCO_PLANNER__TRAJECTORY_SAFETY_CHECKER_HPP_
#define MINCO_PLANNER__TRAJECTORY_SAFETY_CHECKER_HPP_

#include <Eigen/Core>

#include <memory>
#include <limits>
#include <string>

#include "data_structure/base/trajectory.h"
#include "minco_processor/esdf_map.hpp"

namespace minco_planner {

struct TrajectorySafetyReport
{
  bool safe{false};
  std::string reason{"NO_MAP"};
  double min_clearance{std::numeric_limits<double>::infinity()};
  int out_of_bounds_count{0};
};

class TrajectorySafetyChecker
{
public:
  void configure(double safe_dist, double sample_dt);
  void setQuery(std::shared_ptr<minco_processor::EsdfMapInterface> dynamic_query);

  bool checkPoint(const Eigen::Vector3d & pos) const;
  bool checkTrajectory(const geometry_utils::Trajectory & traj) const;
  TrajectorySafetyReport inspectTrajectory(const geometry_utils::Trajectory & traj) const;
  double getDistance(const Eigen::Vector3d & pos) const;
  bool projectOutOfObstacle(Eigen::Vector3d & pos, double margin) const;

private:
  std::shared_ptr<minco_processor::EsdfMapInterface> dynamic_query_;
  double safe_dist_{0.0};
  double sample_dt_{0.05};
};

}  // namespace minco_planner

#endif  // MINCO_PLANNER__TRAJECTORY_SAFETY_CHECKER_HPP_
