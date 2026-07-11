#include <Eigen/Core>

#include "minco_core/components/trajectory_safety_checker.hpp"
#include "minco_processor/minco_pipeline.hpp"
#include "minco_processor/static_sim_esdf_map.hpp"
#include "traj_opt/minco_optimizer.hpp"

namespace {

class FreeSpaceEsdf final : public minco_processor::EsdfMapInterface
{
public:
  minco_processor::QueryResult query(const Eigen::Vector3d &) const override
  {
    minco_processor::QueryResult result;
    result.ok = true;
    result.status = minco_processor::QueryStatus::kOk;
    result.distance = 10.0;
    result.gradient = Eigen::Vector3d::UnitX();
    return result;
  }

  double resolution() const override { return 0.1; }
  bool worldToMap(double, double, unsigned int & mx, unsigned int & my) const override
  {
    mx = 0U;
    my = 0U;
    return true;
  }
  bool isFree(unsigned int, unsigned int) const override { return true; }
};

}  // namespace

int main()
{
  minco_planner::MincoOptimizer::Config cfg;
  cfg.magnitudeBounds = Eigen::VectorXd::Constant(3, 1.0);
  cfg.penaltyWeights = Eigen::VectorXd::Zero(5);
  cfg.print_optimizer_log = false;

  minco_planner::MincoOptimizer optimizer(cfg);
  minco_planner::TrajectorySafetyChecker checker;
  checker.configure(0.3, 0.05);
  minco_processor::MincoPipeline::Config pipeline_cfg;
  pipeline_cfg.optimizer.print_optimizer_log = false;
  pipeline_cfg.optimizer.penaltyWeights.setZero();
  pipeline_cfg.traj_goal_tolerance = 0.5;
  minco_processor::MincoPipeline pipeline(pipeline_cfg);
  pipeline.setMap(std::make_shared<FreeSpaceEsdf>());
  minco_processor::MincoPipeline::Request request;
  request.guide_path = {
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d::UnitX()};
  request.current.position = Eigen::Vector3d::Zero();
  request.current.velocity = Eigen::Vector3d(0.5, 0.0, 0.0);
  request.current.yaw = 0.7;
  request.current.yaw_rate = 0.35;
  request.now = 1.0;
  const auto local_result = pipeline.optimize(request);
  if (!local_result.success || local_result.local_end_is_goal ||
      local_result.end_state.col(1).head<2>().norm() <= 1e-6) {
    return 8;
  }
  request.has_terminal_goal = true;
  request.terminal_goal = Eigen::Vector3d::UnitX();
  const auto result = pipeline.optimize(request);
  if (!result.local_end_is_goal || result.end_state.col(1).head<2>().norm() > 1e-6) {
    return 9;
  }

  (void)optimizer;
  if (checker.getDistance(Eigen::Vector3d::Zero()) != 0.0) {
    return 1;
  }
  if (!result.success || result.sparse_waypoints.size() < 2U || result.initial_times.size() == 0 ||
      result.samples.empty()) {
    return 2;
  }
  if (std::abs(result.samples.front().yaw - request.current.yaw) > 1e-6 ||
      std::abs(result.samples.front().yaw_dot - request.current.yaw_rate) > 1e-6) {
    return 10;
  }
  for (const auto & sample : result.samples) {
    if (sample.pos.z() != 0.0 || sample.vel.z() != 0.0 || sample.acc.z() != 0.0 ||
        sample.jerk.z() != 0.0) {
      return 3;
    }
  }
  minco_processor::StaticSimEsdfMap2D static_map;
  Eigen::MatrixXd distance(3, 3);
  distance << 1.0, 1.0, 1.0,
    1.0, 0.5, 1.0,
    1.0, 1.0, 1.0;
  Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> free(3, 3);
  free.setConstant(1U);
  free(1, 1) = 0U;
  static_map.setMap(distance, free, -0.1, -0.1, 0.1);
  unsigned int mx = 0U;
  unsigned int my = 0U;
  if (!static_map.worldToMap(0.0, 0.0, mx, my) || mx != 1U || my != 1U || static_map.isFree(mx, my)) {
    return 5;
  }
  const auto q = static_map.query(Eigen::Vector3d(0.0, 0.0, 0.0));
  if (!q.ok || std::abs(q.distance - 0.5) > 1e-9 || q.gradient.z() != 0.0) {
    return 6;
  }
  if (static_map.query(Eigen::Vector3d(10.0, 0.0, 0.0)).ok) {
    return 7;
  }
  return result.trajectory.getTotalDuration() > 0.0 ? 0 : 4;
}
