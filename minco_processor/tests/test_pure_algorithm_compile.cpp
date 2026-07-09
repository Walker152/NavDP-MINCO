#include <Eigen/Core>

#include "minco_core/corridor_generator.hpp"
#include "minco_core/components/trajectory_safety_checker.hpp"
#include "minco_processor/minco_pipeline.hpp"
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
  minco_planner::SimpleCorridorGenerator corridor;
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
  request.goal = Eigen::Vector3d::UnitX();
  request.now = 1.0;
  const auto result = pipeline.optimize(request);

  (void)optimizer;
  (void)corridor;
  if (checker.getDistance(Eigen::Vector3d::Zero()) != 0.0) {
    return 1;
  }
  if (!result.success || result.sparse_waypoints.size() < 2U || result.initial_times.size() == 0) {
    return 2;
  }
  return result.trajectory.getTotalDuration() > 0.0 ? 0 : 3;
}
