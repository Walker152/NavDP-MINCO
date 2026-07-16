#include "minco_core/components/trajectory_safety_checker.hpp"

#include <cmath>
#include <limits>

namespace minco_planner {

void TrajectorySafetyChecker::configure(double safe_dist, double sample_dt)
{
  safe_dist_ = safe_dist;
  sample_dt_ = sample_dt > 1e-6 ? sample_dt : 0.05;
}

void TrajectorySafetyChecker::setQuery(std::shared_ptr<minco_processor::EsdfMapInterface> dynamic_query)
{
  dynamic_query_ = std::move(dynamic_query);
}

bool TrajectorySafetyChecker::checkPoint(const Eigen::Vector3d & pos) const
{
  if (!dynamic_query_) {
    return false;
  }

  const auto query = dynamic_query_->query(pos);
  if (!query.ok) {
    return false;
  }
  return std::isfinite(query.distance) && query.distance > safe_dist_;
}

bool TrajectorySafetyChecker::checkTrajectory(const geometry_utils::Trajectory & traj) const
{
  return inspectTrajectory(traj).safe;
}

TrajectorySafetyReport TrajectorySafetyChecker::inspectTrajectory(
  const geometry_utils::Trajectory & traj) const
{
  TrajectorySafetyReport report;
  if (!dynamic_query_) {
    return report;
  }
  const double dur = traj.getTotalDuration();
  for (double t = 0.0; t <= dur; t += sample_dt_) {
    const auto query = dynamic_query_->query(traj.getPos(t));
    if (!query.ok || !std::isfinite(query.distance)) {
      ++report.out_of_bounds_count;
      report.reason = "VALIDATION_ESDF_OOB";
      return report;
    }
    report.min_clearance = std::min(report.min_clearance, query.distance);
    if (query.distance <= safe_dist_) {
      report.reason = "VALIDATION_CLEARANCE";
      return report;
    }
  }
  report.safe = true;
  report.reason = "NONE";
  return report;
}

double TrajectorySafetyChecker::getDistance(const Eigen::Vector3d & pos) const
{
  if (!dynamic_query_) {
    return 0.0;
  }
  double dist = 0.0;
  Eigen::Vector3d grad = Eigen::Vector3d::Zero();
  const auto query = dynamic_query_->query(pos);
  dist = query.distance;
  grad = query.gradient;
  (void)grad;
  return query.ok ? dist : std::numeric_limits<double>::quiet_NaN();
}

bool TrajectorySafetyChecker::projectOutOfObstacle(Eigen::Vector3d & pos, double margin) const
{
  if (!dynamic_query_) {
    return false;
  }
  double esdf_dist = 0.0;
  Eigen::Vector3d esdf_grad = Eigen::Vector3d::Zero();
  const auto query = dynamic_query_->query(pos);
  if (!query.ok) {
    return false;
  }
  esdf_dist = query.distance;
  esdf_grad = query.gradient;
  if (esdf_dist < 0.0 && esdf_grad.norm() > 1e-6) {
    pos += (margin - esdf_dist) * esdf_grad.normalized();
    return true;
  }
  return false;
}

}  // namespace minco_planner
