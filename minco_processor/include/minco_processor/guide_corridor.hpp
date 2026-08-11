#pragma once

#include <Eigen/Core>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "minco_processor/esdf_map.hpp"

namespace minco_processor {

struct GuideCorridorSegment
{
  Eigen::Vector3d start{Eigen::Vector3d::Zero()};
  Eigen::Vector3d end{Eigen::Vector3d::Zero()};
  double radius{0.0};
  double min_clearance{std::numeric_limits<double>::infinity()};
};

struct GuideCorridorReport
{
  bool valid{false};
  std::string reason{"CORRIDOR_NOT_RUN"};
  double min_radius{std::numeric_limits<double>::infinity()};
  double min_clearance{std::numeric_limits<double>::infinity()};
  int query_count{0};
};

class GuideCorridor2D
{
public:
  GuideCorridorReport generate(
    const std::vector<Eigen::Vector3d> & guide,
    const std::shared_ptr<EsdfMapInterface> & map,
    double validation_safe_dist,
    double max_radius,
    double min_radius,
    double sample_step);

  bool contains(
    const Eigen::Vector3d & point,
    int previous_segment,
    int * matched_segment,
    double * signed_margin = nullptr) const;

  static double squaredDistanceAndGradient(
    const Eigen::Vector3d & point,
    const Eigen::Vector3d & start,
    const Eigen::Vector3d & end,
    Eigen::Vector3d * gradient);

  const std::vector<GuideCorridorSegment> & segments() const { return segments_; }
  double minRadius() const { return min_radius_; }

private:
  std::vector<GuideCorridorSegment> segments_;
  double min_radius_{std::numeric_limits<double>::infinity()};
};

}  // namespace minco_processor
