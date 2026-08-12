#include "minco_processor/guide_corridor.hpp"

#include <algorithm>
#include <cmath>

namespace minco_processor {

double GuideCorridor2D::squaredDistanceAndGradient(
  const Eigen::Vector3d & point,
  const Eigen::Vector3d & start,
  const Eigen::Vector3d & end,
  Eigen::Vector3d * gradient)
{
  Eigen::Vector3d segment = end - start;
  segment.z() = 0.0;
  Eigen::Vector3d delta = point - start;
  delta.z() = 0.0;
  const double denominator = segment.head<2>().squaredNorm();
  const double ratio = denominator > 1e-12 ?
    std::clamp(delta.head<2>().dot(segment.head<2>()) / denominator, 0.0, 1.0) : 0.0;
  Eigen::Vector3d residual = point - (start + ratio * segment);
  residual.z() = 0.0;
  if (gradient) {
    *gradient = 2.0 * residual;
  }
  return residual.head<2>().squaredNorm();
}

GuideCorridorReport GuideCorridor2D::generate(
  const std::vector<Eigen::Vector3d> & guide,
  const std::shared_ptr<EsdfMapInterface> & map,
  double validation_safe_dist,
  double max_radius,
  double min_radius,
  double sample_step)
{
  segments_.clear();
  min_radius_ = std::numeric_limits<double>::infinity();
  GuideCorridorReport report;
  if (!map || guide.size() < 2U || !std::isfinite(validation_safe_dist) ||
    !std::isfinite(max_radius) || !std::isfinite(min_radius) ||
    validation_safe_dist < 0.0 || max_radius <= 0.0 || min_radius <= 0.0 ||
    min_radius > max_radius)
  {
    report.reason = "CORRIDOR_GENERATION_FAILED";
    return report;
  }
  const double map_step = std::max(1e-3, 0.5 * map->resolution());
  const double step = std::max(1e-3, std::min(sample_step, map_step));
  for (size_t index = 0; index + 1U < guide.size(); ++index) {
    if (!(guide[index].allFinite() && guide[index + 1U].allFinite())) {
      report.reason = "CORRIDOR_GENERATION_FAILED";
      segments_.clear();
      return report;
    }
    const Eigen::Vector3d delta = guide[index + 1U] - guide[index];
    const double length = delta.head<2>().norm();
    if (!(std::isfinite(length) && length > 1e-6)) {
      continue;
    }
    GuideCorridorSegment segment;
    segment.start = guide[index];
    segment.end = guide[index + 1U];
    const int samples = std::max(1, static_cast<int>(std::ceil(length / step)));
    for (int sample = 0; sample <= samples; ++sample) {
      const double ratio = static_cast<double>(sample) / static_cast<double>(samples);
      Eigen::Vector3d point = segment.start + ratio * delta;
      point.z() = 0.0;
      const auto query = map->query(point);
      ++report.query_count;
      if (!query.ok || !std::isfinite(query.distance)) {
        report.reason = "CORRIDOR_GUIDE_OOB";
        segments_.clear();
        return report;
      }
      segment.min_clearance = std::min(segment.min_clearance, query.distance);
      report.min_clearance = std::min(report.min_clearance, query.distance);
      if (query.distance <= validation_safe_dist) {
        report.reason = query.distance < 0.0 ?
          "CORRIDOR_GUIDE_NEGATIVE_ESDF" : "CORRIDOR_GUIDE_UNSAFE";
        segments_.clear();
        return report;
      }
    }
    segment.radius = std::min(max_radius, segment.min_clearance - validation_safe_dist);
    if (!std::isfinite(segment.radius) || segment.radius < min_radius) {
      report.reason = "CORRIDOR_GENERATION_FAILED";
      segments_.clear();
      return report;
    }
    min_radius_ = std::min(min_radius_, segment.radius);
    segments_.push_back(segment);
  }
  if (segments_.empty()) {
    report.reason = "CORRIDOR_GENERATION_FAILED";
    return report;
  }
  report.valid = true;
  report.reason = "NONE";
  report.min_radius = min_radius_;
  return report;
}

bool GuideCorridor2D::contains(
  const Eigen::Vector3d & point,
  int previous_segment,
  int * matched_segment,
  double * signed_margin) const
{
  if (!point.allFinite() || segments_.empty()) {
    return false;
  }
  const int begin = std::max(0, previous_segment - 1);
  const int end = std::min(
    static_cast<int>(segments_.size()), std::max(0, previous_segment) + 2);
  double best_margin = -std::numeric_limits<double>::infinity();
  int best = -1;
  for (int index = begin; index < end; ++index) {
    const auto & segment = segments_[static_cast<size_t>(index)];
    const double distance = std::sqrt(std::max(
      0.0, squaredDistanceAndGradient(point, segment.start, segment.end, nullptr)));
    const double margin = segment.radius - distance;
    if (margin >= -1e-9 && margin > best_margin) {
      best = index;
      best_margin = margin;
    }
  }
  if (best < 0) {
    return false;
  }
  if (matched_segment) {
    *matched_segment = std::max(previous_segment, best);
  }
  if (signed_margin) {
    *signed_margin = best_margin;
  }
  return true;
}

}  // namespace minco_processor
