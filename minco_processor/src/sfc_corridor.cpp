#include "minco_processor/sfc_corridor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace minco_processor {
namespace {

constexpr double kGeometryTolerance = 1e-9;

struct IndexedGuidePoint
{
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};
  int original_index{-1};
};

bool validParameters(
  const std::shared_ptr<EsdfMapInterface> & map,
  const SfcConfig2D & config,
  double safe_dist, double max_radius, double min_radius, double sample_step)
{
  return map && std::isfinite(safe_dist) && std::isfinite(max_radius) &&
    std::isfinite(min_radius) && std::isfinite(sample_step) &&
    std::isfinite(config.bound_distance) &&
    std::isfinite(config.seed_line_max_length) &&
    std::isfinite(config.min_overlap_depth) &&
    std::isfinite(config.obstacle_sample_step) &&
    std::isfinite(config.halfspace_tolerance) &&
    safe_dist >= 0.0 && max_radius > 0.0 && min_radius > 0.0 &&
    min_radius <= max_radius && sample_step > 0.0 &&
    config.bound_distance > 0.0 && config.seed_line_max_length > 0.0 &&
    config.min_overlap_depth >= 0.0 && config.obstacle_sample_step > 0.0 &&
    config.halfspace_tolerance >= 0.0 && config.max_seed_iterations > 0;
}

std::vector<IndexedGuidePoint> simplifyGuide(
  const std::vector<Eigen::Vector3d> & guide)
{
  std::vector<IndexedGuidePoint> simplified;
  simplified.reserve(guide.size());
  for (size_t index = 0; index < guide.size(); ++index) {
    const auto & point = guide[index];
    if (!point.allFinite()) {
      return {};
    }
    if (simplified.empty() ||
      (point - simplified.back().point).head<2>().norm() > 1e-6)
    {
      simplified.push_back({point, static_cast<int>(index)});
    }
  }
  return simplified;
}

double faceMargin(const SfcHalfspace2D & face, const Eigen::Vector2d & point)
{
  return -(face.normal.dot(point) + face.offset);
}

bool insideCell(const SfcCell2D & cell, const Eigen::Vector2d & point, double * margin)
{
  double smallest = std::numeric_limits<double>::infinity();
  for (const auto & face : cell.halfspaces) {
    const double signed_margin = faceMargin(face, point);
    smallest = std::min(smallest, signed_margin);
    if (!std::isfinite(signed_margin) || signed_margin < -kGeometryTolerance) {
      if (margin) {
        *margin = smallest;
      }
      return false;
    }
  }
  if (margin) {
    *margin = smallest;
  }
  return !cell.halfspaces.empty();
}

std::vector<Eigen::Vector2d> clipWithHalfspace(
  const std::vector<Eigen::Vector2d> & polygon,
  const SfcHalfspace2D & face,
  double tolerance)
{
  std::vector<Eigen::Vector2d> output;
  if (polygon.empty()) {
    return output;
  }
  output.reserve(polygon.size() + 1U);
  for (size_t index = 0; index < polygon.size(); ++index) {
    const Eigen::Vector2d & previous =
      polygon[(index + polygon.size() - 1U) % polygon.size()];
    const Eigen::Vector2d & current = polygon[index];
    const double previous_value = face.normal.dot(previous) + face.offset;
    const double current_value = face.normal.dot(current) + face.offset;
    const bool previous_inside = previous_value <= tolerance;
    const bool current_inside = current_value <= tolerance;
    if (previous_inside != current_inside) {
      const double denominator = previous_value - current_value;
      if (std::abs(denominator) > 1e-14) {
        const double ratio = previous_value / denominator;
        output.push_back(previous + ratio * (current - previous));
      }
    }
    if (current_inside) {
      output.push_back(current);
    }
  }
  std::vector<Eigen::Vector2d> unique;
  unique.reserve(output.size());
  for (const auto & point : output) {
    if (unique.empty() || (point - unique.back()).norm() > 1e-10) {
      unique.push_back(point);
    }
  }
  if (unique.size() > 1U && (unique.front() - unique.back()).norm() <= 1e-10) {
    unique.pop_back();
  }
  return unique;
}

double polygonArea(const std::vector<Eigen::Vector2d> & polygon)
{
  if (polygon.size() < 3U) {
    return 0.0;
  }
  double twice_area = 0.0;
  for (size_t index = 0; index < polygon.size(); ++index) {
    const auto & a = polygon[index];
    const auto & b = polygon[(index + 1U) % polygon.size()];
    twice_area += a.x() * b.y() - a.y() * b.x();
  }
  return 0.5 * std::abs(twice_area);
}

Eigen::Vector2d polygonCentroid(const std::vector<Eigen::Vector2d> & polygon)
{
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto & point : polygon) {
    centroid += point;
  }
  return centroid / static_cast<double>(polygon.size());
}

Eigen::Vector2d closestPointOnSeed(
  const Eigen::Vector2d & point,
  const Eigen::Vector2d & start,
  const Eigen::Vector2d & end)
{
  const Eigen::Vector2d delta = end - start;
  const double squared_length = delta.squaredNorm();
  if (squared_length <= 1e-16) {
    return start;
  }
  const double ratio = std::clamp(
    (point - start).dot(delta) / squared_length, 0.0, 1.0);
  return start + ratio * delta;
}

bool segmentIsSafe(
  const Eigen::Vector3d & start,
  const Eigen::Vector3d & end,
  const std::shared_ptr<EsdfMapInterface> & map,
  double safe_dist,
  double sample_step,
  int * query_count,
  double * min_clearance,
  std::string * reason)
{
  const double length = (end - start).head<2>().norm();
  const int count = std::max(1, static_cast<int>(std::ceil(length / sample_step)));
  for (int index = 0; index <= count; ++index) {
    Eigen::Vector3d point = start +
      (static_cast<double>(index) / static_cast<double>(count)) * (end - start);
    point.z() = 0.0;
    const auto query = map->query(point);
    ++(*query_count);
    if (!query.ok || !std::isfinite(query.distance)) {
      *reason = "SFC_GUIDE_OOB";
      return false;
    }
    *min_clearance = std::min(*min_clearance, query.distance);
    if (query.distance <= safe_dist) {
      *reason = query.distance < 0.0 ?
        "SFC_GUIDE_NEGATIVE_ESDF" : "SFC_GUIDE_UNSAFE";
      return false;
    }
  }
  return true;
}

std::vector<Eigen::Vector2d> occupiedCellCentres(
  const std::shared_ptr<EsdfMapInterface> & map,
  const Eigen::Vector2d & lower,
  const Eigen::Vector2d & upper)
{
  std::vector<Eigen::Vector2d> points;
  const double resolution = map->resolution();
  if (!(std::isfinite(resolution) && resolution > 0.0) ||
    map->sizeX() == 0U || map->sizeY() == 0U)
  {
    return points;
  }
  const int min_x = std::max(0, static_cast<int>(std::floor(
    (lower.x() - map->originX()) / resolution)));
  const int min_y = std::max(0, static_cast<int>(std::floor(
    (lower.y() - map->originY()) / resolution)));
  const int max_x = std::min(
    static_cast<int>(map->sizeX()) - 1,
    static_cast<int>(std::floor((upper.x() - map->originX()) / resolution)));
  const int max_y = std::min(
    static_cast<int>(map->sizeY()) - 1,
    static_cast<int>(std::floor((upper.y() - map->originY()) / resolution)));
  for (int y = min_y; y <= max_y; ++y) {
    for (int x = min_x; x <= max_x; ++x) {
      if (map->isFree(static_cast<unsigned int>(x), static_cast<unsigned int>(y))) {
        continue;
      }
      double wx = 0.0;
      double wy = 0.0;
      map->mapToWorld(static_cast<unsigned int>(x), static_cast<unsigned int>(y), wx, wy);
      if (std::isfinite(wx) && std::isfinite(wy)) {
        points.emplace_back(wx, wy);
      }
    }
  }
  return points;
}

bool buildCell(
  const IndexedGuidePoint & start,
  const IndexedGuidePoint & end,
  const std::shared_ptr<EsdfMapInterface> & map,
  const SfcConfig2D & config,
  double safe_dist,
  double max_radius,
  double min_radius,
  double segment_clearance,
  SfcCell2D * cell,
  SfcCorridorReport * report)
{
  const double extent = std::min(config.bound_distance, max_radius);
  const Eigen::Vector2d seed_start = start.point.head<2>();
  const Eigen::Vector2d seed_end = end.point.head<2>();
  const Eigen::Vector2d lower = seed_start.cwiseMin(seed_end) -
    Eigen::Vector2d::Constant(extent);
  const Eigen::Vector2d upper = seed_start.cwiseMax(seed_end) +
    Eigen::Vector2d::Constant(extent);
  std::vector<Eigen::Vector2d> polygon{
    {lower.x(), lower.y()}, {upper.x(), lower.y()},
    {upper.x(), upper.y()}, {lower.x(), upper.y()},
  };
  std::vector<SfcHalfspace2D> faces{
    {Eigen::Vector2d(1.0, 0.0), -upper.x()},
    {Eigen::Vector2d(-1.0, 0.0), lower.x()},
    {Eigen::Vector2d(0.0, 1.0), -upper.y()},
    {Eigen::Vector2d(0.0, -1.0), lower.y()},
  };

  auto obstacles = occupiedCellCentres(map, lower, upper);
  report->obstacle_sample_count += static_cast<int>(obstacles.size());
  std::sort(obstacles.begin(), obstacles.end(), [&](const auto & left, const auto & right) {
      return (left - closestPointOnSeed(left, seed_start, seed_end)).squaredNorm() <
             (right - closestPointOnSeed(right, seed_start, seed_end)).squaredNorm();
    });
  const double map_resolution = std::max(0.0, map->resolution());
  const double inflated_radius = safe_dist + std::sqrt(0.5) * map_resolution;
  for (const auto & obstacle : obstacles) {
    const Eigen::Vector2d closest = closestPointOnSeed(obstacle, seed_start, seed_end);
    Eigen::Vector2d direction = obstacle - closest;
    const double distance = direction.norm();
    if (!(std::isfinite(distance) && distance > inflated_radius + config.halfspace_tolerance)) {
      report->reason = "SFC_GUIDE_UNSAFE";
      return false;
    }
    direction /= distance;
    SfcHalfspace2D face;
    face.normal = direction;
    face.offset = inflated_radius - direction.dot(obstacle);
    if (faceMargin(face, seed_start) < -config.halfspace_tolerance ||
      faceMargin(face, seed_end) < -config.halfspace_tolerance)
    {
      report->reason = "SFC_GUIDE_UNSAFE";
      return false;
    }
    bool already_excluded = true;
    for (const auto & vertex : polygon) {
      if (faceMargin(face, vertex) < -config.halfspace_tolerance) {
        already_excluded = false;
        break;
      }
    }
    if (already_excluded) {
      continue;
    }
    polygon = clipWithHalfspace(polygon, face, config.halfspace_tolerance);
    if (polygon.size() < 3U || polygonArea(polygon) <= config.halfspace_tolerance) {
      report->reason = "SFC_GENERATION_FAILED";
      return false;
    }
    faces.push_back(face);
  }

  cell->seed_start = start.point;
  cell->seed_end = end.point;
  cell->guide_start_index = start.original_index;
  cell->guide_end_index = end.original_index;
  cell->halfspaces = std::move(faces);
  cell->vertices = std::move(polygon);
  cell->clearance_radius = std::min(max_radius, segment_clearance - safe_dist);
  if (!(std::isfinite(cell->clearance_radius) && cell->clearance_radius >= min_radius)) {
    report->reason = "SFC_GENERATION_FAILED";
    return false;
  }
  return true;
}

bool connectCells(
  const SfcCell2D & left,
  SfcCell2D * right,
  const SfcConfig2D & config,
  double * depth)
{
  std::vector<Eigen::Vector2d> intersection = left.vertices;
  for (const auto & face : right->halfspaces) {
    intersection = clipWithHalfspace(intersection, face, config.halfspace_tolerance);
    if (intersection.size() < 3U) {
      return false;
    }
  }
  if (polygonArea(intersection) <= config.halfspace_tolerance) {
    return false;
  }
  const Eigen::Vector2d witness = polygonCentroid(intersection);
  double overlap_depth = std::numeric_limits<double>::infinity();
  for (const auto & face : left.halfspaces) {
    overlap_depth = std::min(overlap_depth, faceMargin(face, witness));
  }
  for (const auto & face : right->halfspaces) {
    overlap_depth = std::min(overlap_depth, faceMargin(face, witness));
  }
  if (!std::isfinite(overlap_depth) ||
    overlap_depth + config.halfspace_tolerance < config.min_overlap_depth)
  {
    return false;
  }
  right->overlap_polygon = std::move(intersection);
  right->overlap_witness = witness;
  right->overlap_depth = overlap_depth;
  right->overlap_with_previous = overlap_depth;
  *depth = overlap_depth;
  return true;
}

}  // namespace

SfcCorridorReport SuperplannerSfc2D::generate(
  const std::vector<Eigen::Vector3d> & guide,
  const std::shared_ptr<EsdfMapInterface> & map,
  double validation_safe_dist, double max_radius, double min_radius, double sample_step)
{
  cells_.clear();
  SfcCorridorReport report;
  const auto simplified = simplifyGuide(guide);
  if (!validParameters(
      map, config_, validation_safe_dist, max_radius, min_radius, sample_step) ||
    simplified.size() < 2U)
  {
    report.reason = "SFC_GENERATION_FAILED";
    return report;
  }
  const double step = std::max(
    1e-3, std::min(sample_step, std::max(1e-3, 0.5 * map->resolution())));
  size_t first = 0U;
  int iterations = 0;
  while (first + 1U < simplified.size()) {
    if (++iterations > config_.max_seed_iterations) {
      report.reason = "SFC_MAX_ITERATIONS";
      cells_.clear();
      return report;
    }
    size_t selected = first;
    double selected_clearance = std::numeric_limits<double>::infinity();
    std::string selected_reason = "SFC_GENERATION_FAILED";
    for (size_t candidate = first + 1U; candidate < simplified.size(); ++candidate) {
      const double length =
        (simplified[candidate].point - simplified[first].point).head<2>().norm();
      if (length > config_.seed_line_max_length + config_.halfspace_tolerance) {
        break;
      }
      double clearance = std::numeric_limits<double>::infinity();
      std::string reason = "NONE";
      if (!segmentIsSafe(
          simplified[first].point, simplified[candidate].point, map,
          validation_safe_dist, step, &report.query_count, &clearance, &reason))
      {
        selected_reason = reason;
        break;
      }
      selected = candidate;
      selected_clearance = clearance;
    }
    if (selected == first) {
      selected = first + 1U;
      selected_clearance = std::numeric_limits<double>::infinity();
      selected_reason = "NONE";
      if (!segmentIsSafe(
          simplified[first].point, simplified[selected].point, map,
          validation_safe_dist, step, &report.query_count,
          &selected_clearance, &selected_reason))
      {
        report.reason = selected_reason;
        cells_.clear();
        return report;
      }
    }
    report.min_clearance = std::min(report.min_clearance, selected_clearance);
    SfcCell2D cell;
    if (!buildCell(
        simplified[first], simplified[selected], map, config_,
        validation_safe_dist, max_radius, min_radius, selected_clearance,
        &cell, &report))
    {
      cells_.clear();
      return report;
    }
    if (!cells_.empty()) {
      double overlap_depth = 0.0;
      if (!connectCells(cells_.back(), &cell, config_, &overlap_depth)) {
        const IndexedGuidePoint junction = simplified[first];
        SfcCell2D repair;
        double junction_clearance = std::numeric_limits<double>::infinity();
        std::string junction_reason = "NONE";
        if (!segmentIsSafe(
            junction.point, junction.point, map, validation_safe_dist, step,
            &report.query_count, &junction_clearance, &junction_reason) ||
          !buildCell(
            junction, junction, map, config_, validation_safe_dist,
            max_radius, min_radius, junction_clearance, &repair, &report))
        {
          report.reason = junction_reason == "NONE" ?
            "SFC_DISCONNECTED" : junction_reason;
          cells_.clear();
          return report;
        }
        double repair_left_depth = 0.0;
        double repair_right_depth = 0.0;
        if (!connectCells(cells_.back(), &repair, config_, &repair_left_depth) ||
          !connectCells(repair, &cell, config_, &repair_right_depth))
        {
          report.reason = "SFC_DISCONNECTED";
          cells_.clear();
          return report;
        }
        report.min_overlap = std::min(
          report.min_overlap, std::min(repair_left_depth, repair_right_depth));
        cells_.push_back(std::move(repair));
        ++report.repair_cell_count;
      } else {
        report.min_overlap = std::min(report.min_overlap, overlap_depth);
      }
    }
    report.min_radius = std::min(report.min_radius, cell.clearance_radius);
    cells_.push_back(std::move(cell));
    first = selected;
  }
  if (cells_.empty()) {
    report.reason = "SFC_GENERATION_FAILED";
    return report;
  }
  if (cells_.size() == 1U) {
    report.min_overlap = cells_.front().clearance_radius;
  }
  for (size_t index = 0; index < cells_.size(); ++index) {
    cells_[index].cell_index = static_cast<int>(index);
    report.piece_bindings.push_back(
      {static_cast<int>(index), static_cast<int>(index)});
  }
  report.valid = true;
  report.reason = "NONE";
  return report;
}

bool SuperplannerSfc2D::contains(
  const Eigen::Vector3d & point, int * matched_cell, double * signed_margin) const
{
  if (!point.allFinite() || cells_.empty()) {
    return false;
  }
  double best_margin = -std::numeric_limits<double>::infinity();
  int best = -1;
  for (size_t index = 0; index < cells_.size(); ++index) {
    double margin = -std::numeric_limits<double>::infinity();
    if (insideCell(cells_[index], point.head<2>(), &margin) && margin > best_margin) {
      best_margin = margin;
      best = static_cast<int>(index);
    }
  }
  if (best < 0) {
    if (signed_margin) {
      *signed_margin = best_margin;
    }
    return false;
  }
  if (matched_cell) {
    *matched_cell = best;
  }
  if (signed_margin) {
    *signed_margin = best_margin;
  }
  return true;
}

bool SuperplannerSfc2D::containsInCell(
  const Eigen::Vector3d & point, int cell_index, double * signed_margin) const
{
  if (!point.allFinite() || cell_index < 0 ||
    cell_index >= static_cast<int>(cells_.size()))
  {
    if (signed_margin) {
      *signed_margin = -std::numeric_limits<double>::infinity();
    }
    return false;
  }
  return insideCell(
    cells_[static_cast<size_t>(cell_index)], point.head<2>(), signed_margin);
}

}  // namespace minco_processor
