#include "minco_processor/minco_pipeline.hpp"

#include "minco_core/components/trajectory_safety_checker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>

namespace minco_processor {

namespace {

template <typename T>
T clampValue(T value, T low, T high)
{
  return std::max(low, std::min(value, high));
}

double curvatureDecay(
  double angle, double global_vmax, double deadzone, double saturation, double min_vel, double decay_power)
{
  if (angle < deadzone) {
    return global_vmax;
  }
  const double clamped_angle = std::min(angle, saturation);
  const double ratio = (clamped_angle - deadzone) / saturation;
  return std::max(min_vel, min_vel + (global_vmax - min_vel) * std::exp(-ratio * decay_power));
}

double limitLocalVel(const std::vector<Eigen::Vector3d> & sparse_path,
  int seg_idx,
  double global_vmax,
  double deadzone,
  double saturation,
  double min_turn_vel,
  double decay_power)
{
  if (!(std::isfinite(global_vmax) && global_vmax > 0.0)) {
    return 0.0;
  }
  const int n = static_cast<int>(sparse_path.size());
  if (seg_idx < 0 || seg_idx + 2 >= n) {
    return global_vmax;
  }
  const Eigen::Vector2d d1 =
    (sparse_path[static_cast<size_t>(seg_idx + 1)] - sparse_path[static_cast<size_t>(seg_idx)]).head<2>();
  const Eigen::Vector2d d2 =
    (sparse_path[static_cast<size_t>(seg_idx + 2)] - sparse_path[static_cast<size_t>(seg_idx + 1)])
      .head<2>();
  const double n1 = d1.norm();
  const double n2 = d2.norm();
  if (n1 <= 0.2 || n2 <= 0.2) {
    return global_vmax;
  }
  const double dot = clampValue(d1.dot(d2) / (n1 * n2), -1.0, 1.0);
  return curvatureDecay(std::acos(dot), global_vmax, deadzone, saturation, min_turn_vel, decay_power);
}

double computeNextSpeed(double v_curr, double seg_len, double remain_after, double amax, double local_vmax)
{
  if (!(std::isfinite(v_curr) && v_curr >= 0.0)) {
    v_curr = 0.0;
  }
  if (!(std::isfinite(seg_len) && seg_len > 0.0)) {
    return 0.0;
  }
  const double a_safe = std::max(1e-3, std::abs(amax));
  double v_next = std::sqrt(std::max(0.0, v_curr * v_curr + 2.0 * a_safe * seg_len));
  if (std::isfinite(local_vmax) && local_vmax > 0.0) {
    v_next = std::min(v_next, local_vmax);
  }
  const double v_cap_stop =
    (remain_after > 1e-6) ? std::sqrt(std::max(0.0, 2.0 * a_safe * remain_after)) : 0.0;
  v_next = std::min(v_next, v_cap_stop);
  return (std::isfinite(v_next) && v_next >= 0.0) ? v_next : 0.0;
}

double computeSegmentTime(
  double seg_len, double v_curr, double v_next, double local_vmax, double amax, double min_seg_time)
{
  const double t_min = std::max(1e-3, min_seg_time);
  const double v_sum = v_curr + v_next;
  double t = v_sum > 1e-3 ? 2.0 * seg_len / v_sum : seg_len / 0.1;
  const double a_safe = std::max(1e-3, std::abs(amax));
  const double required_dist = std::abs(v_curr * v_curr - v_next * v_next) / (2.0 * a_safe);
  if (seg_len >= required_dist) {
    t = std::max(t, std::abs(v_curr - v_next) / a_safe);
  }
  if (local_vmax > 1e-6 && std::abs(v_next - local_vmax) < 1e-6 && v_curr < local_vmax - 1e-6) {
    const double d_acc = (local_vmax * local_vmax - v_curr * v_curr) / (2.0 * a_safe);
    if (std::isfinite(d_acc) && d_acc > 0.0 && seg_len > d_acc) {
      t = std::max(t, (local_vmax - v_curr) / a_safe + (seg_len - d_acc) / local_vmax);
    }
  }
  return (std::isfinite(t) && t >= t_min) ? t : t_min;
}

void propagateVelocityLimits(const std::vector<double> & seg_len, double amax, std::vector<double> & local_vmaxs)
{
  if (seg_len.empty() || local_vmaxs.empty() || seg_len.size() != local_vmaxs.size()) {
    return;
  }
  const int n = static_cast<int>(seg_len.size());
  const double a_safe = std::max(1e-3, std::abs(amax));
  for (int i = n - 2; i >= 0; --i) {
    const double L = std::max(0.0, seg_len[static_cast<size_t>(i)]);
    const double v_next = std::max(0.0, local_vmaxs[static_cast<size_t>(i + 1)]);
    const double v_cap = std::sqrt(std::max(0.0, v_next * v_next + 2.0 * a_safe * L));
    local_vmaxs[static_cast<size_t>(i)] =
      std::min(std::max(0.0, local_vmaxs[static_cast<size_t>(i)]), v_cap);
  }
  for (int i = 1; i < n; ++i) {
    const double L = std::max(0.0, seg_len[static_cast<size_t>(i - 1)]);
    const double v_prev = std::max(0.0, local_vmaxs[static_cast<size_t>(i - 1)]);
    const double v_cap = std::sqrt(std::max(0.0, v_prev * v_prev + 2.0 * a_safe * L));
    local_vmaxs[static_cast<size_t>(i)] =
      std::min(std::max(0.0, local_vmaxs[static_cast<size_t>(i)]), v_cap);
  }
}

double getDistFromTrapezoid(
  double t, double total_length, double a_ref, double v_peak, double t_acc, double t_flat, double t_dec)
{
  const double t_total = t_acc + std::max(0.0, t_flat) + std::max(0.0, t_dec);
  if (t >= t_total) {
    return total_length;
  }
  const double d_acc = 0.5 * v_peak * v_peak / a_ref;
  const double d_flat = v_peak * std::max(0.0, t_flat);
  if (t <= t_acc) {
    const double v = a_ref * t;
    return clampValue(0.5 * v * v / a_ref, 0.0, total_length);
  }
  if (t <= t_acc + t_flat) {
    return clampValue(d_acc + v_peak * (t - t_acc), 0.0, total_length);
  }
  const double dt = t - (t_acc + t_flat);
  const double v = std::max(0.0, v_peak - a_ref * dt);
  return clampValue(d_acc + d_flat + (v_peak + v) * 0.5 * dt, 0.0, total_length);
}

struct SparseWaypointBuild
{
  std::vector<Eigen::Vector3d> waypoints;
  std::vector<size_t> source_indices;
  std::vector<bool> mandatory;
  int mandatory_corner_count{0};
};

size_t findCornerIndex(const std::vector<Eigen::Vector3d> & path, size_t start_idx, size_t end_idx)
{
  const Eigen::Vector2d a = path[start_idx].head<2>();
  const Eigen::Vector2d b = path[end_idx].head<2>();
  const Eigen::Vector2d ab = b - a;
  const double ab2 = ab.squaredNorm();
  if (end_idx <= start_idx + 1U || ab2 <= 1e-12) {
    return (start_idx + end_idx) / 2U;
  }
  double best_dist2 = -1.0;
  size_t best_k = (start_idx + end_idx) / 2U;
  for (size_t k = start_idx + 1U; k < end_idx; ++k) {
    const Eigen::Vector2d p = path[k].head<2>();
    const double ratio = clampValue((p - a).dot(ab) / ab2, 0.0, 1.0);
    const double dist2 = (p - (a + ratio * ab)).squaredNorm();
    if (dist2 > best_dist2) {
      best_dist2 = dist2;
      best_k = k;
    }
  }
  return best_k;
}

void appendSparsePoint(SparseWaypointBuild & build, const std::vector<Eigen::Vector3d> & path, size_t idx, bool mandatory)
{
  if (idx >= path.size()) {
    return;
  }
  Eigen::Vector3d p = path[idx];
  p.z() = 0.0;
  if (!build.waypoints.empty() && (p - build.waypoints.back()).head<2>().norm() <= 1e-6) {
    if (mandatory && !build.mandatory.empty()) {
      build.mandatory.back() = true;
    }
    return;
  }
  build.waypoints.push_back(p);
  build.source_indices.push_back(idx);
  build.mandatory.push_back(mandatory);
  if (mandatory && idx != 0U && idx + 1U != path.size()) {
    ++build.mandatory_corner_count;
  }
}

SparseWaypointBuild getSparseWaypoints(const std::vector<Eigen::Vector3d> & path,
  double max_vel,
  double max_acc,
  bool goal_reached,
  const std::function<bool(const Eigen::Vector3d &, const Eigen::Vector3d &)> & is_line_free)
{
  SparseWaypointBuild sparse;
  if (path.empty()) {
    return sparse;
  }
  appendSparsePoint(sparse, path, 0U, true);
  if (path.size() < 2U) {
    return sparse;
  }
  if (path.size() == 2U) {
    appendSparsePoint(sparse, path, path.size() - 1U, true);
    return sparse;
  }

  std::vector<double> accumulated_dist(path.size(), 0.0);
  for (size_t i = 1; i < path.size(); ++i) {
    double seg_len = (path[i] - path[i - 1U]).head<2>().norm();
    if (i > 1U) {
      const Eigen::Vector3d v1 = path[i] - path[i - 1U];
      const Eigen::Vector3d v2 = path[i - 1U] - path[i - 2U];
      if (v1.norm() > 1e-3 && v2.norm() > 1e-3 && v1.normalized().dot(v2.normalized()) < 0.7) {
        seg_len *= 2.0;
      }
    }
    accumulated_dist[i] = accumulated_dist[i - 1U] + seg_len;
  }
  const double total_length = accumulated_dist.back();
  if (!(std::isfinite(total_length) && total_length > 1e-3)) {
    appendSparsePoint(sparse, path, path.size() - 1U, true);
    return sparse;
  }

  const double v_ref = std::max(1e-6, max_vel);
  const double a_ref = std::max(1e-6, max_acc);
  const double d_acc_ref = v_ref * v_ref / (2.0 * a_ref);
  double v_peak = v_ref;
  double t_acc = v_ref / a_ref;
  double t_flat = 0.0;
  double t_dec = goal_reached ? t_acc : 0.0;
  const double d_dec = goal_reached ? d_acc_ref : 0.0;
  if (total_length > d_acc_ref + d_dec) {
    t_flat = (total_length - (d_acc_ref + d_dec)) / v_ref;
  } else if (goal_reached) {
    v_peak = std::sqrt(std::max(0.0, total_length * a_ref));
    t_acc = v_peak / a_ref;
    t_dec = t_acc;
  } else {
    v_peak = std::min(v_ref, std::sqrt(std::max(0.0, 2.0 * total_length * a_ref)));
    t_acc = v_peak / a_ref;
    t_dec = 0.0;
    const double d_acc = 0.5 * v_peak * v_peak / a_ref;
    t_flat = v_peak >= v_ref && total_length > d_acc ? (total_length - d_acc) / v_ref : 0.0;
  }

  const double t_total = t_acc + t_flat + std::max(0.0, t_dec);
  if (!(std::isfinite(t_total) && t_total > 1e-6)) {
    appendSparsePoint(sparse, path, path.size() - 1U, true);
    return sparse;
  }

  auto arcLengthToIndex = [&accumulated_dist](double s) -> size_t {
    const double s_clamped = clampValue(s, 0.0, accumulated_dist.back());
    auto it = std::lower_bound(accumulated_dist.begin(), accumulated_dist.end(), s_clamped);
    if (it == accumulated_dist.begin()) {
      return 0U;
    }
    if (it == accumulated_dist.end()) {
      return accumulated_dist.size() - 1U;
    }
    const size_t idx1 = static_cast<size_t>(std::distance(accumulated_dist.begin(), it));
    const size_t idx0 = idx1 - 1U;
    return ((s_clamped - accumulated_dist[idx0]) < (accumulated_dist[idx1] - s_clamped)) ? idx0 : idx1;
  };

  const int n_segments = std::max(4, static_cast<int>(std::ceil(t_total / 0.5)));
  const double dt = t_total / static_cast<double>(n_segments);
  std::vector<size_t> target_indices;
  target_indices.push_back(0U);
  size_t last_added = 0U;
  for (int i = 1; i < n_segments; ++i) {
    const double s = getDistFromTrapezoid(i * dt, total_length, a_ref, v_peak, t_acc, t_flat, t_dec);
    size_t idx = std::max(arcLengthToIndex(s), last_added);
    if (idx == last_added && idx + 1U < path.size()) {
      ++idx;
    }
    if (idx > last_added && idx < path.size()) {
      target_indices.push_back(idx);
      last_added = idx;
    }
  }
  if (target_indices.back() != path.size() - 1U) {
    target_indices.push_back(path.size() - 1U);
  }

  size_t current_safe_idx = 0U;
  for (size_t ti = 1U; ti < target_indices.size(); ++ti) {
    const size_t target_idx = target_indices[ti];
    size_t guard = 0U;
    while (guard++ < 32U && target_idx > current_safe_idx) {
      if (!is_line_free || is_line_free(path[current_safe_idx], path[target_idx])) {
        appendSparsePoint(sparse, path, target_idx, target_idx == path.size() - 1U);
        current_safe_idx = target_idx;
        break;
      }
      size_t corner_idx = findCornerIndex(path, current_safe_idx, target_idx);
      if (corner_idx <= current_safe_idx || corner_idx >= target_idx) {
        corner_idx = current_safe_idx + 1U;
        if (corner_idx >= target_idx) {
          break;
        }
      }
      appendSparsePoint(sparse, path, corner_idx, true);
      current_safe_idx = corner_idx;
    }
  }
  if ((path.back() - sparse.waypoints.back()).head<2>().norm() > 1e-6) {
    appendSparsePoint(sparse, path, path.size() - 1U, true);
  }
  return sparse;
}

}  // namespace

MincoPipeline::Config::Config()
{
  optimizer.max_vel = 2.0;
  optimizer.max_acc = 4.0;
  optimizer.safe_dist = 0.3;
  optimizer.magnitudeBounds.resize(3);
  optimizer.magnitudeBounds << optimizer.safe_dist, optimizer.max_vel, optimizer.max_acc;
  optimizer.penaltyWeights.resize(5);
  optimizer.penaltyWeights << 1000.0, 1000.0, 10000.0, 20.0, 100.0;
}

MincoPipeline::MincoPipeline() : MincoPipeline(Config{})
{
}

MincoPipeline::MincoPipeline(const Config & config)
{
  setConfig(config);
}

void MincoPipeline::setConfig(const Config & config)
{
  config_ = config;
  if (!std::isfinite(config_.max_yaw_rate) || config_.max_yaw_rate <= 0.0) {
    config_.max_yaw_rate = 0.5;
  }
  if (config_.optimizer.magnitudeBounds.size() != 3) {
    config_.optimizer.magnitudeBounds.resize(3);
  }
  config_.optimizer.magnitudeBounds << config_.optimizer.safe_dist, config_.optimizer.max_vel,
    config_.optimizer.max_acc;
  SfcConfig2D sfc_config;
  sfc_config.bound_distance = std::max(1e-3, config_.sfc_bound_distance);
  sfc_config.seed_line_max_length = std::max(1e-3, config_.sfc_seed_line_max_length);
  sfc_config.min_overlap_depth = std::max(0.0, config_.sfc_min_overlap_depth);
  sfc_config.obstacle_sample_step = std::max(1e-3, config_.corridor_sample_step);
  sfc_.setConfig(sfc_config);
  const bool safe_corridor =
    config_.constraint_profile == "safe_corridor_v1" ||
    config_.constraint_profile == "superplanner_sfc_v1";
  if (safe_corridor) {
    if (config_.optimizer.penaltyWeights.size() < 5) {
      config_.optimizer.penaltyWeights.resize(6);
      config_.optimizer.penaltyWeights.head(5) <<
        1000.0, 1000.0, 10000.0, 1000.0, 100.0;
    } else {
      config_.optimizer.penaltyWeights.conservativeResize(6);
    }
    config_.optimizer.penaltyWeights(5) = std::max(0.0, config_.guide_corridor_weight);
  } else if (config_.optimizer.penaltyWeights.size() != 5) {
    config_.optimizer.penaltyWeights.resize(5);
    config_.optimizer.penaltyWeights << 1000.0, 1000.0, 10000.0, 1000.0, 100.0;
  }
  optimizer_ = std::make_unique<minco_planner::MincoOptimizer>(config_.optimizer);
  yaw_optimizer_ = std::make_unique<traj_opt::YawTrajOpt>(config_.max_yaw_rate);
  safety_checker_ = std::make_unique<minco_planner::TrajectorySafetyChecker>();
  safety_checker_->configure(
    config_.validation_safe_dist,
    config_.safety_sample_dt,
    config_.start_validation_exemption_radius);
  setMap(map_);
}

void MincoPipeline::setMap(std::shared_ptr<EsdfMapInterface> map)
{
  map_ = std::move(map);
  if (optimizer_) {
    optimizer_->setMap(map_);
  }
  if (safety_checker_) {
    safety_checker_->setQuery(map_);
  }
}

void MincoPipeline::resetHistory()
{
  has_last_traj_ = false;
  has_last_yaw_traj_ = false;
  last_traj_.clear();
  last_yaw_traj_.clear();
  if (optimizer_) {
    optimizer_->setInitPsAndTs(super_utils::vec_Vec3f{}, super_utils::VecDf{});
  }
}

bool MincoPipeline::commitHistory(const Result & proposal, double applied_time)
{
  if (!proposal.success || proposal.samples.size() < 2U || !std::isfinite(applied_time)) {
    return false;
  }
  last_traj_ = proposal.trajectory;
  last_yaw_traj_ = proposal.yaw_trajectory;
  last_traj_.start_WT = applied_time;
  last_yaw_traj_.start_WT = applied_time;
  has_last_traj_ = true;
  has_last_yaw_traj_ = !proposal.yaw_trajectory.empty();
  return true;
}

MincoPipeline::Result MincoPipeline::optimize(const Request & request)
{
  using Clock = std::chrono::steady_clock;

  Result result;
  result.constraint_profile = config_.constraint_profile;
  result.optimization_safe_dist = config_.optimizer.safe_dist;
  result.validation_safe_dist = config_.validation_safe_dist;
  const auto pipeline_start = Clock::now();
  auto elapsedMs = [](const Clock::time_point & start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  auto finish = [&]() {
    result.dense_path_size = static_cast<int>(result.dense_path.size());
    result.sparse_waypoint_size = static_cast<int>(result.sparse_waypoints.size());
    if (optimizer_) {
      result.optimizer_iteration_count = optimizer_->lastIterationCount();
    }
    result.timing_ms["pipeline_total_ms"] = elapsedMs(pipeline_start);
    return result;
  };

  if (!optimizer_ || request.guide_path.size() < 2U) {
    result.failure_reason = "INVALID_INPUT";
    return finish();
  }

  auto stage_start = Clock::now();
  result.dense_path = extractLocalPath(request.guide_path, request.current.position);
  result.timing_ms["extract_local_path_ms"] = elapsedMs(stage_start);
  if (result.dense_path.size() < 2U) {
    result.failure_reason = "LOCAL_PATH_FAILED";
    return finish();
  }
  const bool local_end_is_goal =
    request.has_terminal_goal && request.terminal_goal.allFinite() &&
    (request.terminal_goal - result.dense_path.back()).head<2>().norm() <= config_.traj_goal_tolerance;
  result.local_end_is_goal = local_end_is_goal;
  if (config_.constraint_profile == "safe_corridor_v1") {
    stage_start = Clock::now();
    const auto corridor_report = corridor_.generate(
      result.dense_path,
      map_,
      config_.validation_safe_dist,
      config_.corridor_max_radius,
      config_.corridor_min_radius,
      config_.corridor_sample_step);
    result.timing_ms["corridor_generation_ms"] = elapsedMs(stage_start);
    result.corridor_failure_reason = corridor_report.reason;
    result.corridor_segment_count = static_cast<int>(corridor_.segments().size());
    result.corridor_segments = corridor_.segments();
    result.corridor_min_radius = corridor_report.min_radius;
    result.corridor_min_clearance = corridor_report.min_clearance;
    if (!corridor_report.valid) {
      result.failure_reason = corridor_report.reason;
      return finish();
    }
    result.corridor_min_overlap = std::numeric_limits<double>::infinity();
    const auto & segments = corridor_.segments();
    const double junction_tolerance = std::max(1e-6, map_->resolution());
    for (size_t index = 0; index + 1U < segments.size(); ++index) {
      const double gap =
        (segments[index].end - segments[index + 1U].start).head<2>().norm();
      if (!std::isfinite(gap) || gap > junction_tolerance) {
        result.corridor_failure_reason = "CORRIDOR_DISCONNECTED";
        result.failure_reason = result.corridor_failure_reason;
        return finish();
      }
      result.corridor_min_overlap = std::min(
        result.corridor_min_overlap,
        std::min(segments[index].radius, segments[index + 1U].radius));
    }
    if (segments.size() == 1U) {
      result.corridor_min_overlap = segments.front().radius;
    }
    if (!(std::isfinite(result.corridor_min_overlap) &&
      result.corridor_min_overlap >= 0.0))
    {
      result.corridor_failure_reason = "CORRIDOR_DISCONNECTED";
      result.failure_reason = result.corridor_failure_reason;
      return finish();
    }
    optimizer_->setGuideCorridor(result.dense_path, corridor_.minRadius());
    // A processor instance may switch profiles between planning calls.  Do not
    // retain cells from a previous SuperPlanner SFC call when the historic
    // capsule profile is requested.
    optimizer_->setSuperplannerSfc({});
  } else if (config_.constraint_profile == "superplanner_sfc_v1") {
    stage_start = Clock::now();
    const auto sfc_report = sfc_.generate(
      result.dense_path,
      map_,
      config_.validation_safe_dist,
      config_.sfc_bound_distance,
      config_.corridor_min_radius,
      config_.corridor_sample_step);
    result.timing_ms["sfc_generation_ms"] = elapsedMs(stage_start);
    result.corridor_schema_version = "superplanner_sfc_2d_v2";
    result.sfc_generation_reason = sfc_report.reason;
    result.sfc_cells = sfc_.cells();
    result.sfc_piece_bindings = sfc_report.piece_bindings;
    result.sfc_obstacle_sample_count = sfc_report.obstacle_sample_count;
    result.sfc_repair_cell_count = sfc_report.repair_cell_count;
    result.sfc_min_overlap = sfc_report.min_overlap;
    result.corridor_failure_reason = sfc_report.reason;
    result.corridor_segment_count = static_cast<int>(result.sfc_cells.size());
    result.corridor_min_clearance = sfc_report.min_clearance;
    result.corridor_min_radius = sfc_report.min_radius;
    if (!sfc_report.valid) {
      result.failure_reason = sfc_report.reason;
      return finish();
    }
    // The historic capsule term is deliberately disabled for the SFC profile.
    // SFC containment is a mandatory acceptance criterion in validateTrajectory.
    optimizer_->setGuideCorridor({}, 0.0);
    optimizer_->setSuperplannerSfc({});
  } else {
    optimizer_->setGuideCorridor({}, 0.0);
    optimizer_->setSuperplannerSfc({});
    result.corridor_failure_reason = "DISABLED";
  }
  stage_start = Clock::now();
  if (config_.constraint_profile == "superplanner_sfc_v1") {
    result.sparse_waypoints.reserve(result.sfc_cells.size() + 1U);
    result.sparse_waypoints.push_back(result.dense_path.front());
    for (size_t cell_index = 1U; cell_index < result.sfc_cells.size(); ++cell_index) {
      Eigen::Vector3d junction = result.dense_path.front();
      junction.head<2>() = result.sfc_cells[cell_index].overlap_witness;
      result.sparse_waypoints.push_back(junction);
    }
    result.sparse_waypoints.push_back(result.dense_path.back());
    result.mandatory_corner_count = std::max(
      0, static_cast<int>(result.sparse_waypoints.size()) - 2);
  } else {
    result.sparse_waypoints = sparsifyPath(
      result.dense_path, local_end_is_goal, &result.mandatory_corner_count);
  }
  result.timing_ms["sparsify_path_ms"] = elapsedMs(stage_start);
  if (result.sparse_waypoints.size() < 2U) {
    result.failure_reason = "SPARSIFY_FAILED";
    return finish();
  }

  stage_start = Clock::now();
  result.planning_state = determinePlanningState(request.current, result.sparse_waypoints, request.now, &result);
  const double history_sample_t = has_last_traj_ ? request.now - last_traj_.start_WT : 0.0;
  if (result.planning_state == PlanningState::kHotStart) {
    prepareHotStart(request.current, history_sample_t, result.start_state);
  } else {
    prepareColdStart(request.current, result.start_state);
  }
  optimizer_->setInitPsAndTs(super_utils::vec_Vec3f{}, super_utils::VecDf{});

  result.end_state.setZero();
  result.end_state.col(0) = result.sparse_waypoints.back();
  if (!local_end_is_goal) {
    Eigen::Vector3d tangent = Eigen::Vector3d::UnitX();
    if (result.sparse_waypoints.size() >= 2U) {
      tangent = result.sparse_waypoints.back() - result.sparse_waypoints[result.sparse_waypoints.size() - 2U];
      tangent.z() = 0.0;
      const double n = tangent.head<2>().norm();
      if (n > 0.1) {
        tangent /= n;
      } else {
        tangent = Eigen::Vector3d::UnitX();
      }
    }
    const double v_curr = std::max(0.0, result.start_state.col(1).head<2>().norm());
    const double amax = std::max(0.0, config_.optimizer.max_acc);
    const double last_seg_len =
      (result.sparse_waypoints.back() - result.sparse_waypoints[result.sparse_waypoints.size() - 2U]).head<2>().norm();
    const double v_max_kinematic = std::sqrt(std::max(0.0, v_curr * v_curr + 2.0 * amax * last_seg_len));
    double local_end_vmax = config_.optimizer.max_vel;
    if (result.sparse_waypoints.size() >= 3U) {
      local_end_vmax = limitLocalVel(result.sparse_waypoints,
        static_cast<int>(result.sparse_waypoints.size()) - 3,
        config_.optimizer.max_vel,
        config_.optimizer.turn_angle_deadzone,
        config_.optimizer.turn_angle_saturation,
        config_.optimizer.min_turn_vel,
        config_.optimizer.decay_power);
    }
    result.end_state.col(1) =
      tangent * std::min({config_.optimizer.max_vel, v_max_kinematic, local_end_vmax});
  }

  while (config_.constraint_profile != "superplanner_sfc_v1" &&
         result.sparse_waypoints.size() > 2U &&
         (result.sparse_waypoints[1] - result.start_state.col(0)).norm() < 0.2) {
    result.sparse_waypoints.erase(result.sparse_waypoints.begin() + 1);
  }
  result.timing_ms["state_prepare_ms"] = elapsedMs(stage_start);

  const int N = static_cast<int>(result.sparse_waypoints.size()) - 1;
  if (config_.constraint_profile == "superplanner_sfc_v1") {
    std::vector<int> piece_to_cell;
    piece_to_cell.reserve(result.sfc_piece_bindings.size());
    for (const auto & binding : result.sfc_piece_bindings) {
      if (binding.piece_index != static_cast<int>(piece_to_cell.size())) {
        result.failure_reason = "SFC_BINDING_INVALID";
        return finish();
      }
      piece_to_cell.push_back(binding.cell_index);
    }
    if (N != static_cast<int>(result.sfc_cells.size()) ||
      !optimizer_->setSuperplannerSfc(result.sfc_cells, piece_to_cell))
    {
      result.failure_reason = "SFC_BINDING_INVALID";
      return finish();
    }
  }
  result.local_vmaxs.resize(N);
  result.initial_times.resize(N);
  super_utils::vec_Vec3f shifted_waypoints;
  super_utils::VecDf shifted_durations;
  if (result.planning_state == PlanningState::kHotStart && N > 0) {
    const double remaining = last_traj_.getTotalDuration() - history_sample_t;
    if (std::isfinite(remaining) && remaining > 0.02) {
      shifted_waypoints.reserve(static_cast<size_t>(N + 1));
      shifted_durations.resize(N);
      const double segment_dt = remaining / static_cast<double>(N);
      for (int i = 0; i <= N; ++i) {
        const double sample_t = std::min(
          last_traj_.getTotalDuration(), history_sample_t + segment_dt * static_cast<double>(i));
        shifted_waypoints.emplace_back(last_traj_.getPos(sample_t));
        if (i < N) {
          shifted_durations(i) = segment_dt;
        }
      }
      result.shifted_seed_valid = shifted_waypoints.size() >= 2U && shifted_durations.size() == N;
    }
  }
  if (result.shifted_seed_valid) {
    result.copied_waypoints = std::min(
      std::max(0, N - 1), std::max(0, static_cast<int>(shifted_waypoints.size()) - 2));
    result.copied_durations = std::min(N, static_cast<int>(shifted_durations.size()));
  }
  stage_start = Clock::now();
  allocatePathTime(result.sparse_waypoints,
    result.start_state,
    local_end_is_goal,
    result.planning_state,
    result.shifted_seed_valid,
    shifted_waypoints,
    shifted_durations,
    result.initial_points,
    result.initial_times,
    result.local_vmaxs);
  result.timing_ms["allocate_time_ms"] = elapsedMs(stage_start);
  optimizer_->setInitPsAndTs(result.initial_points, result.initial_times);

  stage_start = Clock::now();
  result.objective = optimizer_->optimize(
    result.sparse_waypoints, result.start_state, result.end_state, result.local_vmaxs, result.trajectory);
  result.timing_ms["optimizer_ms"] = elapsedMs(stage_start);
  result.optimizer_return_code = optimizer_->lastReturnCode();
  result.optimizer_iteration_count = optimizer_->lastIterationCount();
  const auto & penalty_log = optimizer_->lastPenaltyLog();
  auto penalty_value = [&penalty_log](int index) {
      return index >= 0 && index < penalty_log.size() ? penalty_log(index) : 0.0;
    };
  const bool has_guide_term = penalty_log.size() > 6;
  result.penalty_terms = {
    {"energy", penalty_value(0)},
    {"position", penalty_value(1)},
    {"velocity", penalty_value(2)},
    {"acceleration", penalty_value(3)},
    {"attractor", penalty_value(4)},
    {"guide_corridor", has_guide_term ? penalty_value(5) : 0.0},
    {"time_barrier", penalty_value(has_guide_term ? 6 : 5)},
  };
  if (!std::isfinite(result.objective)) {
    result.failure_reason = "OPTIMIZER_FAILED";
    return finish();
  }
  stage_start = Clock::now();
  if (!validateTrajectory(result.trajectory, result.end_state.col(0), &result)) {
    result.timing_ms["validate_ms"] = elapsedMs(stage_start);
    if (config_.constraint_profile == "safe_corridor_v1" ||
        config_.constraint_profile == "superplanner_sfc_v1") {
      result.timing_ms["adaptive_validation_ms"] = result.timing_ms["validate_ms"];
    }
    result.failure_reason = result.validation_failure_reason;
    return finish();
  }
  result.timing_ms["validate_ms"] = elapsedMs(stage_start);
  if (config_.constraint_profile == "safe_corridor_v1" ||
      config_.constraint_profile == "superplanner_sfc_v1") {
    result.timing_ms["adaptive_validation_ms"] = result.timing_ms["validate_ms"];
  }

  const double goal_yaw = std::isfinite(request.goal_yaw) ? request.goal_yaw : request.current.yaw;
  stage_start = Clock::now();
  if (!optimizeYaw(result.start_state,
        result.trajectory,
        result.yaw_trajectory,
        result.planning_state,
        request.current.yaw,
        request.current.yaw_rate,
        goal_yaw,
        request.now)) {
    Eigen::MatrixXd cMat(3, 6);
    cMat.setZero();
    cMat(0, 5) = std::isfinite(request.current.yaw) ? request.current.yaw : 0.0;
    result.yaw_trajectory.clear();
    result.yaw_trajectory.emplace_back(std::max(0.02, result.trajectory.getTotalDuration()), cMat);
  }
  result.timing_ms["yaw_ms"] = elapsedMs(stage_start);
  if (config_.enable_yaw_wheel_validation ||
      config_.constraint_profile == "safe_corridor_v1" ||
      config_.constraint_profile == "superplanner_sfc_v1") {
    if (!validateYawAndWheels(result.trajectory, result.yaw_trajectory, &result))
    {
      result.failure_reason = result.validation_failure_reason;
      return finish();
    }
  }

  result.trajectory.start_WT = request.now;
  result.yaw_trajectory.start_WT = request.now;
  stage_start = Clock::now();
  result.samples =
    sampleTrajectory(result.trajectory, result.yaw_trajectory, config_.sample_dt, request.current.yaw);
  result.timing_ms["sample_ms"] = elapsedMs(stage_start);
  result.success = true;
  result.failure_reason = "NONE";
  return finish();
}

std::vector<Eigen::Vector3d> MincoPipeline::extractLocalPath(
  const std::vector<Eigen::Vector3d> & guide_path, const Eigen::Vector3d & current_position) const
{
  std::vector<Eigen::Vector3d> local_segment;
  if (guide_path.empty()) {
    return local_segment;
  }
  size_t start_idx = 0U;
  double min_dist_sq = std::numeric_limits<double>::max();
  for (size_t i = 0; i < guide_path.size(); ++i) {
    const double dist_sq = (current_position - guide_path[i]).head<2>().squaredNorm();
    if (dist_sq < min_dist_sq) {
      min_dist_sq = dist_sq;
      start_idx = i;
    }
  }
  Eigen::Vector3d start = current_position;
  start.z() = 0.0;
  local_segment.push_back(start);
  double accum_dist = 0.0;
  Eigen::Vector3d prev = start;
  for (size_t i = start_idx; i < guide_path.size(); ++i) {
    Eigen::Vector3d p = guide_path[i];
    p.z() = 0.0;
    const double dist = (p - prev).head<2>().norm();
    if (dist > 1e-6) {
      accum_dist += dist;
      local_segment.push_back(p);
      prev = p;
    }
    if (accum_dist >= config_.lookahead_dist) {
      break;
    }
  }
  return local_segment;
}

std::vector<Eigen::Vector3d> MincoPipeline::sparsifyPath(
  const std::vector<Eigen::Vector3d> & dense_path, bool local_end_is_goal, int * mandatory_corner_count) const
{
  const auto build = getSparseWaypoints(dense_path,
    config_.optimizer.max_vel,
    config_.optimizer.max_acc,
    local_end_is_goal,
    [this, &dense_path](const Eigen::Vector3d & a, const Eigen::Vector3d & b) {
      return isLineFree(a, b, dense_path.front());
    });
  if (mandatory_corner_count) {
    *mandatory_corner_count = build.mandatory_corner_count;
  }
  return build.waypoints;
}

bool MincoPipeline::isLineFree(
  const Eigen::Vector3d & p1, const Eigen::Vector3d & p2,
  const Eigen::Vector3d & validation_start) const
{
  if (!map_) {
    return true;
  }

  const double resolution = std::max(1e-2, map_->resolution());
  const double step = std::max(0.5 * resolution, 0.02);
  const int steps = std::max(
    1,
    static_cast<int>(std::ceil((p2 - p1).head<2>().norm() / step)));

  for (int i = 0; i <= steps; ++i) {
    const double ratio = static_cast<double>(i) / static_cast<double>(steps);
    Eigen::Vector3d p = p1 + ratio * (p2 - p1);
    p.z() = 0.0;

    const auto query = map_->query(p);
    if (!query.ok || !std::isfinite(query.distance)) {
      return false;
    }
    const bool start_exempt =
      (p - validation_start).head<2>().norm() <=
      config_.start_validation_exemption_radius + 1e-12;
    if (start_exempt) {
      if (query.distance < 0.0) {
        return false;
      }
    } else if (query.distance <= config_.validation_safe_dist) {
      return false;
    }
  }
  return true;
}

MincoPipeline::PlanningState MincoPipeline::determinePlanningState(
  const State & current, const std::vector<Eigen::Vector3d> & sparse_path, double now,
  Result * diagnostics) const
{
  auto reject = [diagnostics](const std::string & reason) {
    if (diagnostics) diagnostics->hot_reject_reason = reason;
    return PlanningState::kColdStart;
  };
  if (!has_last_traj_) {
    return reject("NO_HISTORY");
  }
  const double t_dur = now - last_traj_.start_WT;
  const double total_duration = last_traj_.getTotalDuration();
  if (diagnostics) {
    diagnostics->history_age_s = t_dur;
    diagnostics->remaining_duration = total_duration - t_dur;
  }
  if (t_dur <= 0.0) {
    return reject("HOT_REJECT_TIME_ORDER");
  }
  if (t_dur >= total_duration) {
    return reject("HOT_REJECT_EXPIRED");
  }
  const Eigen::Vector3d pred_pos = last_traj_.getPos(t_dur);
  const Eigen::Vector3d pred_vel = last_traj_.getVel(t_dur);
  const double tracking_error = (current.position - pred_pos).norm();
  const double velocity_error = (current.velocity - pred_vel).norm();
  if (diagnostics) {
    diagnostics->position_error = tracking_error;
    diagnostics->velocity_error = velocity_error;
  }
  const double dynamic_error_threshold = 1.0 + 0.5 * current.velocity.head<2>().norm();
  if (tracking_error > dynamic_error_threshold) {
    return reject("HOT_REJECT_POSITION");
  }
  if (velocity_error > 1.0) {
    return reject("HOT_REJECT_VELOCITY");
  }
  if (map_) {
    const double remaining = total_duration - t_dur;
    const double step = std::max(0.02, config_.validation_sample_dt);
    const int sample_count = std::max(1, static_cast<int>(std::ceil(remaining / step)));
    double min_clearance = std::numeric_limits<double>::infinity();
    Eigen::Vector3d history_start = last_traj_.getPos(t_dur);
    history_start.z() = 0.0;
    for (int i = 0; i <= sample_count; ++i) {
      const double sample_t = std::min(
        total_duration, t_dur + remaining * static_cast<double>(i) / sample_count);
      Eigen::Vector3d sample_pos = last_traj_.getPos(sample_t);
      sample_pos.z() = 0.0;
      const auto query = map_->query(sample_pos);
      if (query.ok) min_clearance = std::min(min_clearance, query.distance);
      const bool start_exempt =
        (sample_pos - history_start).head<2>().norm() <=
        config_.start_validation_exemption_radius + 1e-12;
      const bool unsafe =
        !query.ok || !std::isfinite(query.distance) ||
        (start_exempt ? query.distance < 0.0 : query.distance <= config_.validation_safe_dist);
      if (unsafe) {
        if (diagnostics && std::isfinite(min_clearance)) diagnostics->history_min_clearance = min_clearance;
        return reject(query.ok ? "HOT_REJECT_HISTORY_UNSAFE" : "HOT_REJECT_HISTORY_OOB");
      }
    }
    if (diagnostics && std::isfinite(min_clearance)) diagnostics->history_min_clearance = min_clearance;
  }
  if (sparse_path.size() >= 2U && pred_vel.norm() > 0.1) {
    const Eigen::Vector3d delta = sparse_path[1] - sparse_path[0];
    if (delta.norm() <= 1e-9) return reject("HOT_REJECT_PATH_DIRECTION_INVALID");
    const double direction_dot = pred_vel.normalized().dot(delta.normalized());
    if (diagnostics) diagnostics->direction_dot = direction_dot;
    if (direction_dot < 0.5) {
      return reject("HOT_REJECT_DIRECTION");
    }
  }
  if (diagnostics) diagnostics->hot_reject_reason = "HOT_ACCEPTED";
  return PlanningState::kHotStart;
}

void MincoPipeline::prepareColdStart(const State & current, Eigen::Matrix3d & start_state) const
{
  start_state.setZero();
  start_state.col(0) = current.position;
  start_state.col(1) = current.velocity;
  start_state.col(2) = current.acceleration;
}

void MincoPipeline::prepareHotStart(const State & current, double sample_t, Eigen::Matrix3d & start_state) const
{
  start_state.setZero();
  start_state.col(0) = current.position;
  start_state.col(1) = last_traj_.getVel(sample_t);
  start_state.col(2) = last_traj_.getAcc(sample_t);
}

void MincoPipeline::allocatePathTime(const std::vector<Eigen::Vector3d> & sparse_path,
  const Eigen::Matrix3d & start_state,
  bool goal_reached,
  PlanningState state,
  bool has_shifted_seed,
  const super_utils::vec_Vec3f & shifted_waypoints,
  const super_utils::VecDf & shifted_durations,
  super_utils::vec_Vec3f & init_ps,
  super_utils::VecDf & init_ts,
  super_utils::VecDf & local_vmaxs) const
{
  const int N = static_cast<int>(sparse_path.size()) - 1;
  if (N <= 0) {
    init_ps.clear();
    init_ts.resize(0);
    local_vmaxs.resize(0);
    return;
  }

  const double global_vmax = std::max(0.0, config_.optimizer.max_vel);
  const double amax = std::max(1e-3, config_.optimizer.max_acc);
  const double kMinSegTime = 0.1;
  const double kBrakeSafety = 1.2;
  const double max_brake_dist = (global_vmax * global_vmax) / (2.0 * amax);

  local_vmaxs.resize(N);
  local_vmaxs.setConstant(global_vmax);
  init_ts.resize(N);
  init_ps.clear();
  init_ps.reserve(static_cast<size_t>(std::max(0, N - 1)));

  int copyPs = 0;
  if (state == PlanningState::kHotStart && has_shifted_seed) {
    copyPs = std::min(std::max(0, N - 1), std::max(0, static_cast<int>(shifted_waypoints.size()) - 2));
  }
  for (int j = 0; j < copyPs; ++j) {
    init_ps.emplace_back(shifted_waypoints[static_cast<size_t>(j + 1)]);
  }
  for (int j = copyPs; j < (N - 1); ++j) {
    init_ps.emplace_back(sparse_path[static_cast<size_t>(j + 1)]);
  }

  std::vector<double> seg_len(static_cast<size_t>(N), 0.0);
  for (int i = 0; i < N; ++i) {
    const double dis =
      (sparse_path[static_cast<size_t>(i + 1)] - sparse_path[static_cast<size_t>(i)]).head<2>().norm();
    seg_len[static_cast<size_t>(i)] = (std::isfinite(dis) && dis > 0.0) ? dis : 0.0;
  }
  std::vector<double> remain_after(static_cast<size_t>(N), 0.0);
  for (int i = N - 2; i >= 0; --i) {
    remain_after[static_cast<size_t>(i)] =
      remain_after[static_cast<size_t>(i + 1)] + seg_len[static_cast<size_t>(i + 1)];
  }

  std::vector<double> local_vmax_vec(static_cast<size_t>(N), global_vmax);
  for (int i = 0; i < N - 1; ++i) {
    local_vmax_vec[static_cast<size_t>(i)] = limitLocalVel(sparse_path,
      i,
      global_vmax,
      config_.optimizer.turn_angle_deadzone,
      config_.optimizer.turn_angle_saturation,
      config_.optimizer.min_turn_vel,
      config_.optimizer.decay_power);
  }
  propagateVelocityLimits(seg_len, amax, local_vmax_vec);
  for (int i = 0; i < N; ++i) {
    local_vmaxs(i) = local_vmax_vec[static_cast<size_t>(i)];
  }

  double v_curr = start_state.col(1).head<2>().norm();
  if (!std::isfinite(v_curr) || v_curr < 0.0) {
    v_curr = 0.0;
  }
  const double local_goal_remain = goal_reached ? 0.0 : max_brake_dist;
  for (int i = 0; i < N; ++i) {
    const bool is_last = (i == N - 1);
    const double L = seg_len[static_cast<size_t>(i)];
    const double remain = remain_after[static_cast<size_t>(i)] + local_goal_remain;
    if (L <= 1e-6) {
      init_ts(i) = kMinSegTime;
      continue;
    }
    if (is_last && goal_reached) {
      init_ts(i) = std::max({kMinSegTime, L / std::max(v_curr, 0.1), kBrakeSafety * v_curr / amax});
      v_curr = 0.0;
      continue;
    }
    const double local_vmax = local_vmax_vec[static_cast<size_t>(i)];
    const double v_next = computeNextSpeed(v_curr, L, remain, amax, local_vmax);
    init_ts(i) = computeSegmentTime(L, v_curr, v_next, local_vmax, amax, kMinSegTime);
    v_curr = v_next;
  }

  if (state == PlanningState::kHotStart && has_shifted_seed) {
    const int oldN = std::min(N, static_cast<int>(shifted_durations.size()));
    for (int i = 0; i < oldN; ++i) {
      const double t_seed = shifted_durations(i);
      if (std::isfinite(t_seed) && t_seed > 0.02) {
        init_ts(i) = std::max(init_ts(i), t_seed);
      }
    }
  }
}

bool MincoPipeline::validateTrajectory(
  const geometry_utils::Trajectory & trajectory, const Eigen::Vector3d & expected_end_pos,
  Result * diagnostics) const
{
  const bool use_strict_validation =
      config_.enable_strict_validation ||
      config_.constraint_profile == "safe_corridor_v1" ||
      config_.constraint_profile == "superplanner_sfc_v1";
  if (use_strict_validation) {
    auto reject = [diagnostics](
      const std::string & reason, int index, double t, const Eigen::Vector3d & position,
      double measured, double limit)
    {
      if (diagnostics) {
        diagnostics->validation_failure_reason = reason;
        diagnostics->validation_offending_sample_index = index;
        diagnostics->validation_offending_time_s = t;
        diagnostics->validation_offending_position = position;
        diagnostics->validation_measured_value = measured;
        diagnostics->validation_limit_value = limit;
      }
      return false;
    };
    const double duration = trajectory.getTotalDuration();
    if (!(std::isfinite(duration) && duration > 1e-6)) {
      return reject(
        "VALIDATION_INVALID_DURATION", 0, 0.0, Eigen::Vector3d::Zero(), duration, 0.0);
    }
    if (config_.constraint_profile == "safe_corridor_v1" &&
        (!map_ || corridor_.segments().empty())) {
      return reject(
        "VALIDATION_CORRIDOR", 0, 0.0, trajectory.getPos(0.0), -1.0, 0.0);
    }
    if (config_.constraint_profile == "superplanner_sfc_v1" &&
        (!map_ || sfc_.cells().empty())) {
      return reject(
        "VALIDATION_SFC", 0, 0.0, trajectory.getPos(0.0), -1.0, 0.0);
    }
    Eigen::VectorXd sfc_piece_durations;
    if (config_.constraint_profile == "superplanner_sfc_v1") {
      sfc_piece_durations = trajectory.getDurations();
      const bool bindings_valid = diagnostics &&
        diagnostics->sfc_piece_bindings.size() ==
        static_cast<size_t>(sfc_piece_durations.size()) &&
        diagnostics->sfc_piece_bindings.size() == sfc_.cells().size();
      if (!bindings_valid) {
        return reject(
          "VALIDATION_SFC_BINDING", 0, 0.0, trajectory.getPos(0.0),
          static_cast<double>(sfc_piece_durations.size()),
          static_cast<double>(sfc_.cells().size()));
      }
    }

    struct Node {
      double t;
      Eigen::Vector3d pos;
      Eigen::Vector3d vel;
      Eigen::Vector3d acc;
      Eigen::Vector3d jerk;
      double clearance;
      bool query_ok;
    };
    auto sample = [&](double t) {
      Node node;
      node.t = t;
      node.pos = trajectory.getPos(t);
      node.vel = trajectory.getVel(t);
      node.acc = trajectory.getAcc(t);
      node.jerk = trajectory.getJer(t);
      const auto query = map_->query(node.pos);
      node.query_ok = query.ok && std::isfinite(query.distance);
      node.clearance = node.query_ok ?
        query.distance : -std::numeric_limits<double>::infinity();
      return node;
    };

    std::vector<Node> ordered;
    ordered.reserve(static_cast<size_t>(std::max(2, config_.adaptive_sample_budget)));
    int subdivisions = 0;
    bool budget_exhausted = false;
    bool depth_exhausted = false;
    double depth_exhausted_displacement = std::numeric_limits<double>::quiet_NaN();
    double depth_exhausted_time = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d depth_exhausted_position =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    std::function<void(const Node &, const Node &, int)> refine;
    refine = [&](const Node & left, const Node & right, int depth) {
      if (budget_exhausted || depth_exhausted) {
        return;
      }
      const double dt = right.t - left.t;
      const double displacement = (right.pos - left.pos).head<2>().norm();
      const double derivative_change =
        (right.vel - left.vel).head<2>().norm() * std::max(0.0, dt);
      const double near_limit =
        config_.validation_safe_dist + std::max(0.0, config_.adaptive_near_clearance);
      const bool near_clearance =
        !left.query_ok || !right.query_ok ||
        left.clearance <= near_limit || right.clearance <= near_limit;
      const bool needs_refine =
        dt > 1e-6 &&
        (displacement > config_.adaptive_max_spatial_step ||
        derivative_change > config_.adaptive_max_spatial_step ||
        near_clearance);
      if (needs_refine && depth >= config_.adaptive_max_depth) {
        depth_exhausted = true;
        depth_exhausted_displacement = displacement;
        depth_exhausted_time = 0.5 * (left.t + right.t);
        depth_exhausted_position = 0.5 * (left.pos + right.pos);
        return;
      }
      if (needs_refine) {
        if (static_cast<int>(ordered.size()) + subdivisions + 3 >
          config_.adaptive_sample_budget)
        {
          budget_exhausted = true;
          return;
        }
        const Node middle = sample(0.5 * (left.t + right.t));
        ++subdivisions;
        refine(left, middle, depth + 1);
        refine(middle, right, depth + 1);
      } else {
        ordered.push_back(left);
      }
    };
    const Node start = sample(0.0);
    const Node end = sample(duration);
    refine(start, end, 0);
    ordered.push_back(end);
    if (diagnostics) {
      diagnostics->adaptive_validation_sample_count = static_cast<int>(ordered.size());
      diagnostics->adaptive_validation_subdivision_count = subdivisions;
    }
    if (budget_exhausted) {
      return reject(
        "VALIDATION_BUDGET_EXHAUSTED", static_cast<int>(ordered.size()), duration,
        end.pos, static_cast<double>(ordered.size()), config_.adaptive_sample_budget);
    }
    if (depth_exhausted) {
      return reject(
        "VALIDATION_DEPTH_EXHAUSTED", static_cast<int>(ordered.size()),
        depth_exhausted_time, depth_exhausted_position,
        depth_exhausted_displacement, config_.adaptive_max_spatial_step);
    }

    double min_clearance = std::numeric_limits<double>::infinity();
    int corridor_segment = 0;
    double prior_clearance = start.clearance;
    for (size_t index = 0; index < ordered.size(); ++index) {
      const auto & node = ordered[index];
      const int sample_index = static_cast<int>(index);
      if (!(node.pos.allFinite() && node.vel.allFinite() &&
        node.acc.allFinite() && node.jerk.allFinite()))
      {
        return reject(
          "VALIDATION_NONFINITE_DYNAMICS", sample_index, node.t, node.pos,
          std::numeric_limits<double>::quiet_NaN(), 0.0);
      }
      if (!node.query_ok) {
        if (diagnostics) {
          ++diagnostics->validation_oob_count;
        }
        return reject("VALIDATION_ESDF_OOB", sample_index, node.t, node.pos, -1.0, 0.0);
      }
      min_clearance = std::min(min_clearance, node.clearance);
      if (node.clearance < 0.0) {
        if (diagnostics) {
          ++diagnostics->validation_negative_esdf_count;
        }
        return reject(
          "VALIDATION_NEGATIVE_ESDF", sample_index, node.t, node.pos,
          node.clearance, 0.0);
      }
      const double start_distance = (node.pos - start.pos).head<2>().norm();
      const bool in_start_exemption =
        start_distance <= config_.start_validation_exemption_radius &&
        node.clearance > 0.0 &&
        node.clearance <= config_.validation_safe_dist &&
        node.clearance + 1e-9 >= prior_clearance;
      if (node.clearance <= config_.validation_safe_dist && !in_start_exemption) {
        return reject(
          "VALIDATION_CLEARANCE", sample_index, node.t, node.pos,
          node.clearance, config_.validation_safe_dist);
      }
      if (in_start_exemption && diagnostics) {
        ++diagnostics->validation_start_exempt_count;
      }
      prior_clearance = node.clearance;
      if (config_.constraint_profile == "safe_corridor_v1") {
        int matched_segment = corridor_segment;
        double corridor_margin = 0.0;
        if (!corridor_.contains(
            node.pos, corridor_segment, &matched_segment, &corridor_margin))
        {
          return reject(
            "VALIDATION_CORRIDOR", sample_index, node.t, node.pos, corridor_margin, 0.0);
        }
        corridor_segment = matched_segment;
      }
      if (config_.constraint_profile == "superplanner_sfc_v1") {
        int piece_index = 0;
        double piece_end_time = sfc_piece_durations.size() > 0 ?
          sfc_piece_durations(0) : 0.0;
        while (piece_index + 1 < sfc_piece_durations.size() &&
          node.t > piece_end_time + 1e-9)
        {
          ++piece_index;
          piece_end_time += sfc_piece_durations(piece_index);
        }
        const auto & binding =
          diagnostics->sfc_piece_bindings[static_cast<size_t>(piece_index)];
        double sfc_margin = -std::numeric_limits<double>::infinity();
        if (binding.piece_index != piece_index ||
          !sfc_.containsInCell(node.pos, binding.cell_index, &sfc_margin))
        {
          return reject(
            "VALIDATION_SFC", sample_index, node.t, node.pos, sfc_margin, 0.0);
        }
        if (diagnostics) {
          diagnostics->sfc_min_margin = std::isfinite(diagnostics->sfc_min_margin) ?
            std::min(diagnostics->sfc_min_margin, sfc_margin) : sfc_margin;
        }
      }
      const double speed = node.vel.head<2>().norm();
      const double acceleration = node.acc.head<2>().norm();
      const double jerk = node.jerk.head<2>().norm();
      const double speed_tolerance =
        std::max(1e-6, 1e-3 * config_.optimizer.max_vel);
      const double acceleration_tolerance =
        std::max(1e-6, 1e-3 * config_.optimizer.max_acc);
      const double jerk_tolerance = std::max(1e-6, 1e-3 * config_.max_jerk);
      if (speed > config_.optimizer.max_vel + speed_tolerance) {
        return reject(
          "VALIDATION_VELOCITY", sample_index, node.t, node.pos,
          speed, config_.optimizer.max_vel);
      }
      if (acceleration > config_.optimizer.max_acc + acceleration_tolerance) {
        return reject(
          "VALIDATION_ACCELERATION", sample_index, node.t, node.pos,
          acceleration, config_.optimizer.max_acc);
      }
      if (jerk > config_.max_jerk + jerk_tolerance) {
        return reject(
          "VALIDATION_JERK", sample_index, node.t, node.pos, jerk, config_.max_jerk);
      }
    }
    const Eigen::Vector3d end_pos = trajectory.getPos(duration);
    if (!end_pos.allFinite() ||
      (end_pos - expected_end_pos).norm() > config_.traj_goal_tolerance)
    {
      return reject(
        end_pos.allFinite() ? "VALIDATION_GOAL_TOLERANCE" : "VALIDATION_NONFINITE_END",
        static_cast<int>(ordered.size()) - 1, duration, end_pos,
        (end_pos - expected_end_pos).norm(), config_.traj_goal_tolerance);
    }
    if (diagnostics) {
      diagnostics->validation_min_clearance = min_clearance;
      diagnostics->validation_failure_reason = "NONE";
    }
    return true;
  }
  auto reject = [diagnostics](const std::string & reason) {
    if (diagnostics) {
      diagnostics->validation_failure_reason = reason;
    }
    return false;
  };
  const double dur = trajectory.getTotalDuration();
  if (!(std::isfinite(dur) && dur > 1e-6)) {
    return reject("VALIDATION_INVALID_DURATION");
  }
  const double vmax = config_.optimizer.max_vel;
  const double amax = config_.optimizer.max_acc;
  const double vmax_severe = config_.validation_dynamic_scale * vmax;
  const double amax_severe = config_.validation_dynamic_scale * amax;
  const double dt = std::max(1e-3, config_.validation_sample_dt);
  for (double t = 0.0; t <= dur; t += dt) {
    const Eigen::Vector3d v = trajectory.getVel(t);
    const Eigen::Vector3d a = trajectory.getAcc(t);
    if (!(v.allFinite() && a.allFinite())) {
      return reject("VALIDATION_NONFINITE_DYNAMICS");
    }
    if (v.norm() > vmax_severe) {
      return reject("VALIDATION_VELOCITY_LIMIT");
    }
    if (a.norm() > amax_severe) {
      return reject("VALIDATION_ACCELERATION_LIMIT");
    }
  }
  const Eigen::Vector3d end_pos = trajectory.getPos(dur);
  if (!end_pos.allFinite() || (end_pos - expected_end_pos).norm() > config_.traj_goal_tolerance) {
    return reject(end_pos.allFinite() ? "VALIDATION_GOAL_TOLERANCE" : "VALIDATION_NONFINITE_END");
  }
  if (safety_checker_) {
    const auto report = safety_checker_->inspectTrajectory(trajectory);
    if (diagnostics) {
      diagnostics->validation_min_clearance = report.min_clearance;
      diagnostics->validation_oob_count = report.out_of_bounds_count;
      diagnostics->validation_start_exempt_count = report.start_exempt_sample_count;
      diagnostics->validation_negative_esdf_count = report.negative_esdf_count;
    }
    if (!report.safe) {
      return reject(report.reason);
    }
  }
  if (diagnostics) {
    diagnostics->validation_failure_reason = "NONE";
  }
  return true;
}

bool MincoPipeline::validateYawAndWheels(
  const geometry_utils::Trajectory & trajectory,
  const geometry_utils::Trajectory & yaw_trajectory,
  Result * diagnostics) const
{
  const double duration = trajectory.getTotalDuration();
  const double yaw_duration = yaw_trajectory.getTotalDuration();
  if (!(std::isfinite(duration) && duration > 0.0 &&
    std::isfinite(yaw_duration) && yaw_duration > 0.0))
  {
    if (diagnostics) {
      diagnostics->validation_failure_reason = "VALIDATION_YAW_RATE";
    }
    return false;
  }
  const double dt = std::max(
    1e-3, std::min(config_.validation_sample_dt,
    config_.adaptive_max_spatial_step / std::max(1e-3, config_.optimizer.max_vel)));
  int index = 0;
  for (double t = 0.0; t <= duration + 1e-9; t += dt, ++index) {
    const double sample_t = std::min(t, duration);
    const Eigen::Vector3d position = trajectory.getPos(sample_t);
    const double speed = trajectory.getVel(sample_t).head<2>().norm();
    const double yaw_rate = yaw_trajectory.getVel(std::min(sample_t, yaw_duration))(0);
    auto reject = [&](const std::string & reason, double measured, double limit) {
      if (diagnostics) {
        diagnostics->validation_failure_reason = reason;
        diagnostics->validation_offending_sample_index = index;
        diagnostics->validation_offending_time_s = sample_t;
        diagnostics->validation_offending_position = position;
        diagnostics->validation_measured_value = measured;
        diagnostics->validation_limit_value = limit;
      }
      return false;
    };
    const double yaw_tolerance =
      std::max(1e-6, 1e-3 * config_.max_yaw_rate);
    if (!std::isfinite(yaw_rate) ||
      std::abs(yaw_rate) > config_.max_yaw_rate + yaw_tolerance)
    {
      return reject(
        "VALIDATION_YAW_RATE", std::abs(yaw_rate), config_.max_yaw_rate);
    }
    const double left =
      (2.0 * speed - config_.wheel_base * yaw_rate) /
      (2.0 * config_.wheel_radius);
    const double right =
      (2.0 * speed + config_.wheel_base * yaw_rate) /
      (2.0 * config_.wheel_radius);
    const double measured = std::max(std::abs(left), std::abs(right));
    const double wheel_tolerance =
      std::max(1e-6, 1e-3 * config_.max_wheel_speed);
    if (!std::isfinite(measured) ||
      measured > config_.max_wheel_speed + wheel_tolerance)
    {
      return reject(
        "VALIDATION_WHEEL_SPEED", measured, config_.max_wheel_speed);
    }
  }
  return true;
}

bool MincoPipeline::optimizeYaw(const Eigen::Matrix3d & start_state,
  const geometry_utils::Trajectory & pos_traj,
  geometry_utils::Trajectory & out_yaw_traj,
  PlanningState state,
  double current_yaw,
  double current_yaw_rate,
  double goal_yaw,
  double now) const
{
  (void)state;
  (void)now;
  if (!config_.enable_yaw_opt || !yaw_optimizer_) {
    return false;
  }
  Eigen::Vector4d init_yaw_state = Eigen::Vector4d::Zero();
  Eigen::Vector4d goal_yaw_state = Eigen::Vector4d::Zero();
  double init_yaw = current_yaw;
  if (!std::isfinite(init_yaw)) {
    const Eigen::Vector2d vel_xy = start_state.col(1).head<2>();
    if (vel_xy.allFinite() && vel_xy.norm() > 0.1) {
      init_yaw = std::atan2(vel_xy.y(), vel_xy.x());
    } else {
      init_yaw = 0.0;
    }
  }
  init_yaw_state(0) = init_yaw;
  init_yaw_state(1) = std::clamp(
    std::isfinite(current_yaw_rate) ? current_yaw_rate : 0.0,
    -config_.max_yaw_rate,
    config_.max_yaw_rate);
  if (!std::isfinite(goal_yaw)) {
    goal_yaw = init_yaw_state(0);
  }
  const double yaw_err =
    std::atan2(std::sin(goal_yaw - init_yaw_state(0)), std::cos(goal_yaw - init_yaw_state(0)));
  goal_yaw_state(0) = init_yaw_state(0) + yaw_err;
  return yaw_optimizer_->optimize(init_yaw_state, goal_yaw_state, pos_traj, out_yaw_traj, 5, false, true);
}

std::vector<MincoPipeline::TrajectorySample> MincoPipeline::sampleTrajectory(
  const geometry_utils::Trajectory & pos_traj,
  const geometry_utils::Trajectory & yaw_traj,
  double dt,
  double fallback_yaw) const
{
  std::vector<TrajectorySample> samples;
  const double duration = pos_traj.getTotalDuration();
  if (!(std::isfinite(duration) && duration >= 0.0)) {
    return samples;
  }

  const double step = (std::isfinite(dt) && dt > 1e-6) ? dt : 0.05;
  const double yaw_duration = yaw_traj.getTotalDuration();
  const bool has_yaw = !yaw_traj.empty() && std::isfinite(yaw_duration) && yaw_duration > 1e-6;
  const int sample_count = std::max(1, static_cast<int>(std::ceil(duration / step)) + 1);

  samples.reserve(static_cast<size_t>(sample_count));
  for (int i = 0; i < sample_count; ++i) {
    const double t = std::min(duration, static_cast<double>(i) * step);
    TrajectorySample sample;
    sample.t = t;
    sample.pos = pos_traj.getPos(t);
    sample.vel = pos_traj.getVel(t);
    sample.acc = pos_traj.getAcc(t);
    sample.jerk = pos_traj.getJer(t);

    sample.pos.z() = 0.0;
    sample.vel.z() = 0.0;
    sample.acc.z() = 0.0;
    sample.jerk.z() = 0.0;

    if (has_yaw) {
      const double yaw_t = std::min(t, yaw_duration);
      sample.yaw = yaw_traj.getPos(yaw_t)(0);
      sample.yaw_dot = yaw_traj.getVel(yaw_t)(0);
    } else {
      const double planar_speed = sample.vel.head<2>().norm();
      sample.yaw = planar_speed > 1e-6 ? std::atan2(sample.vel.y(), sample.vel.x()) : fallback_yaw;
      sample.yaw_dot = 0.0;
    }
    // The sampled command/state is also the state handed to the following
    // rolling cycle.  Keep it inside the very same yaw-rate limit used by the
    // optimizer and validator; otherwise a slightly overshooting sampled yaw
    // rate is clipped only on the *next* plan and breaks state continuity.
    sample.yaw_dot = std::clamp(
      sample.yaw_dot, -config_.max_yaw_rate, config_.max_yaw_rate);

    samples.push_back(sample);
  }

  return samples;
}

}  // namespace minco_processor
