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

SparseWaypointBuild limitSparseWaypoints(const SparseWaypointBuild & in,
  const std::vector<Eigen::Vector3d> & dense_path,
  int max_count,
  const std::function<bool(const Eigen::Vector3d &, const Eigen::Vector3d &)> & is_line_free)
{
  const int n = static_cast<int>(in.waypoints.size());
  const int limit = std::max(2, max_count);
  if (n <= limit) {
    return in;
  }

  std::vector<size_t> keep;
  keep.reserve(static_cast<size_t>(limit));
  keep.push_back(0U);
  for (size_t i = 1U; i + 1U < in.waypoints.size(); ++i) {
    if (in.mandatory[i]) {
      keep.push_back(i);
    }
  }
  keep.push_back(in.waypoints.size() - 1U);
  std::sort(keep.begin(), keep.end());
  keep.erase(std::unique(keep.begin(), keep.end()), keep.end());

  if (static_cast<int>(keep.size()) < limit) {
    for (int slot = 1; slot < limit - 1; ++slot) {
      const size_t idx = static_cast<size_t>(std::lround(
        static_cast<double>(slot) * static_cast<double>(n - 1) / static_cast<double>(limit - 1)));
      if (std::find(keep.begin(), keep.end(), idx) == keep.end()) {
        keep.push_back(idx);
      }
    }
    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    while (static_cast<int>(keep.size()) > limit) {
      auto removable = std::find_if(keep.rbegin(), keep.rend(), [&in](size_t idx) {
        return idx != 0U && idx + 1U != in.waypoints.size() && !in.mandatory[idx];
      });
      if (removable == keep.rend()) {
        break;
      }
      keep.erase(std::next(removable).base());
    }
  }

  SparseWaypointBuild out;
  for (size_t idx : keep) {
    appendSparsePoint(out, in.waypoints, idx, in.mandatory[idx]);
    out.source_indices.back() = in.source_indices[idx];
  }
  out.mandatory_corner_count = in.mandatory_corner_count;

  if (is_line_free && dense_path.size() >= 2U) {
    for (size_t i = 0U; i + 1U < out.waypoints.size();) {
      if (is_line_free(out.waypoints[i], out.waypoints[i + 1U])) {
        ++i;
        continue;
      }
      const size_t start_idx = std::min(out.source_indices[i], out.source_indices[i + 1U]);
      const size_t end_idx = std::max(out.source_indices[i], out.source_indices[i + 1U]);
      size_t corner_idx = findCornerIndex(dense_path, start_idx, end_idx);
      if (corner_idx <= start_idx || corner_idx >= end_idx) {
        corner_idx = std::min(start_idx + 1U, end_idx);
      }
      if (corner_idx <= start_idx || corner_idx >= end_idx) {
        ++i;
        continue;
      }
      Eigen::Vector3d corner = dense_path[corner_idx];
      corner.z() = 0.0;
      out.waypoints.insert(out.waypoints.begin() + static_cast<std::ptrdiff_t>(i + 1U), corner);
      out.source_indices.insert(out.source_indices.begin() + static_cast<std::ptrdiff_t>(i + 1U), corner_idx);
      out.mandatory.insert(out.mandatory.begin() + static_cast<std::ptrdiff_t>(i + 1U), true);
      ++out.mandatory_corner_count;
      ++i;
    }
  }
  return out;
}

SparseWaypointBuild getSparseWaypoints(const std::vector<Eigen::Vector3d> & path,
  double max_vel,
  double max_acc,
  bool goal_reached,
  int max_count,
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
  return limitSparseWaypoints(sparse, path, max_count, is_line_free);
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
  if (config_.optimizer.penaltyWeights.size() != 5) {
    config_.optimizer.penaltyWeights.resize(5);
    config_.optimizer.penaltyWeights << 1000.0, 1000.0, 10000.0, 1000.0, 100.0;
  }
  optimizer_ = std::make_unique<minco_planner::MincoOptimizer>(config_.optimizer);
  yaw_optimizer_ = std::make_unique<traj_opt::YawTrajOpt>(config_.max_yaw_rate);
  safety_checker_ = std::make_unique<minco_planner::TrajectorySafetyChecker>();
  safety_checker_->configure(config_.optimizer.safe_dist, config_.safety_sample_dt);
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

MincoPipeline::Result MincoPipeline::optimize(const Request & request)
{
  using Clock = std::chrono::steady_clock;

  Result result;
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
  stage_start = Clock::now();
  result.sparse_waypoints = sparsifyPath(result.dense_path, local_end_is_goal, &result.mandatory_corner_count);
  result.timing_ms["sparsify_path_ms"] = elapsedMs(stage_start);
  if (result.sparse_waypoints.size() < 2U) {
    result.failure_reason = "SPARSIFY_FAILED";
    return finish();
  }

  stage_start = Clock::now();
  result.planning_state = PlanningState::kColdStart;
  prepareColdStart(request.current, result.start_state);
  optimizer_->setInitPsAndTs(super_utils::vec_Vec3f{}, super_utils::VecDf{});

  if (safety_checker_) {
    Eigen::Vector3d start_pos = result.start_state.col(0);
    if (safety_checker_->projectOutOfObstacle(start_pos, config_.start_projection_margin)) {
      result.start_state.col(0) = start_pos;
    }
  }

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

  while (result.sparse_waypoints.size() > 2U &&
         (result.sparse_waypoints[1] - result.start_state.col(0)).norm() < 0.2) {
    result.sparse_waypoints.erase(result.sparse_waypoints.begin() + 1);
  }
  result.timing_ms["state_prepare_ms"] = elapsedMs(stage_start);

  const int N = static_cast<int>(result.sparse_waypoints.size()) - 1;
  result.local_vmaxs.resize(N);
  result.initial_times.resize(N);
  stage_start = Clock::now();
  allocatePathTime(result.sparse_waypoints,
    result.start_state,
    local_end_is_goal,
    PlanningState::kColdStart,
    false,
    super_utils::vec_Vec3f{},
    super_utils::VecDf{},
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
  if (!std::isfinite(result.objective)) {
    result.failure_reason = "OPTIMIZER_FAILED";
    return finish();
  }
  stage_start = Clock::now();
  if (!validateTrajectory(result.trajectory, result.end_state.col(0))) {
    result.timing_ms["validate_ms"] = elapsedMs(stage_start);
    result.failure_reason = "VALIDATION_FAILED";
    return finish();
  }
  result.timing_ms["validate_ms"] = elapsedMs(stage_start);

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
    config_.max_sparse_waypoints,
    [this](const Eigen::Vector3d & a, const Eigen::Vector3d & b) { return isLineFree(a, b); });
  if (mandatory_corner_count) {
    *mandatory_corner_count = build.mandatory_corner_count;
  }
  return build.waypoints;
}

bool MincoPipeline::isLineFree(const Eigen::Vector3d & p1, const Eigen::Vector3d & p2) const
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
    if (!query.ok || query.distance <= config_.optimizer.safe_dist) {
      return false;
    }
  }
  return true;
}

MincoPipeline::PlanningState MincoPipeline::determinePlanningState(
  const State & current, const std::vector<Eigen::Vector3d> & sparse_path, double now) const
{
  if (!has_last_traj_) {
    return PlanningState::kColdStart;
  }
  const double t_dur = now - last_traj_.start_WT;
  if (t_dur <= 0.0 || t_dur >= last_traj_.getTotalDuration()) {
    return PlanningState::kColdStart;
  }
  const Eigen::Vector3d pred_pos = last_traj_.getPos(t_dur);
  const Eigen::Vector3d pred_vel = last_traj_.getVel(t_dur);
  const double tracking_error = (current.position - pred_pos).norm();
  const double dynamic_error_threshold = 1.0 + 0.5 * current.velocity.head<2>().norm();
  if (tracking_error > dynamic_error_threshold) {
    return PlanningState::kColdStart;
  }
  if ((current.velocity - pred_vel).norm() > 1.0) {
    return PlanningState::kHotStart;
  }
  if (sparse_path.size() >= 2U && pred_vel.norm() > 0.1) {
    const Eigen::Vector3d path_dir = (sparse_path[1] - sparse_path[0]).normalized();
    if (pred_vel.normalized().dot(path_dir) < 0.5) {
      return PlanningState::kColdStart;
    }
  }
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
  const geometry_utils::Trajectory & trajectory, const Eigen::Vector3d & expected_end_pos) const
{
  const double dur = trajectory.getTotalDuration();
  if (!(std::isfinite(dur) && dur > 1e-6)) {
    return false;
  }
  const double vmax = config_.optimizer.max_vel;
  const double amax = config_.optimizer.max_acc;
  const double vmax_severe = config_.validation_dynamic_scale * vmax;
  const double amax_severe = config_.validation_dynamic_scale * amax;
  const double dt = std::max(1e-3, config_.validation_sample_dt);
  for (double t = 0.0; t <= dur; t += dt) {
    const Eigen::Vector3d v = trajectory.getVel(t);
    const Eigen::Vector3d a = trajectory.getAcc(t);
    if (!(v.allFinite() && a.allFinite()) || v.norm() > vmax_severe || a.norm() > amax_severe) {
      return false;
    }
  }
  const Eigen::Vector3d end_pos = trajectory.getPos(dur);
  if (!end_pos.allFinite() || (end_pos - expected_end_pos).norm() > config_.traj_goal_tolerance) {
    return false;
  }
  return safety_checker_ ? safety_checker_->checkTrajectory(trajectory) : true;
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

    samples.push_back(sample);
  }

  return samples;
}

}  // namespace minco_processor
