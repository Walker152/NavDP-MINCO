#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <map>
#include <vector>

#include "minco_processor/minco_pipeline.hpp"
#include "minco_processor/static_sim_esdf_map.hpp"

namespace py = pybind11;

namespace {

class MincoProcessorPy
{
public:
  MincoProcessorPy()
  {
    configure(2.0, 4.0, 0.3, 0.05, 64, 0.5, 1000.0, 1000.0, 10000.0, 20.0, 0.01, 100.0);
  }

  void configure(
    double max_vel, double max_acc, double safe_dist, double sample_dt, int max_iterations,
    double max_yaw_rate, double penalty_weight_pos, double penalty_weight_vel,
    double penalty_weight_acc, double penalty_weight_attractor, double time_weight,
    double time_barrier_weight)
  {
    minco_processor::MincoPipeline::Config config;
    config.sample_dt = sample_dt;
    config.validation_sample_dt = sample_dt;
    config.safety_sample_dt = sample_dt;
    config.optimizer.safe_dist = safe_dist;
    config.optimizer.max_vel = max_vel;
    config.optimizer.max_acc = max_acc;
    config.optimizer.max_iterations = max_iterations;
    config.optimizer.rho = time_weight;
    config.optimizer.penaltyWeights.resize(5);
    config.optimizer.penaltyWeights << penalty_weight_pos, penalty_weight_vel, penalty_weight_acc,
      penalty_weight_attractor, time_barrier_weight;
    config.optimizer.print_optimizer_log = false;
    config.max_yaw_rate = max_yaw_rate;
    pipeline_.setConfig(config);
  }

  void set_static_esdf_2d(
    const Eigen::MatrixXd & distance,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> & free,
    const Eigen::Vector2d & origin,
    double resolution)
  {
    map_ = std::make_shared<minco_processor::StaticSimEsdfMap2D>();
    map_->setMap(distance, free, origin.x(), origin.y(), resolution);
    pipeline_.setMap(map_);
  }

  py::dict optimize(
    const Eigen::MatrixXd & guide_path,
    const Eigen::Vector3d & position,
    const Eigen::Vector3d & velocity,
    const Eigen::Vector3d & acceleration,
    double yaw,
    double yaw_rate,
    py::object terminal_goal)
  {
    const auto t0 = std::chrono::steady_clock::now();
    minco_processor::MincoPipeline::Request request;
    request.guide_path.reserve(static_cast<size_t>(guide_path.rows()));
    for (int i = 0; i < guide_path.rows(); ++i) {
      Eigen::Vector3d p = Eigen::Vector3d::Zero();
      p.x() = guide_path(i, 0);
      p.y() = guide_path(i, 1);
      if (guide_path.cols() > 2) {
        p.z() = guide_path(i, 2);
      }
      request.guide_path.push_back(p);
    }
    request.current.position = position;
    request.current.velocity = velocity;
    request.current.acceleration = acceleration;
    request.current.yaw = std::isfinite(yaw) ? yaw : 0.0;
    request.current.yaw_rate = std::isfinite(yaw_rate) ? yaw_rate : 0.0;
    if (!terminal_goal.is_none()) {
      Eigen::VectorXd goal_vec = terminal_goal.cast<Eigen::VectorXd>();
      if (goal_vec.size() >= 3 && goal_vec.head<3>().allFinite()) {
        request.has_terminal_goal = true;
        request.terminal_goal = goal_vec.head<3>();
        request.terminal_goal.z() = 0.0;
      }
    }
    request.now = std::chrono::duration<double>(t0.time_since_epoch()).count();

    minco_processor::MincoPipeline::Result result;
    {
      py::gil_scoped_release release;
      result = pipeline_.optimize(request);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double duration = std::chrono::duration<double>(t1 - t0).count();
    last_result_ = result;
    return toDict(result, duration);
  }

  py::dict optimizePreview(
    const Eigen::MatrixXd & guide_path, const Eigen::Vector3d & position,
    const Eigen::Vector3d & velocity, const Eigen::Vector3d & acceleration,
    double yaw, double yaw_rate, py::object terminal_goal)
  {
    py::dict out = optimize(guide_path, position, velocity, acceleration, yaw, yaw_rate, terminal_goal);
    const int proposal_id = next_proposal_id_++;
    proposals_[proposal_id] = last_result_;
    out["proposal_id"] = proposal_id;
    return out;
  }

  bool commitHistory(int proposal_id, double applied_time)
  {
    const auto it = proposals_.find(proposal_id);
    if (it == proposals_.end()) {
      return false;
    }
    const bool committed = pipeline_.commitHistory(it->second, applied_time);
    if (committed) {
      committed_history_uid_ = "proposal_" + std::to_string(proposal_id);
      proposals_.clear();
    }
    return committed;
  }

  void discardProposal(int proposal_id) { proposals_.erase(proposal_id); }
  void resetHistory() { proposals_.clear(); committed_history_uid_.clear(); pipeline_.resetHistory(); }

private:
  py::dict toDict(const minco_processor::MincoPipeline::Result & result, double duration) const
  {
    double min_esdf = std::numeric_limits<double>::infinity();
    auto accumulateMinEsdf = [this, &min_esdf](const Eigen::Vector3d & point) {
      if (map_) {
        const auto q = map_->query(point);
        if (q.ok) {
          min_esdf = std::min(min_esdf, q.distance);
        }
      }
    };

    const int sample_count = static_cast<int>(result.samples.size());
    Eigen::MatrixXd samples(sample_count, 15);
    Eigen::MatrixXd waypoints(
      sample_count > 0 ? sample_count : static_cast<int>(result.sparse_waypoints.size()), 3);
    if (sample_count > 0) {
      for (int i = 0; i < sample_count; ++i) {
        const auto & sample = result.samples[static_cast<size_t>(i)];
        samples(i, 0) = sample.t;
        samples.block<1, 3>(i, 1) = sample.pos.transpose();
        samples.block<1, 3>(i, 4) = sample.vel.transpose();
        samples.block<1, 3>(i, 7) = sample.acc.transpose();
        samples.block<1, 3>(i, 10) = sample.jerk.transpose();
        samples(i, 13) = sample.yaw;
        samples(i, 14) = sample.yaw_dot;
        waypoints.row(i) = sample.pos.transpose();
        accumulateMinEsdf(sample.pos);
      }
    } else {
      for (int i = 0; i < waypoints.rows(); ++i) {
        const auto & waypoint = result.sparse_waypoints[static_cast<size_t>(i)];
        waypoints.row(i) = waypoint.transpose();
        accumulateMinEsdf(waypoint);
      }
    }
    if (!std::isfinite(min_esdf)) {
      min_esdf = std::numeric_limits<double>::quiet_NaN();
    }

    py::dict out;
    out["success"] = result.success;
    out["failure_reason"] = result.failure_reason;
    out["objective"] = result.objective;
    out["optimizer_return_code"] = result.optimizer_return_code;
    out["duration"] = duration;
    out["cpp_optimize_time_ms"] = duration * 1000.0;
    out["min_esdf"] = min_esdf;
    out["samples"] = samples;
    out["waypoints"] = waypoints;
    out["timing_ms"] = result.timing_ms;
    out["dense_path_size"] = result.dense_path_size;
    out["sparse_waypoint_size"] = result.sparse_waypoint_size;
    out["mandatory_corner_count"] = result.mandatory_corner_count;
    out["local_end_is_goal"] = result.local_end_is_goal;
    out["planning_state"] = result.planning_state == minco_processor::MincoPipeline::PlanningState::kColdStart ?
      "COLD_START" : "HOT_START";
    out["hot_start_accepted"] = result.planning_state == minco_processor::MincoPipeline::PlanningState::kHotStart;
    out["hot_reject_reason"] = result.hot_reject_reason;
    out["history_plan_uid"] = committed_history_uid_.empty() ? py::none() : py::cast(committed_history_uid_);
    out["history_age_s"] = result.history_age_s;
    out["position_error"] = result.position_error;
    out["velocity_error"] = result.velocity_error;
    out["direction_dot"] = result.direction_dot;
    out["remaining_duration"] = result.remaining_duration;
    out["history_min_clearance"] = result.history_min_clearance;
    out["shifted_seed_valid"] = result.shifted_seed_valid;
    out["copied_waypoints"] = result.copied_waypoints;
    out["copied_durations"] = result.copied_durations;
    out["optimizer_iteration_count"] = result.optimizer_iteration_count;

    Eigen::MatrixXd sparse_waypoints(static_cast<int>(result.sparse_waypoints.size()), 3);
    for (int i = 0; i < sparse_waypoints.rows(); ++i) {
      sparse_waypoints.row(i) = result.sparse_waypoints[static_cast<size_t>(i)].transpose();
    }
    out["sparse_waypoints"] = sparse_waypoints;
    return out;
  }

  minco_processor::MincoPipeline pipeline_;
  std::shared_ptr<minco_processor::StaticSimEsdfMap2D> map_;
  minco_processor::MincoPipeline::Result last_result_;
  std::map<int, minco_processor::MincoPipeline::Result> proposals_;
  int next_proposal_id_{1};
  std::string committed_history_uid_;
};

}  // namespace

PYBIND11_MODULE(_minco_processor, m)
{
  py::class_<MincoProcessorPy>(m, "MincoProcessor")
    .def(py::init<>())
    .def("configure", &MincoProcessorPy::configure,
      py::arg("max_vel"),
      py::arg("max_acc"),
      py::arg("safe_dist"),
      py::arg("sample_dt"),
      py::arg("max_iterations"),
      py::arg("max_yaw_rate"),
      py::arg("penalty_weight_pos") = 1000.0,
      py::arg("penalty_weight_vel") = 1000.0,
      py::arg("penalty_weight_acc") = 10000.0,
      py::arg("penalty_weight_attractor") = 20.0,
      py::arg("time_weight") = 0.01,
      py::arg("time_barrier_weight") = 100.0)
    .def("set_static_esdf_2d", &MincoProcessorPy::set_static_esdf_2d,
      py::arg("distance"), py::arg("free"), py::arg("origin"), py::arg("resolution"))
    .def("optimize", &MincoProcessorPy::optimize,
      py::arg("guide_path"),
      py::arg("position"),
      py::arg("velocity"),
      py::arg("acceleration"),
      py::arg("yaw"),
      py::arg("yaw_rate"),
      py::arg("terminal_goal") = py::none())
    .def("optimize_preview", &MincoProcessorPy::optimizePreview,
      py::arg("guide_path"), py::arg("position"), py::arg("velocity"),
      py::arg("acceleration"), py::arg("yaw"), py::arg("yaw_rate"),
      py::arg("terminal_goal") = py::none())
    .def("commit_history", &MincoProcessorPy::commitHistory,
      py::arg("proposal_id"), py::arg("applied_time"))
    .def("discard_proposal", &MincoProcessorPy::discardProposal, py::arg("proposal_id"))
    .def("reset_history", &MincoProcessorPy::resetHistory);
}
