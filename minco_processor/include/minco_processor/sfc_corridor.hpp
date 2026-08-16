#pragma once

#include <Eigen/Core>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "minco_processor/esdf_map.hpp"

namespace minco_processor {

// The local, dependency-free 2-D counterpart of SUPER's corridor generator.
// `normal.dot(point) + offset <= 0` defines the interior side of a face.
struct SfcHalfspace2D
{
  Eigen::Vector2d normal{Eigen::Vector2d::Zero()};
  double offset{0.0};
};

struct SfcConfig2D
{
  double bound_distance{1.0};
  double seed_line_max_length{2.0};
  double min_overlap_depth{0.02};
  double obstacle_sample_step{0.05};
  double halfspace_tolerance{1e-9};
  int max_seed_iterations{1000};
};

struct SfcPieceBinding2D
{
  int piece_index{-1};
  int cell_index{-1};
};

struct SfcCell2D
{
  int cell_index{-1};
  int guide_start_index{-1};
  int guide_end_index{-1};
  Eigen::Vector3d seed_start{Eigen::Vector3d::Zero()};
  Eigen::Vector3d seed_end{Eigen::Vector3d::Zero()};
  std::vector<SfcHalfspace2D> halfspaces;
  std::vector<Eigen::Vector2d> vertices;
  std::vector<Eigen::Vector2d> overlap_polygon;
  Eigen::Vector2d overlap_witness{Eigen::Vector2d::Constant(
    std::numeric_limits<double>::quiet_NaN())};
  double clearance_radius{0.0};
  double overlap_with_previous{0.0};
  double overlap_depth{0.0};
};

struct SfcCorridorReport
{
  bool valid{false};
  std::string reason{"SFC_NOT_RUN"};
  int query_count{0};
  int obstacle_sample_count{0};
  int repair_cell_count{0};
  double min_clearance{std::numeric_limits<double>::infinity()};
  double min_radius{std::numeric_limits<double>::infinity()};
  double min_overlap{std::numeric_limits<double>::infinity()};
  std::vector<SfcPieceBinding2D> piece_bindings;
};

class SuperplannerSfc2D
{
public:
  void setConfig(const SfcConfig2D & config) { config_ = config; }
  const SfcConfig2D & config() const { return config_; }

  SfcCorridorReport generate(
    const std::vector<Eigen::Vector3d> & guide,
    const std::shared_ptr<EsdfMapInterface> & map,
    double validation_safe_dist,
    double max_radius,
    double min_radius,
    double sample_step);

  bool contains(
    const Eigen::Vector3d & point,
    int * matched_cell = nullptr,
    double * signed_margin = nullptr) const;

  bool containsInCell(
    const Eigen::Vector3d & point,
    int cell_index,
    double * signed_margin = nullptr) const;

  const std::vector<SfcCell2D> & cells() const { return cells_; }

private:
  SfcConfig2D config_;
  std::vector<SfcCell2D> cells_;
};

}  // namespace minco_processor
