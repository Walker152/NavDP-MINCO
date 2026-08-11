#include "minco_processor/static_sim_esdf_map.hpp"

#include <algorithm>
#include <cmath>

namespace minco_processor {

void StaticSimEsdfMap2D::setMap(
  const Eigen::MatrixXd & distance,
  const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> & free,
  double origin_x,
  double origin_y,
  double resolution)
{
  distance_ = distance;
  free_ = free;
  origin_x_ = origin_x;
  origin_y_ = origin_y;
  resolution_ = resolution;
  const int value_count = static_cast<int>(std::max<Eigen::Index>(0, free_.rows() * free_.cols()));
  values_.assign(static_cast<size_t>(value_count), 0U);
  for (int y = 0; y < free_.rows(); ++y) {
    for (int x = 0; x < free_.cols(); ++x) {
      values_[static_cast<size_t>(y * free_.cols() + x)] = static_cast<unsigned char>(free_(y, x));
    }
  }
}

QueryResult StaticSimEsdfMap2D::query(const Eigen::Vector3d & pos) const
{
  QueryResult result;
  if (!pos.allFinite()) {
    result.status = QueryStatus::kNonFiniteInput;
    return result;
  }
  if (!hasMap()) {
    result.status = QueryStatus::kUnavailable;
    return result;
  }
  const double raw_x = (pos.x() - origin_x_) / resolution_;
  const double raw_y = (pos.y() - origin_y_) / resolution_;
  if (raw_x < 0.0 || raw_y < 0.0 ||
      raw_x >= static_cast<double>(distance_.cols()) ||
      raw_y >= static_cast<double>(distance_.rows())) {
    result.status = QueryStatus::kOutOfMap;
    return result;
  }
  const double gx = raw_x - 0.5;
  const double gy = raw_y - 0.5;
  double dist = 0.0;
  double dx_grid = 0.0;
  double dy_grid = 0.0;
  if (!interpolateWithGradient(gx, gy, dist, dx_grid, dy_grid)) {
    result.status = QueryStatus::kOutOfMap;
    return result;
  }

  result.ok = true;
  result.status = QueryStatus::kOk;
  result.distance = dist;
  if (analytic_gradient_) {
    result.gradient =
      Eigen::Vector3d(dx_grid / resolution_, dy_grid / resolution_, 0.0);
  } else {
    const auto sample = [this](double x, double y, double & out) {
        return interpolate(x, y, out);
      };
    double dx = 0.0;
    double dy = 0.0;
    double left = 0.0;
    double right = 0.0;
    if (sample(gx - 1.0, gy, left) && sample(gx + 1.0, gy, right)) {
      dx = (right - left) / (2.0 * resolution_);
    } else if (sample(gx, gy, left) && sample(gx + 1.0, gy, right)) {
      dx = (right - left) / resolution_;
    } else if (sample(gx - 1.0, gy, left) && sample(gx, gy, right)) {
      dx = (right - left) / resolution_;
    }
    if (sample(gx, gy - 1.0, left) && sample(gx, gy + 1.0, right)) {
      dy = (right - left) / (2.0 * resolution_);
    } else if (sample(gx, gy, left) && sample(gx, gy + 1.0, right)) {
      dy = (right - left) / resolution_;
    } else if (sample(gx, gy - 1.0, left) && sample(gx, gy, right)) {
      dy = (right - left) / resolution_;
    }
    result.gradient = Eigen::Vector3d(dx, dy, 0.0);
  }
  return result;
}

bool StaticSimEsdfMap2D::worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const
{
  if (!hasMap() || !std::isfinite(wx) || !std::isfinite(wy)) {
    return false;
  }
  const int ix = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
  const int iy = static_cast<int>(std::floor((wy - origin_y_) / resolution_));
  if (ix < 0 || iy < 0 || ix >= distance_.cols() || iy >= distance_.rows()) {
    return false;
  }
  mx = static_cast<unsigned int>(ix);
  my = static_cast<unsigned int>(iy);
  return true;
}

void StaticSimEsdfMap2D::mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const
{
  wx = origin_x_ + (static_cast<double>(mx) + 0.5) * resolution_;
  wy = origin_y_ + (static_cast<double>(my) + 0.5) * resolution_;
}

unsigned int StaticSimEsdfMap2D::sizeX() const
{
  return static_cast<unsigned int>(std::max<Eigen::Index>(0, distance_.cols()));
}
unsigned int StaticSimEsdfMap2D::sizeY() const
{
  return static_cast<unsigned int>(std::max<Eigen::Index>(0, distance_.rows()));
}
double StaticSimEsdfMap2D::resolution() const { return resolution_; }
double StaticSimEsdfMap2D::originX() const { return origin_x_; }
double StaticSimEsdfMap2D::originY() const { return origin_y_; }

uint8_t StaticSimEsdfMap2D::value(unsigned int mx, unsigned int my) const
{
  return isValid(mx, my) ? free_(static_cast<int>(my), static_cast<int>(mx)) : 0U;
}

const unsigned char * StaticSimEsdfMap2D::values() const
{
  return values_.empty() ? nullptr : values_.data();
}

bool StaticSimEsdfMap2D::copyValues(std::vector<unsigned char> & out) const
{
  out = values_;
  return !out.empty();
}

bool StaticSimEsdfMap2D::isValid(unsigned int mx, unsigned int my) const
{
  return hasMap() && mx < sizeX() && my < sizeY();
}

bool StaticSimEsdfMap2D::isFree(unsigned int mx, unsigned int my) const
{
  return isValid(mx, my) && free_(static_cast<int>(my), static_cast<int>(mx)) != 0U;
}

bool StaticSimEsdfMap2D::hasMap() const
{
  return resolution_ > 0.0 && distance_.rows() > 0 && distance_.cols() > 0 && free_.rows() == distance_.rows() &&
         free_.cols() == distance_.cols();
}

bool StaticSimEsdfMap2D::interpolate(double gx, double gy, double & distance) const
{
  double gradient_x = 0.0;
  double gradient_y = 0.0;
  return interpolateWithGradient(
    gx, gy, distance, gradient_x, gradient_y);
}

bool StaticSimEsdfMap2D::interpolateWithGradient(
  double gx, double gy, double & distance, double & gradient_x,
  double & gradient_y) const
{
  if (!hasMap() || gx < -0.5 || gy < -0.5 ||
      gx >= static_cast<double>(distance_.cols()) - 0.5 ||
      gy >= static_cast<double>(distance_.rows()) - 0.5) {
    return false;
  }
  const int cols = static_cast<int>(distance_.cols());
  const int rows = static_cast<int>(distance_.rows());
  const double x = std::clamp(gx, 0.0, static_cast<double>(cols - 1));
  const double y = std::clamp(gy, 0.0, static_cast<double>(rows - 1));
  const int x0 = std::min(static_cast<int>(std::floor(x)), cols - 1);
  const int y0 = std::min(static_cast<int>(std::floor(y)), rows - 1);
  const int x1 = std::min(x0 + 1, cols - 1);
  const int y1 = std::min(y0 + 1, rows - 1);
  const double tx = x - static_cast<double>(x0);
  const double ty = y - static_cast<double>(y0);
  const double d00 = distance_(y0, x0);
  const double d10 = distance_(y0, x1);
  const double d01 = distance_(y1, x0);
  const double d11 = distance_(y1, x1);
  distance = (1.0 - tx) * (1.0 - ty) * d00 + tx * (1.0 - ty) * d10 + (1.0 - tx) * ty * d01 + tx * ty * d11;
  gradient_x =
    (1.0 - ty) * (d10 - d00) + ty * (d11 - d01);
  gradient_y =
    (1.0 - tx) * (d01 - d00) + tx * (d11 - d10);
  return std::isfinite(distance) && std::isfinite(gradient_x) &&
         std::isfinite(gradient_y);
}

}  // namespace minco_processor
