#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <vector>

#include "minco_processor/esdf_map.hpp"

namespace minco_processor {

class StaticSimEsdfMap2D final : public EsdfMapInterface
{
public:
  void setMap(
    const Eigen::MatrixXd & distance,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> & free,
    double origin_x,
    double origin_y,
    double resolution);
  void setAnalyticGradient(bool enabled) { analytic_gradient_ = enabled; }

  QueryResult query(const Eigen::Vector3d & pos) const override;
  bool worldToMap(double wx, double wy, unsigned int & mx, unsigned int & my) const override;
  void mapToWorld(unsigned int mx, unsigned int my, double & wx, double & wy) const override;
  unsigned int sizeX() const override;
  unsigned int sizeY() const override;
  double resolution() const override;
  double originX() const override;
  double originY() const override;
  uint8_t value(unsigned int mx, unsigned int my) const override;
  const unsigned char * values() const override;
  bool copyValues(std::vector<unsigned char> & out) const override;
  bool isValid(unsigned int mx, unsigned int my) const override;
  bool isFree(unsigned int mx, unsigned int my) const override;

private:
  bool hasMap() const;
  bool interpolate(double gx, double gy, double & distance) const;
  bool interpolateWithGradient(
    double gx, double gy, double & distance, double & gradient_x,
    double & gradient_y) const;

  Eigen::MatrixXd distance_;
  Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> free_;
  std::vector<unsigned char> values_;
  double origin_x_{0.0};
  double origin_y_{0.0};
  double resolution_{0.0};
  bool analytic_gradient_{false};
};

}  // namespace minco_processor
