#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace minco_processor {

enum class QueryStatus
{
  kOk,
  kOutOfMap,
  kNonFiniteInput,
  kUnavailable,
};

struct QueryResult
{
  bool ok{false};
  QueryStatus status{QueryStatus::kUnavailable};
  double distance{0.0};
  Eigen::Vector3d gradient{Eigen::Vector3d::Zero()};
};

class EsdfMapInterface
{
public:
  virtual ~EsdfMapInterface() = default;

  virtual QueryResult query(const Eigen::Vector3d & pos) const = 0;

  virtual bool evaluate(const Eigen::Vector3d & pos, double & dist, Eigen::Vector3d & grad) const
  {
    const auto result = query(pos);
    dist = result.distance;
    grad = result.gradient;
    return result.ok;
  }

  virtual bool worldToMap(double, double, unsigned int &, unsigned int &) const { return false; }
  virtual void mapToWorld(unsigned int, unsigned int, double & wx, double & wy) const
  {
    wx = 0.0;
    wy = 0.0;
  }
  virtual unsigned int sizeX() const { return 0U; }
  virtual unsigned int sizeY() const { return 0U; }
  virtual double resolution() const { return 0.0; }
  virtual double originX() const { return 0.0; }
  virtual double originY() const { return 0.0; }
  virtual uint8_t value(unsigned int, unsigned int) const { return 0U; }
  virtual const unsigned char * values() const { return nullptr; }
  virtual bool copyValues(std::vector<unsigned char> & out) const
  {
    out.clear();
    return false;
  }
  virtual bool isValid(unsigned int, unsigned int) const { return false; }
  virtual bool isFree(unsigned int, unsigned int) const { return false; }
};

}  // namespace minco_processor
