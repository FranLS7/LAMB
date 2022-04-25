#ifndef PERF_MODEL_H
#define PERF_MODEL_H

#include <vector>

#include "common.h"


namespace lamb {

class Model {
  int n_elements{0}; // total number of elements in the model
  iVector2D axes; // vector containing the different axes
  double* data = nullptr; // pointer to the actual data

public:
  Model() = default;

  ~Model();

  Model(const iVector1D& axis);

  Model(const iVector2D& axes);

  double operator()(const int d0) const;

  double operator()(const int d0, const int d1) const;

  double operator()(const int d0, const int d1, const int d2) const;

  double operator()(const std::vector<int>& dims) const;

private:
  double linInterpolation(const int dim, const iVector1D& x, const dVector1D& y) const;

  double biInterpolation(const iVector1D& dims, const iVector2D& points,
    const dVector2D& z) const;

  double triInterpolation(const iVector1D& dims, const iVector2D& coords,
    const dVector3D w) const;

  int findGreater(const int value, const iVector1D& axis) const;

  // bool find(const int value, const iVector1D& axis) const;
};

} // end namespace lamb



#endif