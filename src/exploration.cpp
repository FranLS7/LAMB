#include "exploration.h"

#include <iostream>

#include "MCX.h"

namespace lamb
{
  /**
 * @brief creates a set of points where only one dimension is modified given
 * some initial dimensions.
 * 
 * The initial dimensions are centered in the middle of the one-dimensional 
 * region we want to explore. The dimension to modify is identified with dim_id.
 * The max difference with this dimension is given by span and the difference
 * between contiguous points is given by jump.
 * 
 * @param initial_dims  the set of initial dimensions from where to start exploring.
 * @param dim_id        the dimension to modify.
 * @param span          the maximum distance to the initial value.
 * @param jump          the distance between contiguous points.
 * @return              the sets of generated dimensions.
 */
std::vector<std::vector<int>> genPoints1D(const std::vector<int> &initial_dims,
    const int dim_id, const int span, const int jump) {
      
  const int n_points = int(span / jump) * 2 + 1;
  std::vector<std::vector<int>> points;
 
  std::vector<int> base_dims = initial_dims; 
  base_dims[dim_id] -= static_cast<int>(span / jump) * jump;

  if (base_dims[dim_id] < 1) {
    std::cerr << "Error: Dimension " << dim_id << " below 1\n";
    return points;
  }

  auto dims_expl = base_dims;
  for (int point = 0; point < n_points; ++point) {
    dims_expl[dim_id] = base_dims[dim_id] + point * jump;
    points.push_back(dims_expl);
  }

  return points;
}

std::vector<std::vector<int>> genPoints2D(const std::vector<int> &initial_dims, 
    const std::vector<int>& dim_id, const int span, const int jump) {
  
  const int n_points = int(span / jump) * 2 + 1;
  std::vector<std::vector<int>> points;

  std::vector<int> base_dims = initial_dims;
  for (const auto &id : dim_id) {
    base_dims[id] -= static_cast<int>(span / jump) * jump;
    if (base_dims[id] < 1) {
      std::cerr << "Error: Dimension " << id << " below 1\n";
      return points;
    }
  }

  auto dims_expl = base_dims;
  for (int point_i = 0; point_i < n_points; ++point_i) {
    dims_expl[dim_id[0]] = base_dims[dim_id[0]] + point_i * jump;
    for (int point_j = 0; point_j < n_points; ++point_j) {
      dims_expl[dim_id[1]] = base_dims[dim_id[1]] + point_j * jump;
      points.push_back(dims_expl);
    }
  } 
}

dVector3D explore1D(const iVector1D &initial_dims, const int dim_id, const int span, const int jump, 
    const int iterations, const int n_threads, const bool individual) {
  mcx::MCX chain(initial_dims);
  dVector3D result;
  iVector2D points = genPoints1D(initial_dims, dim_id, span, jump);

  for (const auto& point : points) {
    chain.setDims(point);
    result.push_back(chain.executeAll(iterations, n_threads));
  }

  return result;
}

dVector3D explore2D(const iVector1D &initial_dims, const iVector1D &dim_id, const int span, 
    const int jump, const int iterations, const int n_threads, const bool individual) {
  
  mcx::MCX chain(initial_dims);
  dVector3D result;
  iVector2D points = genPoints2D(initial_dims, dim_id, span, jump);

  for (const auto &point : points) {
    std::cout << "Computing {";
    for (unsigned i = 0; i < point.size(); i++)
      std::cout << point[i] << ',';
    std::cout << "}\n";
    chain.setDims(point);
    result.push_back(chain.executeAll(iterations, n_threads));
  }

  return result;
}
  
} // namespace lamb


