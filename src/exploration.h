#ifndef EXPLORE_FUNC
#define EXPLORE_FUNC

#include "common.h"

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
    const int dim_id, const int span, const int jump);

std::vector<std::vector<int>> genPoints2D(const std::vector<int> &initial_dims, 
    const iVector1D &dim_id, const int span, const int jump);
  
dVector3D explore1D(const iVector1D &initial_dims, const int dim_id, const int span, const int jump, 
    const int iterations, const int n_threads, const bool individual=false);

dVector3D explore2D(const iVector1D &initial_dims, const iVector1D &dim_id, const int span, 
    const int jump, const int iterations, const int n_threads, const bool individual=false);

// create function that executes a set of given points with the algs we choose as input.


} // namespace lamb






#endif