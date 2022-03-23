#include "cube.h"

#include <algorithm>
#include <string>
#include <vector>

#include "common.h"
#include "MCX.h"

namespace cube{
  Cube::~Cube(){
    if (data)
      delete[] data;
  }

  Cube::Cube(const iVector2D& axes, const int n_threads){
    this-> axes = axes;
    this->n_threads = n_threads;

    n_elements = 1;
    for (const auto& axis : axes)
      n_elements *= static_cast<int>(axis.size());
  }

  double Cube::operator()(const int d0, const int d1, const int d2) const {
    iVector1D dims {d0, d1, d2};
    return this->operator()(dims);
  }

  double Cube::operator()(const iVector1D& _dims) const {
    if (!data)
      return -1.0;

    iVector1D dims = _dims;
    iVector1D indices {-1, -1, -1};

    clipDims(dims);
    // in_cube = isInCube(dims);

    if (isInCube(dims, indices))
      return fetchValue(indices);

    // TODO: change ranges to 2D vector of ints.
    std::vector<std::pair<int,int>> ranges = getRanges(dims, indices);

    iVector2D coords;
    for (unsigned i = 0; i < ranges.size(); ++i) {
      coords[i].push_back(axes[i][ranges[i].first]);
      coords[i].push_back(axes[i][ranges[i].second]);
    }
    ranges[0][0];
    dVector3D lattice;
    for (unsigned m = 0; m < ranges[0].size(); ++m) {
      for (auto &k : ranges[1]) {
        for (auto &n : ranges[2]) {
          lattice
        }
      }
    }

    return triInterpolation(dims, ranges);

    // for (unsigned i = 0; i < in_cube.size(); ++i) {
    //   if (!in_cube[i])
    // }



    



  }

  void Cube::generate() {
    this->data = new double[n_elements];
    iVector1D gemm_dims {1, 1, 1};
    mcx::MCX chain(gemm_dims);
    dVector2D times;

    int m_points = static_cast<int>(axes[0].size());
    int k_points = static_cast<int>(axes[1].size());
    int n_points = static_cast<int>(axes[2].size());

    for (int m = 0; m < m_points; ++m) {
      for (int k = 0; k < k_points; ++k) {
        for (int n = 0; n < n_points; ++n) {
          gemm_dims = {axes[0][m], axes[1][k], axes[2][n]};
          chain.setDims(gemm_dims);
          times = chain.executeAll(BENCH_REPS, n_threads);
          data[m * k_points * n_points + k * n_points + n] = lamb::medianVector(times[0]);
        }
      }
    }
  }

  iVector1D Cube::clipDims(iVector1D& dims) const {
    for (unsigned i = 0; i < axes.size(); ++i) {
      if (dims[i] < axes[i].front())
        dims[i] = axes[i].front();
      else if (dims[i] > axes[i].back())
        dims[i] = axes[i].back();
    }
    return dims;
  }

  bool Cube::isInCube(const iVector1D& dims, iVector1D& indices) const {
    bool found = true;

    for (unsigned i = 0; i < dims.size(); ++i) {
      auto it = std::find(axes[i].begin(), axes[i].end(), dims[i]);
      if (it == axes[i].end()) {
        found = false;
        indices[i] = -1;
      }
      else {
        indices[i] = it - axes[i].begin();
      }
    }
    return found;
  }

  double Cube::fetchValue(const iVector1D& indices) const {
    return data[indices[0] * axes[1].size() * axes[2].size() +
                indices[1] * axes[2].size() +
                indices[2]];
  }

  std::vector<std::pair<int,int>> Cube::getRanges(const iVector1D& dims, 
      iVector1D& indices) const {
    
    std::vector<std::pair<int,int>> ranges(indices.size());

    for (unsigned i = 0; i < indices.size(); ++i) {
      if (indices[i] != -1) 
        ranges[i] = {indices[i], indices[i]}; // if this dimension is in the cube
          // assign the ranges to be exactly the same value -> no interpolation is needed
          // in this dimension.s
      
      else {
        ranges[i].first = searchLower(axes[i], dims[i]);
        ranges[i].second = ranges[i].first + 1;
      }
    }
    return ranges;
  }

  int Cube::searchLower (const iVector1D& v, const int value) const {
    int idx = 0;
    for (idx = 0; idx < static_cast<int>(v.size()) - 1; idx++)
      if (v[idx + 1] > value) break;
    return idx;
  }

  double Cube::triInterpolation(const iVector1D& dims,
   const std::vector<std::pair<int,int>>& ranges) const {
     dVector1D temporary;
     temporary.push_back(biInterpolation());
  }

  double Cube::linInterpolation(const int dim, ) const {

  }


}