#ifndef GEMM_CUBE
#define GEMM_CUBE

#include <fstream>
#include <string>
#include <vector>

#include "common.h"


namespace cube{

class Cube{
  int n_elements{0}; // total number of elements in the model.
  iVector2D axes; // vector containing the different axes.
  double *data = nullptr; // pointer to the actual data.
  // We could add flops here, taking the value from the chain.
  int n_threads{1};

public:
  Cube() = default;
  
  ~Cube();

  Cube(const iVector2D& axes, const int n_threads);

  double operator()(const int d0, const int d1, const int d2) const;

  double operator()(const iVector1D& _dims) const;

  void generate();

private:
  int findGreater(const int value, const iVector1D& axis) const;

  iVector1D clipDims(iVector1D& dims) const;

  bool isInCube(const iVector1D& dims, iVector1D& indices) const;

  double fetchValue(const iVector1D& indices) const;

  std::vector<std::pair<int,int>> getRanges(const iVector1D& dims, 
      iVector1D& indices) const;

  int searchLower (const iVector1D& v, const int value) const;

  double triInterpolation(const iVector1D& dims,
   const std::vector<std::pair<int,int>>& ranges) const;

  double biInterpolation() const;

  double linInterpolation() const;


};

} // end namespace cube

#endif