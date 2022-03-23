#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "exploration.h"
#include "common.h"

int main(int argc, char **argv) {
  constexpr int ndim = 5;
  constexpr int n_operations = ndim - 2;
  // std::vector<int> initial_dims = {350, 225, 315, 1040, 370};
  std::vector<int> initial_dims (ndim);
  int dim_id, iterations, n_threads, jump, span;
  bool individual = true; // CAREFUL -- THIS WILL NOT WORK AS OF 16/3/2022

  std::string root_dir;
  std::ofstream ofile;

  if (argc != 12) {
    std::cout << "Execution: " << argv[0] << " d0 d1 d2 d3 d4 dim_id iterations n_threads span jump "
    "root_dir " << std::endl;
    exit(-1);
  }
  else {
    for (unsigned i = 0; i < initial_dims.size(); i++)
      initial_dims[i] = atoi(argv[i + 1]);
    dim_id      = atoi (argv[6]);
    iterations  = atoi (argv[7]);
    n_threads   = atoi (argv[8]);
    span        = atoi (argv[9]);
    jump        = atoi (argv[10]);
    root_dir.append (argv[11]);
  }

  iVector2D points = lamb::genPoints1D(initial_dims, dim_id, span, jump);
  dVector3D result = lamb::explore1D(initial_dims, dim_id, span, 
                                     jump, iterations, n_threads, individual);

  // write to files
  for (unsigned alg = 0; alg < result[0].size(); ++alg) {
    ofile.open(root_dir + std::string("alg") + std::to_string(alg) + std::string(".csv"));

    if (ofile.fail()) {
      std::cerr << ">> ERROR: opening output file for alg " << alg << '\n';
      exit(-1);
    }

    lamb::printHeaderTime(ofile, ndim, iterations, n_operations);
    for (unsigned point = 0; point < result.size(); ++point) {
      lamb::printTime(ofile, points[point], result[point][alg]);
    }

    ofile.close();
  }
  
  return 0;
}