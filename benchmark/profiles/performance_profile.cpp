/**
 * @file performance_profile.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief this program is used to generate performance profile data
 * @version 0.1
 * @date 2022-01-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mkl.h"

#include "common.h"
#include "exploration.h"
#include "MCX.h"


int main(int argc, char **argv) {
  int ndim = 3;
  std::vector<int> initial_dims(ndim);

  int iterations, n_threads, span, jump;

  std::string root_dir;
  std::ofstream ofile;

  if (argc != 9) {
    std::cout << "Execution: " << argv[0] << " d0 d1 d2 iterations n_threads span jump "
    "root_dir " << std::endl;
    exit(-1);
  }
  else {
    for (unsigned i = 0; i < ndim; ++i) 
      initial_dims[i] = atoi(argv[i + 1]);
    
    iterations = atoi(argv[ndim + 1]);
    n_threads  = atoi(argv[ndim + 2]);
    span       = atoi(argv[ndim + 3]);
    jump       = atoi(argv[ndim + 4]);
    root_dir = argv[ndim + 5];
  }

  iVector1D dim_id {0, 2};
  iVector2D points = lamb::genPoints2D(initial_dims, dim_id, span, jump);
  std::vector<unsigned long> flops;

  mcx::MCX chain(initial_dims);
  for (const auto &point : points) {
    chain.setDims(point);
    flops.push_back(chain.getFLOPs()[0]);
  }

  dVector3D result = lamb::explore2D(initial_dims, dim_id, span, jump, iterations, n_threads);

  // write to files
  ofile.open(root_dir + std::string("gemm_1000.csv"));

  if (ofile.fail()) {
      std::cerr << ">> ERROR: opening output file for gemm\n";
      exit(-1);
  }
  
  lamb::printHeaderTime(ofile, ndim, iterations, true);
  for (unsigned i = 0; i < result.size(); ++i) {
    lamb::printTime(ofile, points[i], result[i][0], flops[i]);
  }

  ofile.close();
}