/**
 * @file profile_SYRK.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief This program is used to generate performance profile data for SYRK
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

const int ALIGN = 64;

std::vector<unsigned long> getFlopsSYRK (const iVector2D &points);

dVector2D explore2DSYRK(const iVector2D &points, const int iterations, const int n_threads);

int main(int argc, char **argv) {
  int ndim = 2;
  std::vector<int> initial_dims(ndim);

  int iterations, n_threads, span, jump;

  std::string root_dir;
  std::ofstream ofile;

  if (argc != 8) {
    std::cout << "Execution: " << argv[0] << " d0 d1 iterations n_threads span jump "
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

  iVector1D dim_id {0, 1};
  iVector2D points = lamb::genPoints2D(initial_dims, dim_id, span, jump);
  std::vector<unsigned long> flops = getFlopsSYRK(points);

  dVector2D result = explore2DSYRK(points, iterations, n_threads);

  // write to files
  ofile.open(root_dir + std::string("syrk.csv"));

  if (ofile.fail()) {
    std::cerr << ">> ERROR: opening output file for symm\n";
    exit(-1);
  }

  lamb::printHeaderTime(ofile, ndim, iterations, true);
  for (unsigned i = 0; i < result.size(); ++i) {
    lamb::printTime(ofile, points[i], result[i], flops[i]);
  }

  return 0;
}

dVector2D explore2DSYRK(const iVector2D &points, const int iterations, const int n_threads) {
  dVector2D result;

  mkl_set_dynamic(false);
  mkl_set_num_threads(n_threads);

  dVector1D point_times(iterations);

  double *A, *C;

  for (const auto &point : points) {
    std::cout << "{";
    for (const auto &d : point) { 
      std::cout << d << ',';
    }
    std::cout << "}\n";
    A = (double*)mkl_malloc(point[0] * point[1] * sizeof(double), ALIGN);
    C = (double*)mkl_malloc(point[0] * point[0] * sizeof(double), ALIGN);

    for (int i = 0; i < point[0] * point[1]; i++) A[i] = drand48();

    for (int it = 0; it < iterations; ++it) {
      for (int i = 0; i < point[0] * point[0]; i++) C[i] = drand48();
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto time1 = std::chrono::high_resolution_clock::now();
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, point[0], point[1], 1.0, A, point[0], 
      1.0, C, point[0]);
      auto time2 = std::chrono::high_resolution_clock::now();
      point_times[it] = std::chrono::duration<double>(time2-time1).count();
    }
    mkl_free(A);
    mkl_free(C);
    result.push_back(point_times);
  }
  return result;
}


std::vector<unsigned long> getFlopsSYRK (const iVector2D &points) {
  std::vector<unsigned long> flops;

  for (auto const &point : points)
    flops.push_back(static_cast<unsigned long>(point[0] * point[0] * point[1]));
  
  return flops;
}