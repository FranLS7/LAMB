#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mkl.h"

#include "common.h"
#include "MCX.h"

const int ALIGN = 64;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

int main(int argc, char** argv) {
  int ndim, n_operations;
  iVector1D dims;
  int iterations, n_threads;

  std::string root_dir;
  std::ofstream ofile;

  if (argc < 4) {
    std::cout << "Execution: " << argv[0] << "ndim d_0 .. d_ndim-1 iterations n_threads" 
              << std::endl;
    exit(-1);
  }
  else {
    ndim       = atoi(argv[1]);
    n_operations = ndim - 2;
    for (unsigned i = 0; i < ndim; ++i)
      dims.push_back(atoi(argv[i + 2]));

    iterations = atoi(argv[ndim + 2]);
    n_threads  = atoi(argv[ndim + 3]);
  }

  lamb::initialiseMKL();
  mcx::MCX chain(dims);
  dVector2D times = chain.executeAll(iterations, n_threads);
  
  std::cout << "Times with MCX: \n";
  for (const auto& t : times[0]) {
    std::cout << t << std::endl;
  }
  std::cout << "Median value: " << lamb::medianVector<double>(times[0]);
  std::cout << "=============================== \n\n";

   // ====================================================== //
   // ================ TEST SINGLE OPERATION =============== //
   // ====================================================== //
  dVector1D times_raw;
  times_raw.reserve(iterations);
  char trans = 'N';
  double one = 1.0;

  double *A, *B, *C;
  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), ALIGN);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), ALIGN);
  C = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);

  mkl_set_dynamic(false);
  mkl_set_num_threads(n_threads);

  for (int it = 0; it < iterations; ++it) {
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&trans, &trans, &dims[0], &dims[2], &dims[1], &one, A, &dims[0],
           B, &dims[1], &one, C, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();
    times_raw.push_back(std::chrono::duration<double>(end-start).count());
  }
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);

  std::cout << "Times with raw execution: \n";
  for (const auto& t : times_raw) {
    std::cout << t << std::endl;
  }
  std::cout << "Median value: " << lamb::medianVector<double>(times_raw);
  std::cout << "=============================== \n\n";

  return 0;
}

