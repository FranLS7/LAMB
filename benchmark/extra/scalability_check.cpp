#include <float.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mkl.h>

#include "common.h"

// ----------------- CONSTANTS ----------------- //
const int NDIM = 3;
const int ALIGN = 64;
// --------------------------------------------- //

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*,
  double*, int*, double*, double*, int*);

int main (int argc, char** argv){
  std::vector<int> dims (NDIM, 0);
  std::vector<int> points = {20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000};
  int iterations, max_nthreads;

  std::string output_file;

  if (argc != 4){
    std::cout << "Execution: " << argv[0] << " iterations max_nthreads output_file" << std::endl;
    return (-1);
  }
  else {
    iterations = atoi (argv[1]);
    max_nthreads = atoi (argv[2]);
    output_file = argv[3];
  }

  std::vector<double> times (iterations, 0.0f);
  std::ofstream ofiles[max_nthreads];

  // Opening output files.
  for (int i = 0; i < max_nthreads; i++){
    ofiles[i].open (output_file + std::to_string(i + 1) + std::string(".csv"));
    if (ofiles[i].fail()){
      printf("Error opening the file %d\n", i+1);
      exit(-1);
    }
    lamb::printHeaderTime(ofiles[i], NDIM, iterations);
  }

  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  double one = 1.0;
  char transpose = 'N';
  double *A, *B, *C;
  MKL_Set_Dynamic(0);
  MKL_Set_Num_Threads(max_nthreads);

  lamb::initialiseBLAS();

  for (int const size : points){
    for (auto &dim : dims)
      dim = size;
    std::cout << " >> Computing " << "[" << size << ", " << size << ", " << size << "]\n";

    A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), ALIGN);
    B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), ALIGN);
    C = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);

    for (int i = 0; i < dims[0] * dims[1]; i++)
      A[i] = drand48 ();

    for (int i = 0; i < dims[1] * dims[2]; i++)
      B[i] = drand48 ();

    for (int i = 0; i < dims[0] * dims[2]; i++)
      C[i] = drand48 ();

    for (int n_threads = 1; n_threads <= max_nthreads; n_threads++){
      std::cout << "\t ⁄⁄ Computing with " << n_threads << " cores!" << std::endl;
      MKL_Set_Num_Threads(n_threads);

      for (int it = 0; it < iterations; it++){
        lamb::cacheFlush (n_threads);
        lamb::cacheFlush (n_threads);
        lamb::cacheFlush (n_threads);

        auto time1 = std::chrono::high_resolution_clock::now();
        dgemm_ (&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
                &dims[0], B, &dims[1], &one, C, &dims[0]);
        auto time2 = std::chrono::high_resolution_clock::now();

        times[it] = std::chrono::duration<double>(time2 - time1).count();
      }
      lamb::printTime(ofiles[n_threads - 1], dims, times);
    }

    mkl_free (A);
    mkl_free (B);
    mkl_free (C);
  }

  for (int i = 0; i < max_nthreads; i++)
    ofiles[i].close();

  return 0;
}



