#include <float.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include "mkl.h"

#include <common.h>
#include <omp.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*,
  double*, int*, double*, double*, int*);

void compute_points (int min_size, int max_size, int npoints, int *points);

int main (int argc, char** argv){
  int ndim = 3;
  int dims[ndim];
  std::vector<int> points = {20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000};
  int iterations, max_nthreads;

  string output_file;

  if (argc != 4){
    cout << "Execution: " << argv[0] << " iterations max_nthreads output_file" << endl;
    return (-1);
  }
  else {
    iterations = atoi (argv[1]);
    max_nthreads = atoi (argv[2]);
    output_file = argv[3];
  }

  double* times = (double*)malloc(iterations * sizeof(double));
  std::ofstream ofiles[max_nthreads];

  // ==================================================================
  //   - - - - - - - - - - Opening output files - - - - - - - - - - -
  // ==================================================================
  for (int i = 0; i < max_nthreads; i++){
    // cout << output_file + to_string(i) + string(".csv") << endl;
    ofiles[i].open (output_file + to_string(i+1) + string(".csv"));
    if (ofiles[i].fail()){
      printf("Error opening the file %d\n", i+1);
      exit(-1);
    }
    add_headers (ofiles[i], ndim, iterations);
  }

  auto start = std::chrono::high_resolution_clock::now();
  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  double one = 1.0;
  char transpose = 'N';
  double *A, *B, *C;
  int alignment = 64;
  mkl_set_dynamic(0);
  mkl_set_num_threads(max_nthreads);
  initialise_BLAS();


  for (int d : points){
    for (int i = 0; i < ndim; i++)
      dims[i] = d;
    printf(" >> Computing [%d, %d, %d]\n", d, d, d);

    A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);

    for (int i = 0; i < dims[0] * dims[1]; i++)
      A[i] = drand48 ();

    B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);

    for (int i = 0; i < dims[1] * dims[2]; i++)
      B[i] = drand48 ();

    C = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);

    for (int i = 0; i < dims[0] * dims[2]; i++)
      C[i] = drand48 ();

    for (int nthreads = 1; nthreads <= max_nthreads; nthreads++){
      cout << "\t ⁄⁄ Computing with " << nthreads << " cores!" << endl;
      mkl_set_num_threads(nthreads);
      for (int it = 0; it < iterations; it++){
        cache_flush_par (nthreads);
        cache_flush_par (nthreads);
        cache_flush_par (nthreads);

        auto time1 = std::chrono::high_resolution_clock::now();
        dgemm_ (&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
                &dims[0], B, &dims[1], &one, C, &dims[0]);
        auto time2 = std::chrono::high_resolution_clock::now();

        times[it] = std::chrono::duration<double>(time2 - time1).count();
      }
      add_line (ofiles[nthreads - 1], dims, ndim, times, iterations);
    }

    mkl_free (A);
    mkl_free (B);
    mkl_free (C);
  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("The total execution has taken %f seconds\n", std::chrono::duration<double>(end - start).count());


  for (int i = 0; i < max_nthreads; i++)
    ofiles[i].close();

  free(times);

  return 0;
}


// Compute points in the form of a parabola: y = ax2 + bx + c
// This parabola has its vertex in x=0 --> 'b' = 0.
// For the same reason --> 'c' = min_size.
// 'a' is left to compute, then.
void compute_points (int min_size, int max_size, int npoints, int *points){
  double a = (max_size - min_size) / pow(npoints - 1, 2.0);

  for (int i = 0; i < npoints; i++){
    points[i] = int(a * pow(i, 2.0)) + min_size;
    if (i > 0 && points[i] <= points[i - 1])
      points[i] = points[i-1] + 1;
    // printf("points[%d] : %d\n", i, points[i]);
  }
}



