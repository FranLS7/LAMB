#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "common.h"

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

extern "C"{
  void CubeGEMM (int *min_size, int *max_size, int *npoints, int nsamples, int nthreads, char *output_filex){
    printf("The output file is %s\n", output_filex);
    printf("The min sizes are: %d %d %d\n", min_size[0], min_size[1], min_size[2]);
    printf("The max sizes are: %d %d %d\n", max_size[0], max_size[1], max_size[2]);
    printf("The number of points are: %d %d %d\n", npoints[0], npoints[1], npoints[2]);
    printf("The number of cores to use is: %d\n", nthreads);

    double *A, *B, *C;
    double one = 1.0;
    int ndim = 3, dims[ndim], *points[ndim];
    double *times;

    char transpose = 'N';

    std::ofstream ofile;
    ofile.open (output_filex, std::ios::out);

    if (ofile.fail()){
      std::cout << "Error opening the output file.." << std::endl;
      exit(-1);
    }

    for (int i = 0; i < ndim; i++){
      points[i] = (int*)malloc(npoints[i] * sizeof(int));
      compute_points(min_size[i], max_size[i], npoints[i], points[i]);
    }

    print_header_cube(ofile, ndim, min_size, max_size, npoints, nsamples, nthreads, points);

    times = (double*)malloc(nsamples * sizeof(double));

    printf("Computation starts...\n");


    lamb::initialiseBLAS();
    for (int d0 = 0; d0 < npoints[0]; d0++){ //jump between slices
      dims[0] = points[0][d0];
      for (int d1 = 0; d1 < npoints[1]; d1++){ //jump between rows within same file
        dims[1] = points[1][d1];
        for (int d2 = 0; d2 < npoints[2]; d2++){ //jump between contiguous elements within row
          dims[2] = points[2][d2];

          A = (double*)malloc(dims[0] * dims[1] * sizeof(double));
          B = (double*)malloc(dims[1] * dims[2] * sizeof(double));
          C = (double*)malloc(dims[0] * dims[2] * sizeof(double));

          for (int i = 0; i < dims[0] * dims[1]; i++)
            A[i] = drand48();

          for (int i = 0; i < dims[1] * dims[2]; i++)
            B[i] = drand48();

          for (int i = 0; i < dims[0] * dims[2]; i++)
            C[i] = drand48();

          for (int it = 0; it < nsamples; it++){
            cacheFlush();
            auto time1 = std::chrono::high_resolution_clock::now();
            dgemm_(&transpose, &transpose, &dims[0], &dims[2],
              &dims[1], &one, A, &dims[0], B, &dims[1],
              &one, C, &dims[0]);
            auto time2 = std::chrono::high_resolution_clock::now();

            times[it] = std::chrono::duration<double>(time2 - time1).count();
          }

          add_line (ofile, dims, ndim, times, nsamples);

          free(A);
          free(B);
          free(C);
        } // n loop
        printf("Just finished {d0,d1} = {%d,%d}\n", points[0][d0], points[1][d1]);
      } // k loop
    } // m loop

    free(times);

    for (int i = 0; i < ndim; i++)
      free(points[i]);

    ofile.close();
    printf("C++ Function is about to finish now\n");

  }
}

/**
 * Adds the headers to an output file for a generated cube. Format:
 * # nthreads
 * # d0:    min_size, max_size, npoints
 * # ...
 * # di-1:  min_size, max_size, npoints
 * # d0:    points
 * # ...
 * # di-1:  points
 * # | ndim dims || nsamples samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param ndim      The number of dimensions in the cube (usually 3).
 * @param min_size  Allocated memory with the min sizes for each dimension.
 * @param max_size  Allocated memory with the max sizes for each dimension.
 * @param npoints   Allocated memory with the number of points for each
 *      dimension (might be different depending on the dimension).
 * @param nsamples  The number of samples to store in the output file.
 * @param nthreads  The number of threads to use in the computation - is stored.
 * @param points    Allocated memory with the points for each dimension (might
 *      be different depending on the dimension).
 */
void GEMM_Cube::print_header_cube (std::ofstream &ofile, const int ndim, const int* min_size,
    const int* max_size, const int* npoints, const int nsamples, const int nthreads,
    const int **points) {
  ofile << "# " << nthreads << '\n';

  for (int i = 0; i < ndim; i++){
    ofile << "# " << min_size[i] << "," << max_size[i] << "," << npoints[i] << '\n';

    ofile << "# ";
    for (int ii = 0; ii < npoints[i]; ii++){
      ofile << points[i][ii];
      if (ii == npoints[i] - 1)
        ofile << '\n';
      else
        ofile << ',';
    }
  }

  for (int i = 0; i < ndim, i++) 
    ofile << 'd' << i << ',';
  
  for (int i = 0; i < nsamples; i++) {
    ofile << "sample_" << i;

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
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

