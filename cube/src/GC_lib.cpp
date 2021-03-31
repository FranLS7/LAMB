#include <float.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <string>
#include <math.h>

#include "common.h"

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void compute_points (int min_size, int max_size, int npoints, int *points);


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
      cout << "Error opening the output file.." << endl;
      exit(-1);
    }

    for (int i = 0; i < ndim; i++){
      points[i] = (int*)malloc(npoints[i] * sizeof(int));
      compute_points(min_size[i], max_size[i], npoints[i], points[i]);
    }

    add_headers_cube(ofile, ndim, min_size, max_size, npoints, nsamples, nthreads, points);

    times = (double*)malloc(nsamples * sizeof(double));

    printf("Computation starts...\n");


    initialise_BLAS();
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
            cache_flush();
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



