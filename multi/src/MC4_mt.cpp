#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include "mkl.h"

#include <cube.h>
#include <common.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

bool computation_decision (int dims[], int ndim, int threshold);

// inline double gemm_flops (int d0, int d1, int d2);
// inline bool within_range (double x, double y, double lo_margin, double up_margin);
// inline bool compare_flops (double flops_0, double flops_1, double margin_flops);

void parenth_0 (int dims[], double times[], const int iterations);
void parenth_1 (int dims[], double times[], const int iterations);
void parenth_2 (int dims[], double times[], const int iterations);
void parenth_3 (int dims[], double times[], const int iterations);
void parenth_4 (int dims[], double times[], const int iterations);
void parenth_5 (int dims[], double times[], const int iterations);

int main (int argc, char **argv){

  int nparenth = 6, ndim = 5, dims[ndim];
  int min_size, max_size, iterations, nsamples, nthreads;

  int threshold;
  // double lo_margin, up_margin;

  // int overhead = 2;
  // string cube_filename = "../../cube/timings/";

  string output_file = "../timings/MC4_mt/";
  string out_files[nparenth];
  std::ofstream ofiles[nparenth];

  if (argc != 8){
    cout << "Execution: " << argv[0] << " min_size max_size iterations nsamples nthreads"
      /*" lo_margin up_margin cube_file*/
      " threshold output_file" << endl;
    return (-1);
  } else {
    min_size = atoi (argv[1]);
    max_size = atoi (argv[2]);
    iterations = atoi (argv[3]);
    nsamples = atoi (argv[4]);
    nthreads = atoi (argv[5]);
    /*lo_margin = atof (argv[6]);
    up_margin = atof (argv[7]);
    cube_filename.append (argv[8]);*/
    threshold = atoi (argv[6]);
    output_file.append (argv[7]);
  }

  double times[iterations];

  // lamb::GEMM_Cube kubo (cube_filename, overhead);

  // ==================================================================
  //   - - - - - - - - - - Opening output files - - - - - - - - - - -
  // ==================================================================
  for (int i = 0; i < nparenth; i++){
    // cout << output_file + to_string(i) + string(".csv") << endl;
    ofiles[i].open (output_file + to_string(i) + string(".csv"));
    if (ofiles[i].fail()){
      printf("Error opening the file for parenthesisation %d\n", i);
      exit(-1);
    }
    add_headers (ofiles[i], ndim, iterations);
  }

  auto start = std::chrono::high_resolution_clock::now();
  bool compute;
  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  mkl_set_num_threads(nthreads);
  initialise_BLAS();
  for (int i = 0; i < nsamples; ){
    printf("{");
    // Fix possible values at zero and below min_size
    for (int dim = 0; dim < ndim; dim++){
      dims[dim] = int(drand48() * (max_size - min_size)) + min_size;
      // printf("El valor %d es %d\n", i, dims[dim]);
      printf("%d,", dims[dim]);
    }
    printf("}\n");

    compute = computation_decision (dims, ndim, threshold);

    if (compute){
      parenth_0 (dims, times, iterations);
      add_line_ (ofiles[0], dims, ndim, times, iterations);
      parenth_1 (dims, times, iterations);
      add_line_ (ofiles[1], dims, ndim, times, iterations);
      parenth_2 (dims, times, iterations);
      add_line_ (ofiles[2], dims, ndim, times, iterations);
      parenth_3 (dims, times, iterations);
      add_line_ (ofiles[3], dims, ndim, times, iterations);
      parenth_4 (dims, times, iterations);
      add_line_ (ofiles[4], dims, ndim, times, iterations);
      parenth_5 (dims, times, iterations);
      add_line_ (ofiles[5], dims, ndim, times, iterations);
      i++;
    }

    if (i%10 == 0)
      printf("Vamos por la muestra %d\n", i);
  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());

  // ==================================================================
  //   - - - - - - - - - - Closing output files - - - - - - - - - - -
  // ==================================================================
  for (int i = 0; i < nparenth; i++){
    ofiles[i].close();
  }

  return 0;
}

bool computation_decision (int dims[], int ndim, int threshold){
  bool compute = false;

  for (int i = 0; i < ndim; i++){
    if (dims[i] <= threshold)
      compute = true;
  }

  return compute;
}





void parenth_0 (int dims[], double times[], const int iterations){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A1 = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  A2 = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  A3 = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  A4 = (double*)malloc(dims[3] * dims[4] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[4] * sizeof(double));

  M1 = (double*)malloc(dims[0] * dims[2] * sizeof(double));
  M2 = (double*)malloc(dims[0] * dims[3] * sizeof(double));

  initialise_BLAS();

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    A2[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    A3[i] = drand48();

  for (int i = 0; i < dims[3] * dims[4]; i++)
    A4[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    M2[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A1,
      &dims[0], A2, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[2], &one, M1,
      &dims[0], A3, &dims[2], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], A4, &dims[3], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(M1);
  free(M2);
  free(X);
}


void parenth_1 (int dims[], double times[], const int iterations){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A1 = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  A2 = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  A3 = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  A4 = (double*)malloc(dims[3] * dims[4] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[4] * sizeof(double));

  M1 = (double*)malloc(dims[1] * dims[3] * sizeof(double));
  M2 = (double*)malloc(dims[0] * dims[3] * sizeof(double));

  initialise_BLAS();

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    A2[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    A3[i] = drand48();

  for (int i = 0; i < dims[3] * dims[4]; i++)
    A4[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    M2[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, A2,
      &dims[1], A3, &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[1], &one, A1,
      &dims[0], M1, &dims[1], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], A4, &dims[3], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(M1);
  free(M2);
  free(X);
}


void parenth_2 (int dims[], double times[], const int iterations){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A1 = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  A2 = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  A3 = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  A4 = (double*)malloc(dims[3] * dims[4] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[4] * sizeof(double));

  M1 = (double*)malloc(dims[1] * dims[3] * sizeof(double));
  M2 = (double*)malloc(dims[1] * dims[4] * sizeof(double));

  initialise_BLAS();

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    A2[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    A3[i] = drand48();

  for (int i = 0; i < dims[3] * dims[4]; i++)
    A4[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[4]; i++)
    M2[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, A2,
      &dims[1], A3, &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[3], &one, M1,
      &dims[1], A4, &dims[3], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, A1,
      &dims[0], M2, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(M1);
  free(M2);
  free(X);
}


void parenth_3 (int dims[], double times[], const int iterations){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A1 = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  A2 = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  A3 = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  A4 = (double*)malloc(dims[3] * dims[4] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[4] * sizeof(double));

  M1 = (double*)malloc(dims[2] * dims[4] * sizeof(double));
  M2 = (double*)malloc(dims[1] * dims[4] * sizeof(double));

  initialise_BLAS();

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    A2[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    A3[i] = drand48();

  for (int i = 0; i < dims[3] * dims[4]; i++)
    A4[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[4]; i++)
    M2[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, A3,
      &dims[2], A4, &dims[3], &one, M1, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[2], &one, A2,
      &dims[1], M1, &dims[2], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, A1,
      &dims[0], M2, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(M1);
  free(M2);
  free(X);
}


void parenth_4 (int dims[], double times[], const int iterations){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A1 = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  A2 = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  A3 = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  A4 = (double*)malloc(dims[3] * dims[4] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[4] * sizeof(double));

  M1 = (double*)malloc(dims[0] * dims[2] * sizeof(double));
  M2 = (double*)malloc(dims[2] * dims[4] * sizeof(double));

  initialise_BLAS();

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    A2[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    A3[i] = drand48();

  for (int i = 0; i < dims[3] * dims[4]; i++)
    A4[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M2[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A1,
      &dims[0], A2, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, A3,
      &dims[2], A4, &dims[3], &one, M2, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M1,
      &dims[0], M2, &dims[2], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(M1);
  free(M2);
  free(X);
}



void parenth_5 (int dims[], double times[], const int iterations){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A1 = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  A2 = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  A3 = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  A4 = (double*)malloc(dims[3] * dims[4] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[4] * sizeof(double));

  M1 = (double*)malloc(dims[0] * dims[2] * sizeof(double));
  M2 = (double*)malloc(dims[2] * dims[4] * sizeof(double));

  initialise_BLAS();

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    A2[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    A3[i] = drand48();

  for (int i = 0; i < dims[3] * dims[4]; i++)
    A4[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M2[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, A3,
      &dims[2], A4, &dims[3], &one, M2, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A1,
      &dims[0], A2, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M1,
      &dims[0], M2, &dims[2], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  free(A1);
  free(A2);
  free(A3);
  free(A4);
  free(M1);
  free(M2);
  free(X);
}











