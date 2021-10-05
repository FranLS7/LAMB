#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include <random>
#include <algorithm>
#include "mkl.h"

#include <cube.h>
#include <common.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

bool computation_decision (int dims[], int ndim, int threshold);

bool check_anomaly(int dims[], double* times, int nparenth, int iterations,
  double lo_margin, double ratio);

unsigned long long int compute_flops (int dims[], int parenth);
unsigned long long int flops_parenth_0 (int dims[]);
unsigned long long int flops_parenth_1 (int dims[]);

// bool is_anomaly (unsigned long long int flops_a, unsigned long long int flops_b,
//   double times_a[], double times_b[], int iterations, double ratio);

void parenth_0 (int dims[], double times[], const int iterations, const int nthreads);
void parenth_1 (int dims[], double times[], const int iterations, const int nthreads);

int main (int argc, char **argv){

  int nparenth = 2, ndim = 4, dims[ndim];
  int min_size, max_size, iterations, nsamples, nthreads;
  double *times;

  int threshold;
  double lo_margin, ratio;
  unsigned long long int counter = 0;

  // int overhead = 2;
  // string cube_filename;
  string output_file;
  std::ofstream ofiles[nparenth];

  if (argc != 10){
    cout << "Execution: " << argv[0] << " min_size max_size iterations nsamples nthreads"
    " threshold lo_margin ratio output_file" << endl;
    return (-1);
  } else {
    min_size = atoi (argv[1]);
    max_size = atoi (argv[2]);
    iterations = atoi (argv[3]);
    nsamples = atoi (argv[4]);
    nthreads = atoi (argv[5]);
    threshold = atoi (argv[6]);
    lo_margin = atof (argv[7]);
    ratio = atof (argv[8]);
    output_file.append (argv[9]);
  }

  times = (double*)malloc(nparenth * iterations * sizeof(double));


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
  mkl_set_dynamic (0);
  mkl_set_num_threads (nthreads);
  initialise_BLAS();
  for (int i = 0; i < nsamples; ){
    for (int dim = 0; dim < ndim; dim++){
      dims[dim] = int(drand48() * (max_size - min_size)) + min_size;
    }

    compute = computation_decision (dims, ndim, threshold);
    if (compute){
      counter++;
      printf(">> %llu points, %d anomalies found -- {", counter, i);
      for (int i = 0; i < ndim; i++)
        printf("%d, ", dims[i]);
      printf("}\n");

      parenth_0 (dims, times, iterations, nthreads);
      parenth_1 (dims, &times[iterations], iterations, nthreads);

      if (check_anomaly(dims, times, nparenth, iterations, lo_margin, ratio)){
        i++;
        printf("Anomaly found! So far we have found %d\n", i);
        for (int i = 0; i < nparenth; i++)
          add_line_ (ofiles[i], dims, ndim, &times[i * iterations], iterations);

      }
    }


  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());
  printf("TOTAL computed points: %llu\n", counter);

  // ==================================================================
  //   - - - - - - - - - - Closing output files - - - - - - - - - - -
  // ==================================================================
  for (int i = 0; i < nparenth; i++){
    ofiles[i].close();
  }
  free(times);

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



bool check_anomaly(int dims[], double* times, int nparenth, int iterations,
  double lo_margin, double ratio){
  bool anomaly = false;
  unsigned seed;
  unsigned long long int flops[nparenth];
  double median[nparenth];

  for (int p = 0; p < nparenth; p++){
    flops[p] = compute_flops (dims, p);
    // printf("flops parenth %d: %llu\n", p, flops[p]);
    median[p] = median_array (&times[p * iterations], iterations);
  }

  // Shuffle the timings belonging to each parenthesisation
  for (int p = 0; p < nparenth; p++){
    seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    shuffle (&times[p * iterations], &times[p * iterations] + iterations,
      default_random_engine (seed));
  }

  for (int i = 0; i < nparenth; i++){
    for (int j = i + 1; j < nparenth; j++){
      // TODO: OBTAIN THE ANOMALIES
      // printf("\tComparing parenths {%d,%d} with score %f\n", i, j, score(median[i], median[j]));

      if ((score (median[i], median[j]) >= lo_margin) &&
          is_anomaly (flops[i], flops[j],
                      &times[i * iterations],
                      &times[j * iterations],
                      iterations, ratio)){
          // printf("Anomaly found for parenths {%d,%d}\n", i, j);
          anomaly = true;
          break;
      }
    }
  }
  return anomaly;
}

unsigned long long int compute_flops (int dims[], int parenth){
  if (parenth == 0)
    return flops_parenth_0 (dims);
  else if (parenth == 1)
    return flops_parenth_1 (dims);
  else {
    std::cout << ">> ERROR: wrong parenthesisation for MC_flops." << std::endl;
    return -1;
  }
}

unsigned long long int flops_parenth_0 (int dims[]){
  unsigned long long int result;
  result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
  result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  return result * 2;
}

unsigned long long int flops_parenth_1 (int dims[]){
  unsigned long long int result;
  result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[3]);
  return result * 2;
}


void parenth_0 (int dims[], double times[], const int iterations, const int nthreads){
  double *A, *B, *C, *M1, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  C = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), alignment);

  // Initialise each value for each matrix
  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    C[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    X[i] = drand48();

  // Compute the operation as many times as iterations specifies
  for (int it = 0; it < iterations; it++){
    // Flush/scrub the cache so we are working with 'cold' data
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
      &dims[0], B, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[2], &one, M1,
      &dims[0], C, &dims[2], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();
    // Store the timings in one array so we can write them all at once
    // in the output file.
    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  // Free the previously allocated memory.
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
  mkl_free(M1);
  mkl_free(X);
}

void parenth_1 (int dims[], double times[], const int iterations, const int nthreads){
  double *A, *B, *C, *M1, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  C = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), alignment);

  // Initialise each value for each matrix
  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    C[i] = drand48();

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    X[i] = drand48();

  // Compute the operation as many times as iterations specifies
  for (int it = 0; it < iterations; it++){
    // Flush/scrub the cache so we are working with 'cold' data
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, B,
      &dims[1], C, &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[1], &one, A,
      &dims[0], M1, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();
    // Store the timings in one array so we can write them all at once
    // in the output file.
    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  // Free the previously allocated memory.
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
  mkl_free(M1);
  mkl_free(X);
}













