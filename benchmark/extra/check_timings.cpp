#include <float.h>
#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "mkl.h"

#include "common.h"

// ----------------- CONSTANTS ----------------- //
const int NPAR = 1;
const int NDIM = 3;

const int ALIGN = 64;
// --------------------------------------------- //


extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void operations_1 (std::vector<int>& dims, std::vector<double>& times, 
const int iterations, const int n_threads);

void operations_2 (std::vector<int>& dims, std::vector<double>& times, 
const int iterations, const int n_threads);

void operations_3 (std::vector<int>& dims, std::vector<double>& times, 
const int iterations, const int n_threads);

void operations_4 (std::vector<int>& dims, std::vector<double>& times, 
const int iterations, const int n_threads, int size);


int main (int argc, char **argv) {
  std::vector<int> dims;
  int n_threads, iterations;

  std::ofstream ofile;
  std::string output_file;

  if (argc != 7) {
    printf("Execution: %s d0 d1 d2 iterations n_threads output_file\n", argv[0]);
    exit(-1);
  }
  else {
    for (int i = 1; i <= NDIM; i++)
      dims.push_back(atoi(argv[i]));

    iterations = atoi (argv[4]);
    n_threads = atoi (argv[5]);
    output_file = argv[6];
  }

  ofile.open (output_file, std::ofstream::out);
  if (ofile.fail()){
    printf("Error opening the output file...\n");
    exit(-1);
  }
  lamb::printHeaderTime(ofile, NDIM, iterations);

  std::vector<double> times;
  times.resize(iterations);

  mkl_set_dynamic(false);
  mkl_set_num_threads(n_threads);

  lamb::initialiseMKL();
  printf("Executing the first set of operations..\n");
  operations_1 (dims, times, iterations, n_threads);
  lamb::printTime(ofile, dims, times);

  printf("Executing the second set of operations..\n");
  operations_2 (dims, times, iterations, n_threads);
  lamb::printTime(ofile, dims, times);

  printf("Executing the third set of operations..\n");
  operations_3 (dims, times, iterations, n_threads);
  lamb::printTime(ofile, dims, times);

  // printf("Executing the fourth set of operations..\n");
  // for (int size = 2000; size >= 100; size -= 100){
  //   printf("%d..\n", size);
  //   operations_4 (dims, times, iterations, nthreads, size);
  //   add_line (ofile, dims, ndim, times, iterations);
  // }

  ofile.close();


  return 0;
}


void operations_1 (std::vector<int> &dims, std::vector<double>& times, 
    const int iterations, const int n_threads) {
  double *A, *B, *X;
  double one = 1.0;
  char transpose = 'N';
  int nCF = 3;

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), ALIGN);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), ALIGN);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    for (int i = 0; i < nCF; i++)
      lamb::cacheFlush (n_threads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_ (&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
      &dims[0], B, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
    
    for (int i = 0; i < dims[0] * dims[2]; i++)
      X[i] = drand48();
  }

  mkl_free(A);
  mkl_free(B);
  mkl_free(X);

}



void operations_2 (std::vector<int>& dims, std::vector<double>& times, 
    const int iterations, const int n_threads) {
  double *A, *B, *X;
  double one = 1.0;
  char transpose = 'N';
  int nCF = 3;

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), ALIGN);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), ALIGN);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    for (int i = 0; i < nCF; i++)
      lamb::cacheFlush (n_threads);

    lamb::initialiseMKL();

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_ (&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
      &dims[0], B, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();

    for (int i = 0; i < dims[0] * dims[2]; i++)
      X[i] = drand48();
  }

  mkl_free(A);
  mkl_free(B);
  mkl_free(X);
}


void operations_3 (std::vector<int>& dims, std::vector<double>& times, 
    const int iterations, const int n_threads) {
  double *A, *B, *Y, *M1, *X;
  double one = 1.0;
  char transpose = 'N';
  int nCF = 3;

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), ALIGN);
  Y = (double*)mkl_malloc(dims[1] * dims[1] * sizeof(double), ALIGN);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), ALIGN);
  M1 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), ALIGN);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[1]; i++)
    Y[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    for (int i = 0; i < nCF; i++)
      lamb::cacheFlush (n_threads);

    // initialiseBLAS();
    dgemm_ (&transpose, &transpose, &dims[1], &dims[2], &dims[1], &one, Y,
      &dims[1], B, &dims[1], &one, M1, &dims[1]);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_ (&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
      &dims[0], M1, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();

    for (int i = 0; i < dims[0] * dims[2]; i++)
      X[i] = drand48();
  }

  mkl_free(A);
  mkl_free(B);
  mkl_free(Y);
  mkl_free(M1);
  mkl_free(X);
}


void operations_4 (std::vector<int>& dims, std::vector<double>& times, 
    const int iterations, const int n_threads, const int size) {
  double *A, *B, *X;
  double one = 1.0;
  char transpose = 'N';

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), ALIGN);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), ALIGN);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    lamb::cacheFlush (n_threads);
    lamb::cacheFlush (n_threads);
    lamb::cacheFlush (n_threads);
    lamb::initialiseMKL(size);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_ (&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
      &dims[0], B, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2 - time1).count();
    for (int i = 0; i < dims[0] * dims[2]; i++)
      X[i] = drand48();
  }

  mkl_free(A);
  mkl_free(B);
  mkl_free(X);
}







