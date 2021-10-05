#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include <mkl.h>

#include <common.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void operations_1 (int dims[], double times[], int iterations, int nthreads);
void operations_2 (int dims[], double times[], int iterations, int nthreads);
void operations_3 (int dims[], double times[], int iterations, int nthreads);
void operations_4 (int dims[], double times[], int iterations, int nthreads, int size);


int main (int argc, char **argv){

  int ndim = 3, dims[ndim];
  int nthreads, iterations;

  std::ofstream ofile;
  string output_file;

  double *times;

  if (argc != 7){
    printf("Execution: %s d0 d1 d2 iterations nthreads output_file\n", argv[0]);
    exit(-1);
  }
  else{
    for (int i = 0; i < ndim; i++)
      dims[i] = atoi(argv[i + 1]);

    iterations = atoi (argv[4]);
    nthreads = atoi (argv[5]);
    output_file = argv[6];
  }

  ofile.open (output_file, std::ofstream::out);
  if (ofile.fail()){
    printf("Error opening the output file...\n");
    exit(-1);
  }
  add_headers (ofile, ndim, iterations);

  times = (double*)malloc(iterations * sizeof(double));

  mkl_set_dynamic(false);
  mkl_set_num_threads(nthreads);

  initialise_mkl();
  printf("Executing the first set of operations..\n");
  operations_1 (dims, times, iterations, nthreads);
  add_line (ofile, dims, ndim, times, iterations);

  printf("Executing the second set of operations..\n");
  operations_2 (dims, times, iterations, nthreads);
  add_line (ofile, dims, ndim, times, iterations);

  printf("Executing the third set of operations..\n");
  operations_3 (dims, times, iterations, nthreads);
  add_line (ofile, dims, ndim, times, iterations);

  // printf("Executing the fourth set of operations..\n");
  // for (int size = 2000; size >= 100; size -= 100){
  //   printf("%d..\n", size);
  //   operations_4 (dims, times, iterations, nthreads, size);
  //   add_line (ofile, dims, ndim, times, iterations);
  // }

  free(times);
  ofile.close();


  return 0;
}


void operations_1 (int dims[], double times[], int iterations, int nthreads){
  double *A, *B, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;
  int nCF = 3;

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    for (int i = 0; i < nCF; i++)
      cache_flush_par (nthreads);

    // initialise_BLAS();
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



void operations_2 (int dims[], double times[], int iterations, int nthreads){
  double *A, *B, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;
  int nCF = 3;

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    for (int i = 0; i < nCF; i++)
      cache_flush_par (nthreads);

    initialise_mkl();
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



void operations_3 (int dims[], double times[], int iterations, int nthreads){
  double *A, *B, *Y, *M1, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;
  int nCF = 3;

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  Y = (double*)mkl_malloc(dims[1] * dims[1] * sizeof(double), alignment);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  M1 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);

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
      cache_flush_par (nthreads);

    // initialise_BLAS();
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


void operations_4 (int dims[], double times[], int iterations, int nthreads, int size){
  double *A, *B, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  A = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  B = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);

  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    initialise_mkl_variable(size);
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







