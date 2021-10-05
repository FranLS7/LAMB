#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include <vector>

#include "mkl.h"
#include "common.h"
#include <MC4.h>

using namespace std;

const int ALIGN = 64;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void MC4 (int parenth, double** matrices, int dims[], double times[],
  const int iterations, const int n_measures, const int nthreads){
    if (parenth == 0){
      MC4_parenth0 (matrices, dims, times, iterations, n_measures, nthreads);
    } else if (parenth == 1){
      MC4_parenth1 (matrices, dims, times, iterations, n_measures, nthreads);
    } else if (parenth == 2){
      MC4_parenth2 (matrices, dims, times, iterations, n_measures, nthreads);
    } else if (parenth == 3){
      MC4_parenth3 (matrices, dims, times, iterations, n_measures, nthreads);
    } else if (parenth == 4){
      MC4_parenth4 (matrices, dims, times, iterations, n_measures, nthreads);
    } else if (parenth == 5){
      MC4_parenth5 (matrices, dims, times, iterations, n_measures, nthreads);
    } else
      printf("Error: No matching parenthesisation\n");
}

void MC4_parenth0 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads){
  double *M1, *M2;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), alignment);

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48 ();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    M2[i] = drand48 ();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, matrices[0],
      &dims[0], matrices[1], &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[2], &one, M1,
      &dims[0], matrices[2], &dims[2], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], matrices[3], &dims[3], &one, matrices[4], &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free (M1);
  mkl_free (M2);
}

void MC4_parenth1 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads){
  double *M1, *M2;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), alignment);

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48 ();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    M2[i] = drand48 ();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, matrices[1],
      &dims[1], matrices[2], &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[1], &one, matrices[0],
      &dims[0], M1, &dims[1], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], matrices[3], &dims[3], &one, matrices[4], &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free (M1);
  mkl_free (M2);
}


void MC4_parenth2 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads){
  double *M1, *M2;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[1] * dims[4] * sizeof(double), alignment);

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48 ();

  for (int i = 0; i < dims[1] * dims[4]; i++)
    M2[i] = drand48 ();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, matrices[1],
      &dims[1], matrices[2], &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[3], &one, M1,
      &dims[1], matrices[3], &dims[3], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, matrices[0],
      &dims[0], M2, &dims[1], &one, matrices[4], &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free (M1);
  mkl_free (M2);
}


void MC4_parenth3 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads){
  double *M1, *M2;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  M1 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[1] * dims[4] * sizeof(double), alignment);

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M1[i] = drand48 ();

  for (int i = 0; i < dims[1] * dims[4]; i++)
    M2[i] = drand48 ();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, matrices[2],
      &dims[2], matrices[3], &dims[3], &one, M1, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[2], &one, matrices[1],
      &dims[1], M1, &dims[2], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, matrices[0],
      &dims[0], M2, &dims[1], &one, matrices[4], &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free (M1);
  mkl_free (M2);
}


void MC4_parenth4 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads){
  double *M1, *M2;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), alignment);

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48 ();

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M2[i] = drand48 ();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, matrices[0],
      &dims[0], matrices[1], &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, matrices[2],
      &dims[2], matrices[3], &dims[3], &one, M2, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M1,
      &dims[0], M2, &dims[2], &one, matrices[4], &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free (M1);
  mkl_free (M2);
}

void MC4_parenth5 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads){
  double *M1, *M2;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  M1 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M1[i] = drand48 ();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M2[i] = drand48 ();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, matrices[2],
      &dims[2], matrices[3], &dims[3], &one, M1, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, matrices[0],
      &dims[0], matrices[1], &dims[1], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M2,
      &dims[0], M1, &dims[2], &one, matrices[4], &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free (M1);
  mkl_free (M2);
}

std::vector<double*> generateMatrices (std::vector<int> &dims){
  std::vector<double*> matrices;
  for (int i = 0; i < static_cast<int>(dims.size() - 1); i++) {
    double *matrix = (double*)mkl_malloc (dims[i] * dims[i+1] * sizeof(double), ALIGN);

    for (int j = 0; j < dims[j] * dims[j+1]; j++)
      matrix[j] = drand48();

    matrices.push_back (matrix);
  }
  return matrices;
}

void free_matrices (std::vector<double*> matrices){
  for (auto &mat : matrices)
    mkl_free(mat);
}

std::vector<std::vector<double>> MC4_execute_par (std::vector<int> &dims, std::vector<int> &parenths,
  const int iterations, const int n_threads){

  std::vector<std::vector<double>> both_times;
  std::vector<double*> imatrices = generateMatrices (dims);

  // Helper variable
  std::vector<double> times;

  // Execute both parenthesisations
  for (auto p : parenths) {
    times = MC4_execute (dims, imatrices, p, iterations, n_threads);
    both_times.push_back (times);
  }
  free_matrices (imatrices);
  return both_times;
}

std::vector<double> MC4_execute (std::vector<int> &dims, std::vector<double*> imatrices,
  const int parenth, const int iterations, const int n_threads){
  switch (parenth) {
    case 0  : return MC4_p0 (dims, imatrices, iterations, n_threads);
    case 1  : return MC4_p1 (dims, imatrices, iterations, n_threads);
    case 2  : return MC4_p2 (dims, imatrices, iterations, n_threads);
    case 3  : return MC4_p3 (dims, imatrices, iterations, n_threads);
    case 4  : return MC4_p4 (dims, imatrices, iterations, n_threads);
    case 5  : return MC4_p5 (dims, imatrices, iterations, n_threads);
    default : return std::vector<double>(); //empty vector
  }
}

std::vector<double> MC4_p0 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads){
  std::vector<double> times;
  double *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for intermediate and final matrices.
  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);
  M2 = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    M2[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, imatrices[0],
      &dims[0], imatrices[1], &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[2], &one, M1,
      &dims[0], imatrices[2], &dims[2], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], imatrices[3], &dims[3], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);

  return times;
}

std::vector<double> MC4_p1 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads){
  std::vector<double> times;
  double *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for intermediate and final matrices.
  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), ALIGN);
  M2 = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    M2[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, imatrices[1],
      &dims[1], imatrices[2], &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[1], &one, imatrices[0],
      &dims[0], M1, &dims[1], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], imatrices[3], &dims[3], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);

  return times;
}

std::vector<double> MC4_p2 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads){
  std::vector<double> times;
  double *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for intermediate and final matrices.
  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), ALIGN);
  M2 = (double*)mkl_malloc(dims[1] * dims[4] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[4]; i++)
    M2[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, imatrices[1],
      &dims[1], imatrices[2], &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[3], &one, M1,
      &dims[1], imatrices[3], &dims[3], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, imatrices[0],
      &dims[0], M2, &dims[1], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);

  return times;
}

std::vector<double> MC4_p3 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads){
  std::vector<double> times;
  double *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for intermediate and final matrices.
  M1 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), ALIGN);
  M2 = (double*)mkl_malloc(dims[1] * dims[4] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[1] * dims[4]; i++)
    M2[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, imatrices[2],
      &dims[2], imatrices[3], &dims[3], &one, M1, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[2], &one, imatrices[1],
      &dims[1], M1, &dims[2], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, imatrices[0],
      &dims[0], M2, &dims[1], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);

  return times;
}

std::vector<double> MC4_p4 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads){
  std::vector<double> times;
  double *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for intermediate and final matrices.
  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);
  M2 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M2[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, imatrices[0],
      &dims[0], imatrices[1], &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, imatrices[2],
      &dims[2], imatrices[3], &dims[3], &one, M2, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M1,
      &dims[0], M2, &dims[2], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);

  return times;
}

std::vector<double> MC4_p5 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads){
  std::vector<double> times;
  double *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for intermediate and final matrices.
  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);
  M2 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[2] * dims[4]; i++)
    M2[i] = drand48();

  for (int i = 0; i < dims[0] * dims[4]; i++)
    X[i] = drand48();

  for (int it = 0; it < iterations; it++){
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);
    cache_flush_par (n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, imatrices[2],
      &dims[2], imatrices[3], &dims[3], &one, M2, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, imatrices[0],
      &dims[0], imatrices[1], &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M1,
      &dims[0], M2, &dims[2], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);

  return times;
}

/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been declared nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_0 (int dims[], double times[], const int iterations, const int n_measures,
    const int nthreads){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A1 = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  A2 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  A3 = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  A4 = (double*)mkl_malloc(dims[3] * dims[4] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), alignment);

  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), alignment);

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
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A1,
      &dims[0], A2, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[2], &one, M1,
      &dims[0], A3, &dims[2], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], A4, &dims[3], &one, X, &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free(A1);
  mkl_free(A2);
  mkl_free(A3);
  mkl_free(A4);
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);
}


/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been declared nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_1 (int dims[], double times[], const int iterations, const int n_measures,
    const int nthreads){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A1 = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  A2 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  A3 = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  A4 = (double*)mkl_malloc(dims[3] * dims[4] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), alignment);

  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), alignment);

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
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, A2,
      &dims[1], A3, &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[1], &one, A1,
      &dims[0], M1, &dims[1], &one, M2, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[3], &one, M2,
      &dims[0], A4, &dims[3], &one, X, &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free(A1);
  mkl_free(A2);
  mkl_free(A3);
  mkl_free(A4);
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);
}

/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been declared nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_2 (int dims[], double times[], const int iterations, const int n_measures,
    const int nthreads){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A1 = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  A2 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  A3 = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  A4 = (double*)mkl_malloc(dims[3] * dims[4] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), alignment);

  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[1] * dims[4] * sizeof(double), alignment);

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
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, A2,
      &dims[1], A3, &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[3], &one, M1,
      &dims[1], A4, &dims[3], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, A1,
      &dims[0], M2, &dims[1], &one, X, &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free(A1);
  mkl_free(A2);
  mkl_free(A3);
  mkl_free(A4);
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);
}

/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been declared nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_3 (int dims[], double times[], const int iterations, const int n_measures,
    const int nthreads){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A1 = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  A2 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  A3 = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  A4 = (double*)mkl_malloc(dims[3] * dims[4] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), alignment);

  M1 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[1] * dims[4] * sizeof(double), alignment);

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
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, A3,
      &dims[2], A4, &dims[3], &one, M1, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[1], &dims[4], &dims[2], &one, A2,
      &dims[1], M1, &dims[2], &one, M2, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[1], &one, A1,
      &dims[0], M2, &dims[1], &one, X, &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free(A1);
  mkl_free(A2);
  mkl_free(A3);
  mkl_free(A4);
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);
}

/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been declared nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_4 (int dims[], double times[], const int iterations, const int n_measures,
    const int nthreads){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A1 = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  A2 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  A3 = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  A4 = (double*)mkl_malloc(dims[3] * dims[4] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), alignment);

  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), alignment);

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
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A1,
      &dims[0], A2, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, A3,
      &dims[2], A4, &dims[3], &one, M2, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M1,
      &dims[0], M2, &dims[2], &one, X, &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free(A1);
  mkl_free(A2);
  mkl_free(A3);
  mkl_free(A4);
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);
}

/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been declared nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_5 (int dims[], double times[], const int iterations, const int n_measures,
    const int nthreads){
  double *A1, *A2, *A3, *A4, *M1, *M2, *X;
  double one = 1.0;
  char transpose = 'N';
  int alignment = 64;

  // Memory allocation for all the matrices
  A1 = (double*)mkl_malloc(dims[0] * dims[1] * sizeof(double), alignment);
  A2 = (double*)mkl_malloc(dims[1] * dims[2] * sizeof(double), alignment);
  A3 = (double*)mkl_malloc(dims[2] * dims[3] * sizeof(double), alignment);
  A4 = (double*)mkl_malloc(dims[3] * dims[4] * sizeof(double), alignment);
  X = (double*)mkl_malloc(dims[0] * dims[4] * sizeof(double), alignment);

  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), alignment);
  M2 = (double*)mkl_malloc(dims[2] * dims[4] * sizeof(double), alignment);

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
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, A3,
      &dims[2], A4, &dims[3], &one, M2, &dims[2]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A1,
      &dims[0], A2, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, M1,
      &dims[0], M2, &dims[2], &one, X, &dims[0]);
    auto time4 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time4-time1).count();
  }

  mkl_free(A1);
  mkl_free(A2);
  mkl_free(A3);
  mkl_free(A4);
  mkl_free(M1);
  mkl_free(M2);
  mkl_free(X);
}

/**
 * Computes the amount of FLOPs for one parenthesisation.
 *
 * @param dims    The array with the matrices dimensions.
 * @param parenth The parenthesisation for which FLOPs is computed.
 * @return        The number of FLOPs for a given parenthesisation.
 */
unsigned long long int MC4_flops (int dims[], int parenth){
  switch (parenth) {
    case 0 : return MC4_flops_0 (dims);
    case 1 : return MC4_flops_1 (dims);
    case 2 : return MC4_flops_2 (dims);
    case 3 : return MC4_flops_3 (dims);
    case 4 : return MC4_flops_4 (dims);
    case 5 : return MC4_flops_4 (dims);
    default : std::cout << ">> ERROR: wrong parenthesisation for MC_flops." << std::endl;
              return -1;
  }
}

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The array with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions.
 */
unsigned long long int MC4_flops_0 (int dims[]){
  unsigned long long int result;
  result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
  result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  return result * 2;
}

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The array with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions.
 */
unsigned long long int MC4_flops_1 (int dims[]){
  unsigned long long int result;
  result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  return result * 2;
}

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The array with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions.
 */
unsigned long long int MC4_flops_2 (int dims[]){
  unsigned long long int result;
  result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[1]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]);
  return result * 2;
}

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The array with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions.
 */
unsigned long long int MC4_flops_3 (int dims[]){
  unsigned long long int result;
  result = uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  result += uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[4]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]);
  return result * 2;
}

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The array with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions.
 */
unsigned long long int MC4_flops_4 (int dims[]){
  unsigned long long int result;
  result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
  result += uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[4]);
  return result * 2;
}


// These functions are the same but are applied upon vectors.
unsigned long long int MC4_flops (std::vector<int> dims, int parenth){
  switch (parenth) {
    case 0 : return MC4_flops_0 (dims);
    case 1 : return MC4_flops_1 (dims);
    case 2 : return MC4_flops_2 (dims);
    case 3 : return MC4_flops_3 (dims);
    case 4 : return MC4_flops_4 (dims);
    case 5 : return MC4_flops_4 (dims);
    default : std::cout << ">> ERROR: wrong parenthesisation for MC_flops." << std::endl;
              return -1;
  }
}

unsigned long long int MC4_flops_0 (std::vector<int> dims){
  unsigned long long int result;
  result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
  result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  return result * 2;
}

unsigned long long int MC4_flops_1 (std::vector<int> dims){
  unsigned long long int result;
  result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  return result * 2;
}

unsigned long long int MC4_flops_2 (std::vector<int> dims){
  unsigned long long int result;
  result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[1]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]);
  return result * 2;
}

unsigned long long int MC4_flops_3 (std::vector<int> dims){
  unsigned long long int result;
  result = uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  result += uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[4]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]);
  return result * 2;
}

unsigned long long int MC4_flops_4 (std::vector<int> dims){
  unsigned long long int result;
  result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
  result += uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]);
  result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[4]);
  return result * 2;
}