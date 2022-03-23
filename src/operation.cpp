/**
 * @file operation.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief implementation of algorithms to solve A*trans(A)*B
 * @version 0.1
 * @date 2022-01-27
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "operation.h"

#include <chrono>
#include <string>
#include <vector>

#include "mkl.h"

#include "anomalies.h"
#include "common.h"

const int ALIGN = 64;

dVector1D syrkAndSymm(const Matrix& A, const Matrix& B, Matrix& X) {
  dVector1D times (BENCH_REPS);

  Matrix M;
  M.name = "M";
  M.rows = A.rows;
  M.columns = A.rows;
  M.data = (double*)mkl_malloc(M.rows * M.columns * sizeof(double), ALIGN);

  for (int it = 0; it < BENCH_REPS; ++it) {
    for (int i = 0; i < M.rows * M.columns; ++i) M.data[i] = 0.0;
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);

    auto begin = std::chrono::high_resolution_clock::now();
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
        A.data, A.rows, 0.0, M.data, M.rows);

    cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
        M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
    auto end = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(end - begin).count();
  }

  mkl_free(M.data);
  return times;
}

dVector1D syrkAndGemm(const Matrix& A, const Matrix& B, Matrix& X) {
  dVector1D times (BENCH_REPS);

  Matrix M;
  M.name = "M";
  M.rows = A.rows;
  M.columns = A.rows;
  M.data = (double*)mkl_malloc(M.rows * M.columns * sizeof(double), ALIGN);

  for (int it = 0; it < BENCH_REPS; ++it) {
    for (int i = 0; i < M.rows * M.columns; ++i) M.data[i] = 0.0;
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);

    auto begin = std::chrono::high_resolution_clock::now();
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
        A.data, A.rows, 0.0, M.data, M.rows);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
        1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
    auto end = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(end - begin).count();
  }

  mkl_free(M.data);
  return times;
}

dVector1D gemmAndSymm(const Matrix& A, const Matrix& B, Matrix& X) {
  dVector1D times (BENCH_REPS);

  Matrix M;
  M.name = "M";
  M.rows = A.rows;
  M.columns = A.rows;
  M.data = (double*)mkl_malloc(M.rows * M.columns * sizeof(double), ALIGN);

  for (int it = 0; it < BENCH_REPS; ++it) {
    for (int i = 0; i < M.rows * M.columns; ++i) M.data[i] = 0.0;
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);

    auto begin = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
        1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);
    
    cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
        M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
    auto end = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(end - begin).count();
  }

  mkl_free(M.data);
  return times;
}

dVector1D lgemmAndGemm(const Matrix& A, const Matrix& B, Matrix& X) {
  dVector1D times (BENCH_REPS);

  Matrix M;
  M.name = "M";
  M.rows = A.rows;
  M.columns = A.rows;
  M.data = (double*)mkl_malloc(M.rows * M.columns * sizeof(double), ALIGN);

  for (int it = 0; it < BENCH_REPS; ++it) {
    for (int i = 0; i < M.rows * M.columns; ++i) M.data[i] = 0.0;
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);

    auto begin = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
        1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
        1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
    auto end = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(end - begin).count();
  }

  mkl_free(M.data);
  return times;
}

dVector1D rgemmAndGemm(const Matrix& A, const Matrix& B, Matrix& X) {
  dVector1D times (BENCH_REPS);

  Matrix M;
  M.name = "M";
  M.rows = A.columns;
  M.columns = B.columns;
  M.data = (double*)mkl_malloc(M.rows * M.columns * sizeof(double), ALIGN);

  for (int it = 0; it < BENCH_REPS; ++it) {
    for (int i = 0; i < M.rows * M.columns; ++i) M.data[i] = 0.0;
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);
    lamb::cacheFlush(N_THREADS);

    auto begin = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M.rows, M.columns, B.rows,
        1.0, A.data, A.rows, B.data, B.rows, 0.0, M.data, M.rows);
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows, M.columns, M.rows,
        1.0, A.data, A.rows, M.data, M.rows, 0.0, X.data, X.rows);
    auto end = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(end - begin).count();
  }

  mkl_free(M.data);
  return times;
}

unsigned long flopsSyrkAndSymm(const int m, const int k, const int n) {
  unsigned long flops = 0;
  flops += static_cast<unsigned long>(m) * static_cast<unsigned long>(m + 1) * 
      static_cast<unsigned long>(k);
  flops += static_cast<unsigned long>(2 * m) * static_cast<unsigned long>(m) *
      static_cast<unsigned long>(n);
  return flops;
}


unsigned long flopsSyrkAndGEMM(const int m, const int k, const int n) {
  unsigned long flops = 0;
  flops += static_cast<unsigned long>(m) * static_cast<unsigned long>(m + 1) *
      static_cast<unsigned long>(k);
  flops += static_cast<unsigned long>(2 * m) * static_cast<unsigned long>(m) *
      static_cast<unsigned long>(n);
  return flops;
}


unsigned long flopsGemmAndSymm(const int m, const int k, const int n) {
  unsigned long flops = 0;
  flops += static_cast<unsigned long>(2 * m) * static_cast<unsigned long>(k) *
      static_cast<unsigned long>(m);
  flops += static_cast<unsigned long>(2 * m) * static_cast<unsigned long>(m) * 
      static_cast<unsigned long>(n);
  return flops;
}


unsigned long flopsLGemmAndGemm(const int m, const int k, const int n) {
  unsigned long flops = 0;
  flops += static_cast<unsigned long>(2 * m) * static_cast<unsigned long>(k) * 
      static_cast<unsigned long>(m);
  flops += static_cast<unsigned long>(2 * m) * static_cast<unsigned long>(m) * 
      static_cast<unsigned long>(n);
  return flops;
}


unsigned long flopsRGemmAndGemm(const int m, const int k, const int n) {
  unsigned long flops = 0;
  flops += static_cast<unsigned long>(2 * k) * static_cast<unsigned long>(m) * 
      static_cast<unsigned long>(n);
  flops += static_cast<unsigned long>(2 * m) * static_cast<unsigned long>(k) * 
      static_cast<unsigned long>(n);
  return flops;
}


dVector2D executeAll(const int m, const int k, const int n) {
  dVector2D result;

  mkl_set_dynamic(false);
  mkl_set_num_threads(N_THREADS);

  Matrix A, B, X;

  A.name = "A";
  A.rows = m;
  A.columns = k;
  A.data = (double*)mkl_malloc(A.rows * A.columns * sizeof(double), ALIGN);
  for (int i = 0; i < A.rows * A.columns; i++) A.data[i] = drand48();

  B.name = "B";
  B.rows = m;
  B.columns = n;
  B.data = (double*)mkl_malloc(B.rows * B.columns * sizeof(double), ALIGN);
  for (int i = 0; i < B.rows * B.columns; i++) B.data[i] = drand48();

  X.name = "X";
  X.rows = m;
  X.columns = n;
  X.data = (double*)mkl_malloc(X.rows * X.columns * sizeof(double), ALIGN);
  for (int i = 0; i < X.rows * X.columns; i++) X.data[i] = 0.0;

  result.push_back(syrkAndSymm(A, B, X));
  for (int i = 0; i < X.rows * X.columns; i++) X.data[i] = 0.0;
  result.push_back(syrkAndGemm(A, B, X));
  for (int i = 0; i < X.rows * X.columns; i++) X.data[i] = 0.0;
  result.push_back(gemmAndSymm(A, B, X));
  for (int i = 0; i < X.rows * X.columns; i++) X.data[i] = 0.0;
  result.push_back(lgemmAndGemm(A, B, X));
  for (int i = 0; i < X.rows * X.columns; i++) X.data[i] = 0.0;
  result.push_back(rgemmAndGemm(A, B, X));

  mkl_free(A.data);
  mkl_free(B.data);
  mkl_free(X.data);

  return result;
}

dVector2D executeAll(const iVector1D& dims) {
  return executeAll(dims[0], dims[1], dims[2]);
}


std::vector<unsigned long> flopsAll(const int m, const int k, const int n) {
  std::vector<unsigned long> result;

  result.push_back(flopsSyrkAndSymm(m, k, n));
  result.push_back(flopsSyrkAndGEMM(m, k, n));
  result.push_back(flopsGemmAndSymm(m, k, n));
  result.push_back(flopsLGemmAndGemm(m, k, n));
  result.push_back(flopsRGemmAndGemm(m, k, n));

  return result;
}

std::vector<unsigned long> flopsAll(const iVector1D& dims){
  return flopsAll(dims[0], dims[1], dims[2]);
}


