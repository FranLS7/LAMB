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
#include <vector>
#include <thread>

#include <common.h>
#include <MC4.h>
#include <omp.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*,
  double*, int*, double*, double*, int*);

bool computation_decision (int dims[], int ndim, int threshold);


void allocate_imatrices (double ** imatrices, int dims[], int ndim, int alignment);
void allocate_interm_matrices (double ** interm_matrices, int dims[], int alignment);

void gemm (int id, int nthreads, char* T_a, char* T_b, int* m, int* n, int* k,
  double* alpha, double* A, int* ldA, double* B, int* ldB, double* beta,
  double* C, int* ldC);

void compute_and_store (vector<thread> &threads, int dims[], int ndim, int iterations, int total_nthreads, double* imatrices[],
  double* interm_matrices[], ofstream ofiles[], double times[]);

void compute_OMP (int dims[], int ndim, int iterations, int total_nthreads,
  double** imatrices, double** interm_matrices, ofstream ofiles[], double times[]);

void parenth4 (int dims[], int ndim, double times[], int iterations, int nthreads,
  double** imatrices, double** interm_matrices, ofstream ofiles[]);

int main (int argc, char **argv){
  vector <thread> threads; // threads will be stored in this vector

  int ndim = 5, dims[ndim]; // In MC4 we have 5 dimensions
  double *imatrices[ndim];
  double *interm_matrices[2];
  int min_size, max_size, iterations, nsamples, total_nthreads, threshold;
  string output_file;
  // - imatrices: array that contains pointers to the input matrices. The result
  //              matrix is contained in the last slot.
  // - interm_matrices: array that contains pointers to the intermediate resulting
  //              matrices. In this case there are only 2 of them.
  // - min_size: the minimum size a certain dimension can take
  // - max_size: the maximum size a certain dimension can take
  // - iterations: total number of iteration each of the threads distributions will perform.
  //               This is done to minimise the system's noise.
  // - nsamples: number of points that will be evaluated.
  // - total_nthreads: number of total threads that will be used.
  // - threshold: at least one of the dimensions must be smaller than this number.
  // - output_file: filename used as a base for the different files created
  //                throughout the execution

  if (argc != 8){
    cout << "Execution: " << argv[0] << " min_size max_size iterations nsamples total_nthreads"
    " threshold output_file" << endl;
    return (-1);
  }
  else {
    min_size = atoi (argv[1]);
    max_size = atoi (argv[2]);
    iterations = atoi (argv[3]);
    nsamples = atoi (argv[4]);
    total_nthreads = atoi (argv[5]);
    threshold = atoi (argv[6]);
    output_file = argv[7];
  }

  double* times = (double*)malloc(iterations * sizeof(double));
  std::ofstream ofiles[total_nthreads];

  // ==================================================================
  //   - - - - - - - - - - Opening output files - - - - - - - - - - -
  // ==================================================================
  for (int i = 0; i < total_nthreads; i++){
    ofiles[i].open (output_file + to_string(i) + string(".csv"));
    if (ofiles[i].fail()){
      printf("Error opening the file %d\n", i);
      exit (-1);
    }
    add_headers (ofiles[i], ndim, iterations);
  }

  auto start = std::chrono::high_resolution_clock::now();

  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  mkl_set_dynamic(0);
  mkl_set_num_threads(total_nthreads);
  int alignment = 64;
  initialise_BLAS();

  for (int i = 0; i < nsamples; ){
    printf("{");
    for (int dim = 0; dim < ndim; dim++){
      dims[dim] = int(drand48() * (max_size - min_size)) + min_size;
      printf("%d,", dims[dim]);
    }
    printf("}\n");

    // ***************************************************************
    int dims []= {20000, 200, 1000, 50000, 10};
    printf("{");
    for (int dim = 0; dim < ndim; dim++){
      // dims[dim] = 1500;
      printf("%d,", dims[dim]);
    }
    printf("}\n");
    // ***************************************************************

    if (computation_decision (dims, ndim, threshold)){
      allocate_imatrices (imatrices, dims, ndim, alignment);

      allocate_interm_matrices (interm_matrices, dims, alignment);

      parenth4(dims, ndim, times, iterations, total_nthreads, imatrices, interm_matrices, ofiles);

      compute_OMP (dims, ndim, iterations, total_nthreads, imatrices, interm_matrices, ofiles, times);
      // compute_and_store (threads, dims, ndim, iterations, total_nthreads, imatrices, interm_matrices, ofiles, times);

      // void compute_and_store (vector<thread> &threads, int dims[], int ndim, int iterations,
      //   int total_nthreads, double** imatrices, double** interm_matrices,
      //   ofstream ofiles[], double times[])

      for (int j = 0; j < ndim; j++)
        free(imatrices[j]);

      for (int j = 0; j < 2; j++)
        free(interm_matrices[j]);

      i++;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  printf("The total execution has taken %f\n", std::chrono::duration<double>(end - start).count());

  for (int j = 0; j < total_nthreads; j++)
    ofiles[j].close();

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




void allocate_imatrices (double ** imatrices, int dims[], int ndim, int alignment){
  for (int i = 0; i < ndim - 1; i++){
    imatrices[i] = (double*)malloc(dims[i] * dims[i + 1] * sizeof(double));//, alignment);

    for (int j = 0; j < dims[i] * dims[i + 1]; j++)
      imatrices[i][j] = drand48 ();
  }

  imatrices[ndim - 1] = (double*)malloc(dims[0] * dims[ndim - 1] * sizeof(double));//, alignment);
  for (int j = 0; j < dims[0] * dims[ndim - 1]; j++)
    imatrices[ndim - 1][j] = drand48 ();
}




void allocate_interm_matrices (double ** interm_matrices, int dims[], int alignment){
  interm_matrices[0] = (double*)malloc(dims[0] * dims[2] * sizeof(double));//, alignment);
  for (int j = 0; j < dims[0] * dims[2]; j++)
    interm_matrices[0][j] = drand48 ();

  interm_matrices[1] = (double*)malloc(dims[2] * dims[4] * sizeof(double));//, alignment);
  for (int j = 0; j < dims[2] * dims[4]; j++)
    interm_matrices[1][j] = drand48 ();
}




void gemm (int id, int nthreads, char* T_a, char* T_b, int* m, int* n, int* k,
  double* alpha, double* A, int* ldA, double* B, int* ldB, double* beta,
  double* C, int* ldC){
    // auto time1 = std::chrono::high_resolution_clock::now();
    mkl_set_num_threads_local(nthreads);
    // auto time2 = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i < 1; i++)
    dgemm_ (T_a, T_b, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    // auto time3 = std::chrono::high_resolution_clock::now();
  }




void compute_and_store (vector<thread> &threads, int dims[], int ndim, int iterations,
  int total_nthreads, double** imatrices, double** interm_matrices,
  ofstream ofiles[], double times[]){
    double one = 1.0;
    char transpose = 'N';

    for (int nth_M1 = 1; nth_M1 < total_nthreads; nth_M1++){
      // printf("nth_M1: %d\n", nth_M1);
      int nth_M2 = total_nthreads - nth_M1;
      // printf("nth_M2: %d\n", nth_M2);
      for (int it = 0; it < iterations; it++){
        // printf("nthreads: %d\n", mkl_get_max_threads());
        cache_flush_par (total_nthreads);
        cache_flush_par (total_nthreads);
        cache_flush_par (total_nthreads);

        auto time1 = std::chrono::high_resolution_clock::now();
        threads.push_back (thread (gemm, 1, total_nthreads, &transpose, &transpose,
        &dims[0], &dims[2], &dims[1], &one, imatrices[0], &dims[0],
        imatrices[1], &dims[1], &one, interm_matrices[0], &dims[0]));

        threads.push_back (thread (gemm, 2, total_nthreads, &transpose, &transpose,
        &dims[2], &dims[4], &dims[3], &one, imatrices[2], &dims[2],
        imatrices[3], &dims[3], &one, interm_matrices[1], &dims[2]));

        for (auto &t: threads)
          t.join();

        // auto time2 = std::chrono::high_resolution_clock::now();
        mkl_set_num_threads_local (total_nthreads);
        dgemm_ (&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one,
        interm_matrices[0], &dims[0], interm_matrices[1], &dims[2], &one,
        imatrices[4], &dims[0]);

        auto time3 = std::chrono::high_resolution_clock::now();
        times[it] = std::chrono::duration<double>(time3 - time1).count();
        // getchar();
        // printf("GEMM3 distributed time: %f\n", std::chrono::duration<double>(time3 - time2).count());
        threads.clear();
      }
      add_line (ofiles[nth_M1], dims, ndim, times, iterations);
      // getchar();
    }

}

void compute_OMP (int dims[], int ndim, int iterations, int total_nthreads,
  double** imatrices, double** interm_matrices, ofstream ofiles[], double times[]){
    double one = 1.0;
    char transpose = 'N';

    for (int nth_M1 = 1; nth_M1 < total_nthreads; nth_M1++){
      int nth_M2 = total_nthreads - nth_M1;
      for (int it = 0; it < iterations; it++){
        cache_flush_par (total_nthreads);
        cache_flush_par (total_nthreads);
        cache_flush_par (total_nthreads);

        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(2)
        {
          #pragma omp sections
          {
            #pragma omp section
            {
              // printf("Section 1 - number omp threads: %d\n", omp_get_num_threads());
              mkl_set_num_threads_local(nth_M1);
              // printf("Section 1 - number mkl threads: %d\n", mkl_get_max_threads());
              // auto bgemm1 = std::chrono::high_resolution_clock::now();
              dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, imatrices[0],
                &dims[0], imatrices[1], &dims[1], &one, interm_matrices[0], &dims[0]);
              // auto agemm1 = std::chrono::high_resolution_clock::now();
              // printf(">> Section 1 - GEMM1 time with %d cores: %f\n", nth_M1, chrono::duration<double>(agemm1 - bgemm1).count());
            }
            #pragma omp section
            {
              // printf("Section 2 - number omp threads: %d\n", omp_get_num_threads());
              mkl_set_num_threads_local(nth_M2);
              // printf("Section 2 - number mkl threads: %d\n", mkl_get_max_threads());
              // auto bgemm2 = std::chrono::high_resolution_clock::now();
              dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, imatrices[2],
                &dims[2], imatrices[3], &dims[3], &one, interm_matrices[1], &dims[2]);
              // auto agemm2 = std::chrono::high_resolution_clock::now();
              // printf(">> Section 2 - GEMM2 time with %d cores: %f\n", nth_M2, chrono::duration<double>(agemm2 - bgemm2).count());
            }
          }
        }
        mkl_set_num_threads_local(total_nthreads);
        // auto bgemm3 = std::chrono::high_resolution_clock::now();
        dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, interm_matrices[0],
          &dims[0], interm_matrices[1], &dims[2], &one, imatrices[4], &dims[0]);
        // auto agemm3 = std::chrono::high_resolution_clock::now();
        // printf(">> GEMM3 time with %d cores: %f\n", total_nthreads, chrono::duration<double>(agemm3 - bgemm3).count());


        auto end = std::chrono::high_resolution_clock::now();
        // printf(">> Total time: %f\n", std::chrono::duration<double>(end - start).count());
        times[it] = std::chrono::duration<double>(end - start).count();
        // getchar();
      }
      add_line (ofiles[nth_M1], dims, ndim, times, iterations);
    }
  }






void parenth4 (int dims[], int ndim, double times[], int iterations, int nthreads,
  double** imatrices, double** interm_matrices, ofstream ofiles[]){
  double one = 1.0;
  char transpose = 'N';
  mkl_set_num_threads_local(nthreads);

  for (int it = 0; it < iterations; it++){
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);
    cache_flush_par (nthreads);

    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, imatrices[0],
      &dims[0], imatrices[1], &dims[1], &one, interm_matrices[0], &dims[0]);
    // auto time_gemm1 = std::chrono::high_resolution_clock::now();


    dgemm_(&transpose, &transpose, &dims[2], &dims[4], &dims[3], &one, imatrices[2],
      &dims[2], imatrices[3], &dims[3], &one, interm_matrices[1], &dims[2]);
    // auto time_gemm2 = std::chrono::high_resolution_clock::now();

    dgemm_(&transpose, &transpose, &dims[0], &dims[4], &dims[2], &one, interm_matrices[0],
      &dims[0], interm_matrices[1], &dims[2], &one, imatrices[4], &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();

    times[it] = std::chrono::duration<double>(time2-time1).count();

    // printf(">> GEMM1_time total_nthreads: %f\n", std::chrono::duration<double>(time_gemm1 - time1).count());
    // printf("\tGEMM2_time total_nthreads: %f\n", std::chrono::duration<double>(time_gemm2 - time_gemm1).count());
    // printf("\tGEMM3_time total_nthreads: %f\n", std::chrono::duration<double>(time2 - time_gemm2).count());
    // printf("\ttotal_nthreads time: %f\n", std::chrono::duration<double>(time2 - time1).count());
    // getchar();
  }
  add_line (ofiles[0], dims, ndim, times, iterations);
}











