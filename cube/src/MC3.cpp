#include "MC3.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "mkl.h"

#include "common.h"

const int ALIGN = 64;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

namespace mc3 {
/**
 * Computes the execution times for certain parenthesisations and returns those
 * inside a vector of vectors (these vectors contain the different execution times, as many
 * as the number of iterations specifies).
 * 
 * @param dims          Vector containing the dimensions of the problem.
 * @param parenths      Vector containing the parenthesisations to execute.
 * @param iterations    Number of times to execute each parenthesisation.
 * @param n_threads     Number of threads to use in the computation.
 * @return              Vector with all the execution times.
 */
std::vector<std::vector<double>> MC3_execute_par (std::vector<int> dims, std::vector<int> parenths,
    const int iterations, const int n_threads) {

  std::vector<std::vector<double>> all_times;
  std::vector<double*> input_matrices = lamb::generateMatrices(dims);

  // Execute all parenthesisations.
  for (auto const p : parenths)
    all_times.push_back (MC3_execute(dims, input_matrices, p, iterations, n_threads));
    
  lamb::freeMatrices(input_matrices);  
  return all_times;
}

/**
 * Executes a certain parenthesisation of the Matrix Chain of length 3 problem. Takes as input 
 * the dimensions of the problem, the input matrices already allocated and initialised, the
 * parenthesisation to compute, the number of iterations to measure the time, and the number
 * of threads to use. Returns the execution times in the form of a vector.
 * 
 * @param dims            Vector containing the dimensions of the problem.
 * @param input_matrices  Vector containing the input matrices already allocated and initialised.
 * @param parenth         Indicates which parenthesisation to evaluate.
 * @param iterations      Number of iterations to execute the problem.
 * @param n_threads       Number of threads to use during computation.
 * @return                Execution times in the form of a vector.
 */
std::vector<double> MC3_execute (std::vector<int> dims, std::vector<double*> input_matrices,
    const int parenth, const int iterations, const int n_threads) {

  switch (parenth) {
    case 0  : return MC3_p0(dims, input_matrices, iterations, n_threads);
    case 1  : return MC3_p1(dims, input_matrices, iterations, n_threads);
    default : return std::vector<double>(); // empty vector
  }
}

/**
 * Each of these functions computes a certain MC3 parenthesisation and returns
 * execution times. These functions assume the initial matrices have been allocated 
 * and initialised.
 *
 * @param dims            The vector with the matrices dimensions.
 * @param input_matrices  Vector containing pointers to memory allocated for matrices.
 * @param iterations      The number of iterations to compute.
 * @param nthreads        The number of threads to used during computation.
 * @return                The execution times in the form of a vector.
 */
std::vector<double> MC3_p0 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads) {
  
  std::vector<double> times;    
  double *M1, *X;
  double one = 1.0f;
  char transpose = 'N';

  M1 = (double*)mkl_malloc(dims[0] * dims[2] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = 0.0f;

  for (int i = 0; i < dims[0] * dims[3]; i++)
    X[i] = 0.0f;

  for (int it = 0; it < iterations; it++){
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, input_matrices[0],
      &dims[0], input_matrices[1], &dims[1], &one, M1, &dims[0]);

    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[2], &one, M1,
      &dims[0], input_matrices[2], &dims[2], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(X);

  return times;
}

/**
 * Each of these functions computes a certain MC3 parenthesisation and returns
 * execution times. These functions assume the initial matrices have been allocated 
 * and initialised.
 *
 * @param dims            The vector with the matrices dimensions.
 * @param input_matrices  Vector containing pointers to memory allocated for matrices.
 * @param iterations      The number of iterations to compute.
 * @param nthreads        The number of threads to used during computation.
 * @return                The execution times in the form of a vector.
 */
std::vector<double> MC3_p1 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads){

  std::vector<double> times;    
  double *M1, *X;
  double one = 1.0f;
  char transpose = 'N';

  M1 = (double*)mkl_malloc(dims[1] * dims[3] * sizeof(double), ALIGN);
  X  = (double*)mkl_malloc(dims[0] * dims[3] * sizeof(double), ALIGN);

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = 0.0f;

  for (int i = 0; i < dims[0] * dims[3]; i++)
    X[i] = 0.0f;

  for (int it = 0; it < iterations; it++){
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);

    auto start = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, input_matrices[1],
      &dims[1], input_matrices[2], &dims[2], &one, M1, &dims[1]);

    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[1], &one, input_matrices[0],
      &dims[0], M1, &dims[1], &one, X, &dims[0]);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  mkl_free(M1);
  mkl_free(X);

  return times;
}

/**
 * Computes the amount of FLOPs for one parenthesisation.
 *
 * @param dims    The vector with the matrices dimensions.
 * @param parenth The parenthesisation for which #FLOPs is computed.
 * @return        The number of FLOPs for a given parenthesisation.
 */
unsigned long long int MC3_flops (std::vector<int> dims, int parenth) {
  switch (parenth) {
    case 0  : return MC3_flops_1(dims);
    case 1  : return MC3_flops_0(dims);
    default : std::cout << ">> ERROR: wrong parenthesisation for MC_flops." << std::endl;
              return -1;
  }
}

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The vector with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions and parenthesisation.
 */
unsigned long long int MC3_flops_0 (std::vector<int> dims) {
  unsigned long long int result;
  result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
  result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  return result * 2;
}

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The vector with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions and parenthesisation.
 */
unsigned long long int MC3_flops_1 (std::vector<int> dims) {
  unsigned long long int result;
  result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
  result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[3]);
  return result * 2;
}

} // namespace mc3