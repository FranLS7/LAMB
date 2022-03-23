#include "common.h"

#include <stdlib.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "omp.h"
#include "mkl.h"

const int ALIGN = 64;

static double *cs = NULL;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

namespace lamb {
/**
 * Generates all the input matrices given a set of dimensions that idenfity a certain 
 * Matrix Chain problem.
 * 
 * @param dims    Vector containing the dimensions of the input matrices.
 * @return        Vector containing pointers to each of the matrices.
 */
std::vector<double*> generateMatrices (std::vector<int> dims){
  std::vector<double*> matrices;
  for (unsigned i = 0; i < (dims.size() - 1); i++) {
    double *matrix = (double*)mkl_malloc (dims[i] * dims[i + 1] * sizeof(double), ALIGN);

    for (int j = 0; j < dims[i] * dims[i + 1]; j++)
      matrix[j] = drand48();

    matrices.push_back (matrix);
  }
  return matrices;
}

/**
 * Frees all the input matrices previously allocated with generateMatrices.
 * 
 * @param matrices  Vector containing pointers to each of the matrices.
 */
void freeMatrices (std::vector<double*>& matrices){
  for (auto &mat : matrices)
    mkl_free(mat);
}

/**
 * Initialises the BLAS runtime by performing a GEMM operation with fixed
 * problem sizes (m=n=k=600). No memory alignment is performed. The used memory
 * is allocated and freed within the function.
 */
void initialiseBLAS() {
  double *A, *B, *C;
  double one = 1.0;

  int m = 600, n = 600, k = 600;
  char transpose = 'N';

  A = (double*)malloc(m * k * sizeof(double));
  B = (double*)malloc(k * n * sizeof(double));
  C = (double*)malloc(m * n * sizeof(double));

  for (int i = 0; i < m * k; i++)
    A[i] = drand48();

  for (int i = 0; i < k * n; i++)
    B[i] = drand48();

  for (int i = 0; i < m * n; i++)
    C[i] = drand48();

  dgemm_(&transpose, &transpose, &m, &n, &k, &one, A, &m, B, &k, &one, C, &m);

  free(A);
  free(B);
  free(C);
}

/**
 * Initialises the BLAS runtime by performing a GEMM operation with fixed
 * problem sizes (m=n=k=1200) with memory alignment to 64bytes. The used memory
 * is allocated and freed within the function.
 */
void initialiseMKL() {
  double *A, *B, *C;
  double one = 1.0;
  char transpose = 'N';

  int m = 1200, n = 1200, k = 1200;

  A = (double*)mkl_malloc(m * k * sizeof(double), ALIGN);
  B = (double*)mkl_malloc(k * n * sizeof(double), ALIGN);
  C = (double*)mkl_malloc(m * n * sizeof(double), ALIGN);

  for (int i = 0; i < m * k; i++)
    A[i] = drand48();

  for (int i = 0; i < k * n; i++)
    B[i] = drand48();

  for (int i = 0; i < m * n; i++)
    C[i] = drand48();

  dgemm_(&transpose, &transpose, &m, &n, &k, &one, A, &m, B, &k, &one, C, &m);

  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
}

/**
 * Initialises the BLAS runtime by performing a GEMM operation with variable
 * problem sizes (m=n=k=size) with memory alignment to 64bytes. The used memory
 * is allocated and freed within the function.
 *
 * @param size  The problem sizes - recommended to be great enough if only
 * performed once. Caution is advised if this function is used within a loop.
 */
void initialiseMKL(int size) {
  double *A, *B, *C;
  double one = 1.0;
  char transpose = 'N';

  int m = size, n = size, k = size;

  A = (double*)mkl_malloc(m * k * sizeof(double), ALIGN);
  B = (double*)mkl_malloc(k * n * sizeof(double), ALIGN);
  C = (double*)mkl_malloc(m * n * sizeof(double), ALIGN);

  for (int i = 0; i < m * k; i++)
    A[i] = drand48();

  for (int i = 0; i < k * n; i++)
    B[i] = drand48();

  for (int i = 0; i < m * n; i++)
    C[i] = drand48();

  dgemm_(&transpose, &transpose, &m, &n, &k, &one, A, &m, B, &k, &one, C, &m);

  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
}

/**
 * Flushes the cache memory for a single thread.
 */
void cacheFlush(){
	if (cs == NULL){
		cs = (double*)malloc(GEMM_L3_CACHE_SIZE * sizeof(double));
		for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++)
			cs[i] = drand48();
	}
	for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++)
		cs[i] += 1e-3;
}

/**
 * Function that flushes the cache memory for a certain number of threads.
 *
 * @param n_threads  The number of threads for which the cache is flushed.
 */
void cacheFlush(int n_threads) {
	if (cs == NULL){
		cs = (double*)malloc(GEMM_L3_CACHE_SIZE * sizeof(double));
		for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++)
			cs[i] = drand48();
	}
  omp_set_num_threads (n_threads);
  #pragma omp parallel shared(cs)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++){
  		cs[i] += 1e-3;
    }
  }
}


/**
 * Adds the headers to an output file with execution times. Format:
 * | ndim dims || nsamples samples |
 *
 * @param ofile     The output file manager, which has been previously opened.
 * @param ndim      The number of problem dimensions.
 * @param nsamples  The number of samples that will be computed.
 */
void printHeaderTime (std::ofstream& ofile, const int ndim, const int iterations, 
    const bool flops) {
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';
  
  if (flops)
    ofile << "flops,";

  for (int i = 0; i < iterations; i++){
    ofile << "sample_" << i;

    if (i == iterations - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

void printHeaderTime (std::ofstream &ofile, const int ndim, const int iterations, 
    const int n_operations) {
  for (int i = 0; i < ndim; ++i)
    ofile << 'd' << i << ',';
  
  for (int i = 0; i < iterations; ++i){
    for (int j = 0; j < n_operations; ++j)
      ofile << "sample" << i << "_op" << j << ',';
    
    ofile << "sample" << i << "_total";

    if (i == iterations - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

void printHeaderTime (std::ofstream &ofile, const int ndim, const int iterations, 
    const int n_operations, const bool flops) {
  for (int i = 0; i < ndim; ++i)
    ofile << 'd' << i << ',';
  
  for (int i = 0; i < n_operations; ++i)
    ofile << "flops_op" << i << ',';
  ofile << "flops_total,";

  for (int i = 0; i < iterations; ++i){
    for (int j = 0; j < n_operations; ++j)
      ofile << "sample" << i << "_op" << j << ',';
    
    ofile << "sample" << i << "_total";

    if (i == iterations - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

/**
 * Adds a line in the already opened output file. Format:
 * | dims || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      Vector containing the dimensions.
 * @param times     The vector that contains the execution times (already computed).
 */
void printTime (std::ofstream &ofile, const iVector1D &dims, const dVector1D &times) {
  for (const auto &d : dims)
    ofile << d << ',';

  for (unsigned i = 0; i < times.size(); ++i) {
    ofile << std::fixed << std::setprecision(10) << times[i];
    if (i == times.size() - 1)
      ofile << '\n';
    else 
      ofile << ',';
  }
}

/**
 * Adds a line in the already opened output file. Format:
 * | dims || flops || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      Vector containing the dimensions.
 * @param times     The vector that contains the execution times (already computed).
 * @param flops     Vector that contains the number of flops
 */
void printTime (std::ofstream &ofile, const iVector1D &dims, const dVector1D &times, 
    unsigned long flops) {
  for (const auto &d : dims)
    ofile << d << ',';

  ofile << flops << ',';

  for (unsigned i = 0; i < times.size(); ++i) {
    ofile << std::fixed << std::setprecision(10) << times[i];
    if (i == times.size() - 1)
      ofile << '\n';
    else 
      ofile << ',';
  }
}

void printTime (std::ofstream &ofile, const iVector1D &dims, const dVector1D &times, 
    const std::vector<unsigned long> flops) {
  for (const auto &d : dims)
    ofile << d << ',';

  for (const auto& f : flops)
    ofile << f << ',';

  for (unsigned i = 0; i < times.size(); ++i) {
    ofile << std::fixed << std::setprecision(10) << times[i];
    if (i == times.size() - 1)
      ofile << '\n';
    else 
      ofile << ',';
  }
}

} // namespace lamb