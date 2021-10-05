#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <random>

#include "omp.h"
#include "mkl.h"

#include "common.h"

using namespace std;

static double *cs = NULL;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/**
 * Initialises the BLAS runtime by performing a GEMM operation with fixed
 * problem sizes (m=n=k=600). No memory alignment is performed. The used memory
 * is allocated and freed within the function.
 */
void initialise_BLAS(){
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
void initialise_mkl(){
  double *A, *B, *C;
  double one = 1.0;
  int alignment = 64;

  int m = 1200, n = 1200, k = 1200;
  char transpose = 'N';

  A = (double*)mkl_malloc(m * k * sizeof(double), alignment);
  B = (double*)mkl_malloc(k * n * sizeof(double), alignment);
  C = (double*)mkl_malloc(m * n * sizeof(double), alignment);

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
void initialise_mkl_variable(int size){
  double *A, *B, *C;
  double one = 1.0;
  int alignment = 64;

  int m = size, n = size, k = size;
  char transpose = 'N';

  A = (double*)mkl_malloc(m * k * sizeof(double), alignment);
  B = (double*)mkl_malloc(k * n * sizeof(double), alignment);
  C = (double*)mkl_malloc(m * n * sizeof(double), alignment);

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
 * Adds the headers to an output file. Format:
 * | ndim dims || nsamples samples |
 *
 * @param ofile     The output file manager, which has been previously opened.
 * @param ndim      The number of problem dimensions.
 * @param nsamples  The number of samples that will be computed.
 */
void add_headers (std::ofstream &ofile, int ndim, int nsamples){
  ofile << "# ";
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << "Sample_" << i;

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

/**
 * Adds the headers to an output file in validation phase. Format:
 * | ndim dims || parenth || nsamples samples |
 *
 * @param ofile     The output file manager, which has been previously opened.
 * @param ndim      The number of problem dimensions.
 * @param nsamples  The number of samples that will be computed.
 */
void add_headers_val (std::ofstream &ofile, int ndim, int nsamples){
  ofile << "# ";
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  ofile << "parenth" << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << "Sample_" << i;

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

/**
 * Adds the headers to an output file for a generated cube. Format:
 * # nthreads
 * # d0:    min_size, max_size, npoints
 * # ...
 * # di-1:  min_size, max_size, npoints
 * # d0:    points
 * # ...
 * # di-1:  points
 * # | ndim dims || nsamples samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param ndim      The number of dimensions in the cube (usually 3).
 * @param min_size  Allocated memory with the min sizes for each dimension.
 * @param max_size  Allocated memory with the max sizes for each dimension.
 * @param npoints   Allocated memory with the number of points for each
 *      dimension (might be different depending on the dimension).
 * @param nsamples  The number of samples to store in the output file.
 * @param nthreads  The number of threads to use in the computation - is stored.
 * @param points    Allocated memory with the points for each dimension (might
 *      be different depending on the dimension).
 */
void add_headers_cube (std::ofstream &ofile, int ndim, int *min_size, int *max_size,
  int *npoints, int nsamples, int nthreads, int **points){

  ofile << "# " << nthreads << std::endl;

  for (int i = 0; i < ndim; i++){
    ofile << "# " << min_size[i] << "," << max_size[i] << "," << npoints[i] << std::endl;
    // ofile << "# " << max_size[i] << endl;
    // ofile << "# " << npoints[i] << endl;
    ofile << "# ";

    for (int ii = 0; ii < npoints[i]; ii++){
      ofile << points[i][ii];
      if (ii == npoints[i] - 1)
        ofile << '\n';
      else
        ofile << ',';
    }
  }

  add_headers (ofile, ndim, nsamples);
}

/**
 * Adds the headers to an output file for anomalies. Format:
 * | dims || parenth_i || parenth_j || flops_score || times_score |
 *
 * @param ofile         Output file manager already opened.
 * @param ndim          The number of dimensions in the problem.
 */
void add_headers_anomalies (std::ofstream &ofile, int ndim) {
  ofile << "# ";
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  ofile << "n_threads,parenth_i,parenth_j,flops_score,time_score" << std::endl;
}

/**
 * Prints the headers to an output validation file for anomalies. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score || old_time_score |
 *
 * @param ofile         Output file manager already opened.
 * @param ndim          The number of dimensions in the problem.
 */
void print_header_validation (std::ofstream &ofile, int ndim) {
  ofile << "# ";
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  ofile << "n_threads,parenth_i,parenth_j,flops_score,time_score,old_time_score" << std::endl;
}

/**
 * Adds a line in the already opened output file. Format:
 * | dims || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      Allocated memory containing the dimensions.
 * @param ndim      The number of dimensions in the problem.
 * @param times     The array that contains the execution times (already computed).
 * @param nsamples  The number of samples to store in the output file.
 */
void add_line (std::ofstream &ofile, int *dims, int ndim, double *times, int nsamples){
  for (int i = 0; i < ndim; i++)
    ofile << dims[i] << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << std::fixed << std::setprecision(10) << times[i];

    if (i == nsamples - 1)
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
 * @param dims      The array with the problem dimensions.
 * @param ndim      The number of dimensions in the problem.
 * @param times     The array that contains the execution times (already computed).
 * @param nsamples  The number of samples to store in the output file.
 */
void add_line_ (std::ofstream &ofile, int dims[], int ndim, double times[], int nsamples){
  for (int i = 0; i < ndim; i++)
    ofile << dims[i] << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << std::fixed << std::setprecision(10) << times[i];

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

/**
 * Adds a line in the already opened ofile for validation phase. Format:
 * | dims || nparenth || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      The array with the problem dimensions.
 * @param ndim      The number of dimensions in the problem.
 * @param times     The array that contains the execution times (already computed).
 * @param nsamples  The number of samples to store in the output file.
 * @param parenth   Value that indicates which parenthesisation is stored.
 */
void add_line_val (std::ofstream &ofile, int dims[], int ndim, double times[], int nsamples, int parenth){
  for (int i = 0; i < ndim; i++)
    ofile << dims[i] << ',';

  ofile << parenth << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << std::fixed << std::setprecision(10) << times[i];

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

/**
 * Adds a line to an output file for anomalies. Format:
 * | dims || parenth_i || parenth_j || flops_score || times_score |
 *
 * @param ofile         Output file manager already opened.
 * @param dims          The array containing the dimensions.
 * @param ndim          The number of dimensions in the problem.
 * @param parenth_i     First parenthesisation involved in the anomaly.
 * @param parenth_j     Second parenthesisation involved in the anomaly.
 * @param flops_score   The flops_score value.
 * @param time_score    The time_score value.
 */
 void add_line_anomalies (std::ofstream &ofile, int dims[], int ndim, int n_threads,
   int parenth_i, int parenth_j, double flops_score, double time_score) {

  for (int i = 0; i < ndim; i++)
    ofile << dims[i] << ',';

  ofile << n_threads << ',' << parenth_i << ',' << parenth_j << ',' << flops_score << ',' <<
    time_score << std::endl;
}

/**
 * Adds a line to an output file for anomalies. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score |
 *
 * @param ofile Output file manager already opened.
 * @param an    The anomaly to be printed in the file.
 */
void add_line_anomalies (std::ofstream &ofile, anomaly an){
  for (unsigned i = 0; i < an.dims.size(); i++)
    ofile << an.dims[i] << ',';

  ofile << an.n_threads << ',' << an.parenths[0] << ',' << an.parenths[1] << ',' << an.flops_score <<
    ',' << an.time_score << std::endl;
}

void add_line_anomalies (std::ostream &ofile, anomaly an){
  for (unsigned i = 0; i < an.dims.size(); i++)
    ofile << an.dims[i] << ',';

  ofile << an.n_threads << ',' << an.parenths[0] << ',' << an.parenths[1] << ',' << an.flops_score <<
    ',' << an.time_score << std::endl;
}

void print_anomaly (std::ofstream &ofile, anomaly an) {
  for (auto &d : an.dims)
    ofile << d << ',';

  ofile << an.n_threads << ',';

  for (auto &p : an.parenths)
    ofile << p << ',';

  ofile << an.flops_score << ',' << an.time_score << std::endl;
}

/**
 * Prints a line in the output anomaly validation file. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score || old_time_score |
 *
 * @param ofile           Output file manager already opened.
 * @param an              The anomaly to be printed in the file.
 * @param old_time_score  The old time score validated.
 */
void print_validation (std::ofstream &ofile, anomaly an, float old_time_score){
  for (auto &d : an.dims)
    ofile << d << ',';

  ofile << an.n_threads << ',';

  for (auto &p : an.parenths)
    ofile << p << ',';

  ofile << an.flops_score << ',' << an.time_score << ',' << old_time_score << std::endl;
}

/**
 * Flushes the cache memory for a single thread.
 */
void cache_flush(){
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
 * @param nthreads  The number of threads for which the cache is flushed.
 */
void cache_flush_par(int nthreads){
	if (cs == NULL){
		cs = (double*)malloc(GEMM_L3_CACHE_SIZE * sizeof(double));
		for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++)
			cs[i] = drand48();
	}
  omp_set_num_threads (nthreads);
  #pragma omp parallel shared(cs)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++){
      // printf("Thread %d - iteration %d\n", omp_get_thread_num(), i);
  		cs[i] += 1e-3;
    }
  }

}

/**
 * Computes an array's minimum value.
 *
 * @param arr   The array of which the minimum value is computed.
 * @param size  The size of the array.
 * @return      The array's minimum value.
 */
double min_array (double arr[], int size){
  double x = numeric_limits<double>::max();

  for (int i = 0; i < size; i++){
    if (arr[i] < x)
      x = arr[i];
  }

  return x;
}

/**
 * Computes an array's mean value.
 *
 * @param arr   The array of which the mean value is computed.
 * @param size  The size of the array.
 * @return      The array's mean value.
 */
double mean_array (double arr[], int size){
  double x = 0.0;

  for (int i = 0; i < size; i++){
      x += arr[i];
  }

  return x/size;
}

/**
 * Finds an array's median value.
 *
 * @param arr   The array of which the median value is searched.
 * @param n     The size of the array.
 * @return      The array's median value.
 */
double median_array (double arr[], int size){
  sort (arr, arr + size);

  if (size % 2 != 0){
    return arr[size];
  } else
    return double ((arr[(size - 1) / 2] + arr[size / 2]) / 2.0);
}

// double median_vector (std::vector<double> v) {
//   int n = v.size();
//   if (n % 2 != 0){
//     std::nth_element (v.begin(), v.begin() + n / 2, v.end());
//     return static_cast<double> (v[n / 2]);
//   }
//   else {
//     std::nth_element (v.begin(), v.begin() + n / 2, v.end());
//     std::nth_element (v.begin(), v.begin() + (n - 1) / 2, v.end());
//     return static_cast<double> ((v[(n - 1) / 2] + v[n / 2]) / 2.0);
//   }
// }

// template <class T>
// T avg_vector (std::vector<T> v) {
//   return v[0];
// }

/**
 * Computes the defined "score" between two values. Score: abs(x - y) / max(x, y).
 *
 * @param x   The first value.
 * @param y   The second value.
 * @return    The "score" between both values.
 */
// double score (double x, double y){
//   return abs(x - y) / max(x, y);
// }

/**
 * Checks whether there is an anomaly between one pair of parenthesisations.
 *
 * @param flops_a     The #FLOPs for the first parenthesisation.
 * @param flops_b     The #FLOPs for the second parenthesisation.
 * @param times_a     The execution times for the first parenthesisation.
 * @param times_b     The execution times for the second parenthesisation.
 * @param iterations  The number of times each parenthesisation is computed.
 * @param ratio       The percentage of total executions for which the
 * parenthesisation with more FLOPs must be faster.
 * @return            Whether there is an anomaly for this pair.
 */
bool is_anomaly (unsigned long long int flops_a, unsigned long long int flops_b,
  double times_a[], double times_b[], int iterations, double ratio){
  int n_anomalies = 0;

  if (flops_a < flops_b){
    for (int i = 0; i < iterations; i++){
      if (times_b[i] < times_a[i])
        n_anomalies++;
    }
  } else if (flops_b < flops_a){
    for (int i = 0; i < iterations; i++){
      if (times_a[i] < times_b[i])
        n_anomalies++;
    }
  }
  return n_anomalies >= int(iterations * ratio);
}

bool is_anomaly (unsigned long long int flops_a, unsigned long long int flops_b,
  std::vector<double> times_a, std::vector<double> times_b, double ratio){
  int n_anomalies = 0;

  if (flops_a < flops_b) {
    for (unsigned i = 0; i < times_a.size(); i++){
      if (times_b[i] < times_a[i])
        n_anomalies++;
    }
  } else if (flops_b < flops_a){
    for (unsigned i = 0; i < times_b.size(); i++){
      if (times_a[i] < times_b[i])
        n_anomalies++;
    }
  }
  return n_anomalies >= int(times_a.size() * ratio);
}