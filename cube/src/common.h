#ifndef COMMON_FUNC
#define COMMON_FUNC

#include <cmath>
#include <algorithm>
#include <vector>

struct anomaly {
  std::vector<int> dims;
  std::vector<int> parenths;
  int n_threads;
  float flops_score, time_score;
  bool isAnomaly;
};

/**
 * Initialises the BLAS runtime by performing a GEMM operation with fixed
 * problem sizes (m=n=k=600). No memory alignment is performed. The used memory
 * is allocated and freed within the function.
 */
void initialise_BLAS();

/**
 * Initialises the BLAS runtime by performing a GEMM operation with fixed
 * problem sizes (m=n=k=1200) with memory alignment to 64bytes. The used memory
 * is allocated and freed within the function.
 */
void initialise_mkl();

/**
 * Initialises the BLAS runtime by performing a GEMM operation with variable
 * problem sizes (m=n=k=size) with memory alignment to 64bytes. The used memory
 * is allocated and freed within the function.
 *
 * @param size  The problem sizes - recommended to be great enough if only
 * performed once. Caution is advised if this function is used within a loop.
 */
void initialise_mkl_variable(int size);

/**
 * Adds the headers to an output file. Format:
 * | ndim dims || nsamples samples |
 *
 * @param ofile     The output file manager, which has been previously opened.
 * @param ndim      The number of problem dimensions.
 * @param nsamples  The number of samples that will be computed.
 */
void add_headers (std::ofstream & ofile, int ndim, int nsamples);

/**
 * Adds the headers to an output file in validation phase. Format:
 * | ndim dims || parenth || nsamples samples |
 *
 * @param ofile     The output file manager, which has been previously opened.
 * @param ndim      The number of problem dimensions.
 * @param nsamples  The number of samples that will be computed.
 */
void add_headers_val (std::ofstream &ofile, int ndim, int nsamples);

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
  int *npoints, int nsamples, int nthreads, int **points);

/**
 * Adds the headers to an output file for anomalies. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score |
 *
 * @param ofile         Output file manager already opened.
 * @param ndim          The number of dimensions in the problem.
 */
void add_headers_anomalies (std::ofstream &ofile, int ndim);

/**
 * Prints the headers to an output validation file for anomalies. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score || old_time_score |
 *
 * @param ofile         Output file manager already opened.
 * @param ndim          The number of dimensions in the problem.
 */
void print_header_validation (std::ofstream &ofile, int ndim);

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
void add_line (std::ofstream &ofile, int *dims, int ndim, double *times, int nsamples);

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
void add_line_ (std::ofstream &ofile, int dims[], int ndim, double times[], int nsamples);

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
void add_line_val (std::ofstream &ofile, int dims[], int ndim, double times[], int nsamples, int parenth);

/**
 * Adds a line to an output file for anomalies. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score |
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
  int parenth_i, int parenth_j, double flops_score, double time_score);

/**
 * Adds a line to an output file for anomalies. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score |
 *
 * @param ofile Output file manager already opened.
 * @param an    The anomaly to be printed in the file.
 */
void add_line_anomalies (std::ofstream &ofile, anomaly an);
void add_line_anomalies (std::ostream &ofile, anomaly an);
void print_anomaly (std::ofstream &ofile, anomaly an);

/**
 * Prints a line in the output anomaly validation file. Format:
 * | dims || n_threads || parenth_i || parenth_j || flops_score || times_score || old_time_score |
 *
 * @param ofile           Output file manager already opened.
 * @param an              The anomaly to be printed in the file.
 * @param new_time_score  The old time score validated.
 */
void print_validation (std::ofstream &ofile, anomaly an, float old_time_score);

/**
 * Flushes the cache memory for a single thread.
 */
void cache_flush();

/**
 * Function that flushes the cache memory for a certain number of threads.
 *
 * @param nthreads  The number of threads for which the cache is flushed.
 */
void cache_flush_par (int nthreads);

/**
 * Computes an array's minimum value.
 *
 * @param arr   The array of which the minimum value is computed.
 * @param size  The size of the array.
 * @return      The array's minimum value.
 */
double min_array (double arr[], int size);

/**
 * Computes an array's mean value.
 *
 * @param arr   The array of which the mean value is computed.
 * @param size  The size of the array.
 * @return      The array's mean value.
 */
double mean_array (double arr[], int size);

/**
 * Finds an array's median value.
 *
 * @param arr   The array of which the median value is searched.
 * @param size  The size of the array.
 * @return      The array's median value.
 */
double median_array (double arr[], int size);

template<typename T> T median_vector (const std::vector<T> &v){
  std::vector<T> aux = v;
  int n = aux.size()
  if (n % 2 != 0){
    std::nth_element (aux.begin(), aux.begin() + n / 2, aux.end());
    return aux[n / 2];
  }
  else {
    std::nth_element (aux.begin(), aux.begin() + n / 2, aux.end());
    std::nth_element (aux.begin(), aux.begin() + (n - 1) / 2, aux.end());
    return static_cast<T> ((aux[(n - 1) / 2] + aux[n / 2]) / 2.0);
  }
}

template<typename T> T avg_vector (const std::vector<T> &v){
  return v[0];
}

template<typename T> T abs_ (const T x) {
  return x>=0 ? x : -x;
}

/**
 * Computes the defined "score" between two values. Score: abs(x - y) / max(x, y).
 *
 * @param x   The first value.
 * @param y   The second value.
 * @return    The "score" between both values.
 */
template<typename T> double score (const T x, const T y) {
  return abs_<T>(x - y) / std::max(x, y);
}

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
  double times_a[], double times_b[], int iterations, double ratio);

bool is_anomaly (unsigned long long int flops_a, unsigned long long int flops_b,
  std::vector<double> times_a, std::vector<double> times_b, double ratio);

#endif