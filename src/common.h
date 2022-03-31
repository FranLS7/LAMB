#ifndef COMMON_FUNC
#define COMMON_FUNC

#include <algorithm>
#include <iostream> // @TODO ERASE THIS 
#include <string>
#include <vector>

using iVector1D = std::vector<int>;
using iVector2D = std::vector<iVector1D>;

using dVector1D = std::vector<double>;
using dVector2D = std::vector<dVector1D>;
using dVector3D = std::vector<dVector2D>;

struct Matrix{
  std::string name;
  int rows, columns;
  double* data = nullptr;
};

struct DataPoint{
  iVector1D dims;
  std::vector<unsigned long> flops;
  dVector1D samples;
};

namespace lamb {

/**
 * Generates all the input matrices given a set of dimensions that idenfity a certain 
 * Matrix Chain problem.
 * 
 * @param dims    Vector containing the dimensions of the input matrices.
 * @return        Vector containing pointers to each of the matrices.
 */
std::vector<double*> generateMatrices (std::vector<int> dims);

/**
 * Frees all the input matrices previously allocated with generateMatrices.
 * 
 * @param matrices  Vector containing pointers to each of the matrices.
 */
void freeMatrices (std::vector<double*>& matrices);

/**
 * Initialises the BLAS runtime by performing a GEMM operation with fixed
 * problem sizes (m=n=k=600). No memory alignment is performed. The used memory
 * is allocated and freed within the function.
 */
void initialiseBLAS();

/**
 * Initialises the BLAS runtime by performing a GEMM operation with fixed
 * problem sizes (m=n=k=1200) with memory alignment to 64bytes. The used memory
 * is allocated and freed within the function.
 */
void initialiseMKL();

/**
 * Initialises the BLAS runtime by performing a GEMM operation with variable
 * problem sizes (m=n=k=size) with memory alignment to 64bytes. The used memory
 * is allocated and freed within the function.
 *
 * @param size  The problem sizes - recommended to be great enough if only
 * performed once. Caution is advised if this function is used within a loop.
 */
void initialiseMKL(int size);

/**
 * Flushes the cache memory for a single thread.
 */
void cacheFlush();

/**
 * Function that flushes the cache memory for a certain number of threads.
 *
 * @param n_threads  The number of threads for which the cache is flushed.
 */
void cacheFlush (const int n_threads);

/**
 * Computes an array's minimum value.
 *
 * @tparam T    The type to work with.
 * @param arr   The array of which the minimum value is computed (of type T).
 * @param size  The size of the array.
 * @return      The array's minimum value (of type T).
 */
template <typename T> 
T minArray (const T arr[], const int size) {
  T x = arr[0];

  for (int i = 0; i < size; i++) {
    if (arr[i] < x)
      x = arr[i];
  }
  return x;
}

/**
 * Computes an array's mean value.
 *
 * @tparam T    The type to work with.
 * @param arr   The array of which the mean value is computed (of type T).
 * @param size  The size of the array.
 * @return      The array's mean value (of type T).
 */
template <typename T> 
T meanArray (const T arr[], const int size) {
  T x;

  for (int i = 0; i < size; i++)
    x += arr[i];

  return x/size;
}

/**
 * Finds an array's median value.
 *
 * @tparam T    The type to work with.
 * @param arr   The array of which the median value is searched (of type T).
 * @param size  The size of the array.
 * @return      The array's median value (of type T).
 */
// double median_array (double arr[], int size);
template <typename T> 
T medianArray (T arr[], const int size) {
  sort (arr, arr + size);

  if (size % 2 != 0) 
    return arr[size];
  else
    return static_cast<T>((arr[(size - 1) / 2] + arr[size / 2]) / 2.0);
}

// @TODO -- document this function.
template<typename T> 
unsigned idxMinVector (const std::vector<T>& v){
  T min = v[0];
  unsigned idx = 0;
  for (unsigned i = 0; i < v.size(); i++) {
    if (v[i] < min){
      min = v[i];
      idx = i;
    }
  }
  return idx;
}

/**
 * Computes the average value of a vector.
 * 
 * @tparam T  The type to work with.
 * @param v   The input vector (of type T) of which the average is computed.
 * @return    The vector's average value (of type T).
 */
template<typename T> 
T avgVector (const std::vector<T>& v){
  return v[0];
}

/**
 * Computes the absolute value of a certain input.
 * 
 * @tparam T  The type to work with.
 * @param x   The variable of type T of which the absolute value is computed.
 * @return    The variable's absolute value (of type T).
 */
template<typename T> 
T abs_ (const T x) {
  return x>=0 ? x : -x;
}

/**
 * Computes the defined "score" between two values. Score: abs(x - y) / max(x, y).
 *
 * @tparam T    The type to work with.
 * @param x,y   The values of which the score is computed (of type T).
 * @return      The score between both values (a double).
 */
template<typename T> 
double score (const T x, const T y) {
  return (std::abs(double(x) - double(y)) / double(std::max(x, y)));
}

template<typename T> 
T medianVector(std::vector<T>& v) {
  std::vector<T> aux = v;
  int n = aux.size();
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

/**
 * Appends the elements in a vector to an output string. This is used as a key for hash tables
 * where there is no other way to identify a certain vector (point in a hyperspace - given by 
 * determined dimensions).
 * 
 * @tparam T  Type to work with
 * @param v   The input vector to serialise - a vector of type T
 * @return    A string containing the serialised vector.
 */
template<typename T> 
std::string serialiseVector (std::vector<T> v) {
  std:: string buffer;

  for (auto const &x : v) 
    buffer.append (std::to_string(x) + "/");
  
  return buffer; 
}

/**
 * Adds the headers to an output file with execution times. Format:
 * | ndim dims || nsamples samples |
 *
 * @param ofile       The output file manager, which has been previously opened.
 * @param ndim        The number of problem dimensions.
 * @param iterations  The number of samples that will be computed.
 */
void printHeaderTime (std::ofstream& ofile, const int ndim, const int iterations, 
    const bool flops=false);

/**
 * @brief Prints headers to a file with results.
 * 
 * This function overload is used to print the headers of a result file with times
 * for each operation. Format:
 * || ndim dims || sample0_op0 | sample0_op1 | sample0_op2 | sample0_total | ...| ||
 * 
 * @param ofile         Output file manager, already opened.
 * @param ndim          Number of problem dimensions.
 * @param iterations    Number of samples that will be computed.
 * @param n_operations  Number of operations performed in each iteration.
 */
void printHeaderTime (std::ofstream& ofile, const int ndim, const int iterations, 
    const int n_operations);

/**
 * @brief Prints headers to a file with results.
 * 
 * This function overload is used to print the headers of a result file with times and flops
 * for each operation. Format:
 * || ndim dims || flops_op0 | ... | flops_total || sample0_op0 | ... | sample0_total | ...| ||
 * 
 * @param ofile         Output file manager, already opened.
 * @param ndim          Number of problem dimensions.
 * @param iterations    Number of samples that will be computed.
 * @param n_operations  Number of operations performed in each iteration.
 */
void printHeaderTime (std::ofstream &ofile, const int ndim, const int iterations, 
    const int n_operations, const bool flops);

/**
 * Adds a line in the already opened output file. Format:
 * | dims || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      Vector containing the dimensions.
 * @param times     The vector that contains the execution times (already computed).
 */
void printTime (std::ofstream &ofile, const iVector1D &dims, const dVector1D &times);

/**
 * Adds a line in the already opened output file. Format:
 * | dims || flops || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      Vector containing the dimensions.
 * @param times     The vector that contains the execution times (already computed).
 * @param flops     Number of flops
 */
void printTime (std::ofstream &ofile, const iVector1D &dims, const dVector1D &times, 
    unsigned long flops);

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
    const std::vector<unsigned long> flops);

} // namespace lamb

#endif