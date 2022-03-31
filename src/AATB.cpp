#include "AATB.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "mkl.h"

#include "common.h"

const int ALIGN = 64;

namespace aatb
{

  /**
   * @brief Construct a new AATB object. Default constructor.
   */
  AATB::AATB() {}

  /**
   * @brief Additional constructor with a set of dimensions.
   *
   * This constructor uses the input set of dimensions to create empty input and intermediate
   * matrices that are used to compute the corresponding cost in terms of FLOPs.
   *
   * @param dimensions input vector with the set of dimensions that represent the problem.
   */
  AATB::AATB(const iVector1D &dimensions)
  {
    dims = dimensions;
    genEmptyInput();
    genEmptyInter();
    computeFLOPs();
  }

  /**
   * @brief Additional constructor with individual dimensions.
   *
   * @param m first dimension  - rows of A
   * @param k second dimension - columns of A
   * @param n third dimension  - columns of B
   */
  AATB::AATB(const int m, const int k, const int n)
  {
    iVector1D dimensions = {m, k, n};
    dims = dimensions;
    genEmptyInput();
    genEmptyInter();
    computeFLOPs();
  }

  /**
   * @brief Destroy the AATB object. Default destructor.
   */
  AATB::~AATB() {}

  /**
   * @brief Returns the #FLOPs for all the algorithms
   *
   * @return Vector with the #FLOPs for each algorithm.
   */
  std::vector<unsigned long> AATB::getFLOPs()
  {
    return FLOPs;
  }

  /**
   * @brief Returns the #FLOPs for all the algorithms and each operation. Returns a 2D vector
   * where each 1D vector corresponds to an algorithm. Inside each 1D vector there is an element
   * for each operation and, at the end, another element for the total number of FLOPs for that
   * algorithm.
   *
   * @return 2D vector where each element corresponds to an algorithm.
   */
  std::vector<std::vector<unsigned long>> AATB::getFLOPsInd()
  {
    std::vector<std::vector<unsigned long>> flops_all;
    flops_all.push_back(flopsIndAlg0());
    flops_all.push_back(flopsIndAlg1());
    flops_all.push_back(flopsIndAlg2());
    flops_all.push_back(flopsIndAlg3());
    flops_all.push_back(flopsIndAlg4());
    return flops_all;
  }

  /**
   * @brief Executes all the algorithms and returns the execution times.
   *
   * The purpose is to measure execution times of all the algorithms. The input matrices are
   * allocated within this function, in such a way that these matrices are only allocated once.
   * As expected, these matrices are freed at the end of the function.
   *
   * @param iterations Integer that specifies the number of times each algorithm is executed.
   * @param n_theads Integer with the number of threads to be used.
   *
   * @return A 2D vector with all the execution times for the algorithms.
   */
  dVector2D AATB::executeAll(const int iterations, const int n_threads) {}

  /**
   * @brief Executes all the algorithms and measures the execution time of
   * each operation individually.
   *
   * The purpose is to measure the execution time of each operation. The input matrices are
   * allocated within this function - only once. As expected, these matrices are freed at the
   * end on the function.
   *
   * @param iterations Number of times each algorithm is executed.
   * @param n_threads  Number of threads to be used.
   * @return 3 levels: {algorithm, operation, iterations}
   */
  dVector3D AATB::executeAllInd(const int iterations, const int n_threads) {}

  /**
   * @brief Executes all the algorithms and measures the execution time of
   * each operation without cache effects.
   *
   * The purpose is to measure the execution time of each operation without cache effects
   * The input matrices are allocated within this function - only once. As expected,
   * these matrices are freed at the end on the function.
   *
   * @param iterations Number of times each algorithm is executed.
   * @param n_threads  Number of threads to be used.
   * @return 3 levels: {algorithm, operation, iterations}
   */
  dVector3D AATB::executeAllIsolated(const int iterations, const int n_threads) {}

  /**
   * @brief Sets the set of dimensions of the problem to be the one given as an argument.
   *
   * @param dimensions input vector with the set of dimensions that represent the new problem.
   */
  void AATB::setDims(const iVector1D &dimensions) {}

  /**
   * @brief Get the dimensions of the object.
   *
   * @return vector<int> with the dimensions.
   */
  iVector1D AATB::getDims() const {
    return dims;
  }

  /**
   * @brief returns the number of algorithms (5).
   */
  unsigned AATB::getNumAlgs() const {
    return 5U;
  }

  /**
   * @brief Computes the cost in terms of FLOPs for each algorithm.
   */
  void AATB::computeFLOPs()
  {
    FLOPs.clear();
    FLOPs.push_back(flopsAlg0());
    FLOPs.push_back(flopsAlg1());
    FLOPs.push_back(flopsAlg2());
    FLOPs.push_back(flopsAlg3());
    FLOPs.push_back(flopsAlg4());
  }

  unsigned long AATB::flopsAlg0()
  {
    unsigned long flops = 0;
    flops += static_cast<unsigned long>(dims[0]) * static_cast<unsigned long>(dims[0] + 1) *
             static_cast<unsigned long>(dims[1]);
    flops += static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
             static_cast<unsigned long>(dims[2]);
    return flops;
  }

  unsigned long AATB::flopsAlg1()
  {
    unsigned long flops = 0;
    flops += static_cast<unsigned long>(dims[0]) * static_cast<unsigned long>(dims[0] + 1) *
             static_cast<unsigned long>(dims[1]);
    flops += static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
             static_cast<unsigned long>(dims[2]);
    return flops;
  }

  unsigned long AATB::flopsAlg2()
  {
    unsigned long flops = 0;
    flops += static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[1]) *
             static_cast<unsigned long>(dims[0]);
    flops += static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
             static_cast<unsigned long>(dims[2]);
    return flops;
  }

  unsigned long AATB::flopsAlg3()
  {
    unsigned long flops = 0;
    flops += static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[1]) *
             static_cast<unsigned long>(dims[0]);
    flops += static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
             static_cast<unsigned long>(dims[2]);
    return flops;
  }

  unsigned long AATB::flopsAlg4()
  {
    unsigned long flops = 0;
    flops += static_cast<unsigned long>(2 * dims[1]) * static_cast<unsigned long>(dims[0]) *
             static_cast<unsigned long>(dims[2]);
    flops += static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[1]) *
             static_cast<unsigned long>(dims[2]);
    return flops;
  }

  std::vector<unsigned long> AATB::flopsIndAlg0()
  {
    std::vector<unsigned long> flops;
    flops.push_back(static_cast<unsigned long>(dims[0]) * static_cast<unsigned long>(dims[0] + 1) *
                    static_cast<unsigned long>(dims[1]));
    flops.push_back(static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
                    static_cast<unsigned long>(dims[2]));
    flops.push_back(FLOPs[0]);
    return flops;
  }

  std::vector<unsigned long> AATB::flopsIndAlg1()
  {
    std::vector<unsigned long> flops;
    flops.push_back(static_cast<unsigned long>(dims[0]) * static_cast<unsigned long>(dims[0] + 1) *
                    static_cast<unsigned long>(dims[1]));
    flops.push_back(static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
                    static_cast<unsigned long>(dims[2]));
    flops.push_back(FLOPs[1]);
    return flops;
  }

  std::vector<unsigned long> AATB::flopsIndAlg2()
  {
    std::vector<unsigned long> flops;
    flops.push_back(static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[1]) *
                    static_cast<unsigned long>(dims[0]));
    flops.push_back(static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
                    static_cast<unsigned long>(dims[2]));
    flops.push_back(FLOPs[2]);
    return flops;
  }

  std::vector<unsigned long> AATB::flopsIndAlg3()
  {
    std::vector<unsigned long> flops;
    flops.push_back(static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[1]) *
                    static_cast<unsigned long>(dims[0]));
    flops.push_back(static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[0]) *
                    static_cast<unsigned long>(dims[2]));
    flops.push_back(FLOPs[3]);
    return flops;
  }

  std::vector<unsigned long> AATB::flopsIndAlg4()
  {
    std::vector<unsigned long> flops;
    flops.push_back(static_cast<unsigned long>(2 * dims[1]) * static_cast<unsigned long>(dims[0]) *
                    static_cast<unsigned long>(dims[2]));
    flops.push_back(static_cast<unsigned long>(2 * dims[0]) * static_cast<unsigned long>(dims[1]) *
                    static_cast<unsigned long>(dims[2]));
    flops.push_back(FLOPs[4]);
    return flops;
  }

  /**
   * @brief Changes the dimensions of the input matrices.
   *
   * This function is only used when the original set of dimensions is replaced by a new
   * set of dimensions.
   */
  void AATB::resizeInput() {}

  /**
   * @brief Generates empty input matrices.
   *
   * These matrices have names and sizes. However,
   * there is no memory allocation performed within this function.
   */
  void AATB::genEmptyInput()
  {
    A.name = "A";
    A.rows = dims[0];
    A.columns = dims[1];

    B.name = "B";
    B.rows = dims[0];
    B.columns = dims[2];
  }

  /**
   * @brief Generates empty intermediate matrix.
   *
   * This matrix has a name, however, it does not have sizes nor allocated memory, since
   * those depend on the actual algorithm to be executed.
   */
  void AATB::genEmptyInter()
  {
    M.name = "M";
  }

  /**
   * @brief Allocates memory for the input matrices.
   *
   * This function allocates memory for the input matrices and initialises them with
   * random values in the range [0,1).
   */
  void AATB::allocInput() {}

  /**
   * @brief Allocates memory for the intermediate matrix.
   *
   * This function allocates memory for the intermediate matrix and initialises it
   * with zeroes. The actual sizes of this matrix depend on the algorithm to be solved.
   *
   * @param alg
   */
  void AATB::allocInter() {}

  /**
   * @brief Frees the memory allocated to the matrices within the input vector.
   *
   * @param matrices Vector with matrices of which the memory is freed.
   */
  void AATB::freeMatrices(std::vector<Matrix> &matrices) {}

} // end namespace aatb