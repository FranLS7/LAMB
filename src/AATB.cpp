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
  dVector2D AATB::executeAll(const int iterations, const int n_threads)
  {
    dVector2D times;
    allocInput();

    times.push_back(alg0(iterations, n_threads));
    times.push_back(alg1(iterations, n_threads));
    times.push_back(alg2(iterations, n_threads));
    times.push_back(alg3(iterations, n_threads));
    times.push_back(alg4(iterations, n_threads));

    freeInput();
    return times;
  }

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
  dVector3D AATB::executeAllInd(const int iterations, const int n_threads)
  {
    dVector3D times;
    allocInput();

    times.push_back(alg0Ind(iterations, n_threads));
    times.push_back(alg1Ind(iterations, n_threads));
    times.push_back(alg2Ind(iterations, n_threads));
    times.push_back(alg3Ind(iterations, n_threads));
    times.push_back(alg4Ind(iterations, n_threads));

    freeInput();
    return times;
  }

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
  dVector3D AATB::executeAllIsolated(const int iterations, const int n_threads)
  {
    dVector3D times;
    allocInput();

    times.push_back(alg0Isolated(iterations, n_threads));
    times.push_back(alg1Isolated(iterations, n_threads));
    times.push_back(alg2Isolated(iterations, n_threads));
    times.push_back(alg3Isolated(iterations, n_threads));
    times.push_back(alg4Isolated(iterations, n_threads));

    freeInput();
    return times;
  }

  /**
   * @brief Sets the set of dimensions of the problem to be the one given as an argument.
   *
   * @param dimensions input vector with the set of dimensions that represent the new problem.
   */
  void AATB::setDims(const iVector1D &dimensions)
  {
    if (dims.empty())
    {
      dims = dimensions;
      genEmptyInput();
      genEmptyInter();
      computeFLOPs();
    }
    else if (dims.size() == dimensions.size())
    {
      dims = dimensions;
      resizeInput();
      computeFLOPs();
    }
    else
    {
      dims = dimensions;
      genEmptyInput();
      genEmptyInter();
      computeFLOPs();
    }
  }

  /**
   * @brief Sets the set of dimensions of the problem to be the one given as an argument.
   *
   * @param m first dimension  - rows of A
   * @param k second dimension - columns of A
   * @param n third dimension  - columns of B
   */
  void AATB::setDims(const int m, const int k, const int n)
  {
    iVector1D dimensions = {m, k, n};
    setDims(dimensions);
  }

  /**
   * @brief Get the dimensions of the object.
   *
   * @return vector<int> with the dimensions.
   */
  iVector1D AATB::getDims() const
  {
    return dims;
  }

  /**
   * @brief returns the number of algorithms (5).
   */
  unsigned AATB::getNumAlgs() const
  {
    return 5U;
  }

  dVector1D AATB::alg0(const int iterations, const int n_threads)
  {
    dVector1D times(iterations);
    allocInter(true);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
                  A.data, A.rows, 0.0, M.data, M.rows);

      cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
                  M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto end = std::chrono::high_resolution_clock::now();
      times[it] = std::chrono::duration<double>(end - begin).count();
    }
    freeInter();
    return times;
  }

  dVector1D AATB::alg1(const int iterations, const int n_threads)
  {
    dVector1D times(iterations);
    allocInter(true);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
                  A.data, A.rows, 0.0, M.data, M.rows);

      copyHalfInterm();

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
                  1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto end = std::chrono::high_resolution_clock::now();
      times[it] = std::chrono::duration<double>(end - begin).count();
    }
    freeInter();
    return times;
  }

  dVector1D AATB::alg2(const int iterations, const int n_threads)
  {
    dVector1D times(iterations);
    allocInter(true);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
                  1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);

      cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
                  M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto end = std::chrono::high_resolution_clock::now();
      times[it] = std::chrono::duration<double>(end - begin).count();
    }
    freeInter();
    return times;
  }

  dVector1D AATB::alg3(const int iterations, const int n_threads)
  {
    dVector1D times(iterations);
    allocInter(true);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
                  1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
                  1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto end = std::chrono::high_resolution_clock::now();
      times[it] = std::chrono::duration<double>(end - begin).count();
    }
    freeInter();
    return times;
  }

  dVector1D AATB::alg4(const int iterations, const int n_threads)
  {
    dVector1D times(iterations);
    allocInter(false);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M.rows, M.columns, B.rows,
                  1.0, A.data, A.rows, B.data, B.rows, 0.0, M.data, M.rows);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows, M.columns, M.rows,
                  1.0, A.data, A.rows, M.data, M.rows, 0.0, X.data, X.rows);
      auto end = std::chrono::high_resolution_clock::now();
      times[it] = std::chrono::duration<double>(end - begin).count();
    }
    freeInter();
    return times;
  }

  dVector2D AATB::alg0Ind(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    for (auto &op : times)
      op.resize(iterations);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
                  A.data, A.rows, 0.0, M.data, M.rows);

      auto mid = std::chrono::high_resolution_clock::now();
      cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
                  M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);

      auto end = std::chrono::high_resolution_clock::now();

      times[0][it] = std::chrono::duration<double>(mid - begin).count();
      times[1][it] = std::chrono::duration<double>(end - mid).count();
      times.back()[it] = std::chrono::duration<double>(end - begin).count();
    }

    freeInter();
    return times;
  }

  dVector2D AATB::alg1Ind(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    for (auto &op : times)
      op.resize(iterations);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
                  A.data, A.rows, 0.0, M.data, M.rows);
      auto mid_precopy = std::chrono::high_resolution_clock::now();

      copyHalfInterm();

      auto mid_postcopy = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
                  1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);

      auto end = std::chrono::high_resolution_clock::now();

      times[0][it] = std::chrono::duration<double>(mid_precopy - begin).count();
      times[1][it] = std::chrono::duration<double>(end - mid_postcopy).count();
      times.back()[it] = std::chrono::duration<double>(end - begin).count();
    }

    freeInter();
    return times;
  }

  dVector2D AATB::alg2Ind(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    for (auto &op : times)
      op.resize(iterations);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
                  1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);

      auto mid = std::chrono::high_resolution_clock::now();
      cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
                  M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);

      auto end = std::chrono::high_resolution_clock::now();

      times[0][it] = std::chrono::duration<double>(mid - begin).count();
      times[1][it] = std::chrono::duration<double>(end - mid).count();
      times.back()[it] = std::chrono::duration<double>(end - begin).count();
    }

    freeInter();
    return times;
  }

  dVector2D AATB::alg3Ind(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    for (auto &op : times)
      op.resize(iterations);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
                  1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);

      auto mid = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
                  1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);

      auto end = std::chrono::high_resolution_clock::now();

      times[0][it] = std::chrono::duration<double>(mid - begin).count();
      times[1][it] = std::chrono::duration<double>(end - mid).count();
      times.back()[it] = std::chrono::duration<double>(end - begin).count();
    }

    freeInter();
    return times;
  }

  dVector2D AATB::alg4Ind(const int iterations, const int n_threads)
  {
    allocInter(false);

    dVector2D times(3);
    for (auto &op : times)
      op.resize(iterations);

    mkl_set_dynamic(false);
    mkl_set_num_threads(n_threads);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);

      auto begin = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M.rows, M.columns, B.rows,
                  1.0, A.data, A.rows, B.data, B.rows, 0.0, M.data, M.rows);

      auto mid = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows, M.columns, M.rows,
                  1.0, A.data, A.rows, M.data, M.rows, 0.0, X.data, X.rows);

      auto end = std::chrono::high_resolution_clock::now();

      times[0][it] = std::chrono::duration<double>(mid - begin).count();
      times[1][it] = std::chrono::duration<double>(end - mid).count();
      times.back()[it] = std::chrono::duration<double>(end - begin).count();
    }

    freeInter();
    return times;
  }

  dVector2D AATB::alg0Isolated(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    times[0].resize(iterations);
    times[1].resize(iterations);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
                  A.data, A.rows, 0.0, M.data, M.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[0][it] = std::chrono::duration<double>(t1 - t0).count();
    }

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
                  M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[1][it] = std::chrono::duration<double>(t1 - t0).count();
    }
    times[2].push_back(lamb::medianVector(times[0]) + lamb::medianVector(times[1]));

    freeInter();
    return times;
  }

  dVector2D AATB::alg1Isolated(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    times[0].resize(iterations);
    times[1].resize(iterations);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, A.rows, A.columns, 1.0,
                  A.data, A.rows, 0.0, M.data, M.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[0][it] = std::chrono::duration<double>(t1 - t0).count();
    }
    copyHalfInterm();

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
                  1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[1][it] = std::chrono::duration<double>(t1 - t0).count();
    }
    times[2].push_back(lamb::medianVector(times[0]) + lamb::medianVector(times[1]));

    freeInter();
    return times;
  }

  dVector2D AATB::alg2Isolated(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    times[0].resize(iterations);
    times[1].resize(iterations);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
                  1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[0][it] = std::chrono::duration<double>(t1 - t0).count();
    }

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M.rows, B.columns, 1.0,
                  M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[1][it] = std::chrono::duration<double>(t1 - t0).count();
    }
    times[2].push_back(lamb::medianVector(times[0]) + lamb::medianVector(times[1]));

    freeInter();
    return times;
  }

  dVector2D AATB::alg3Isolated(const int iterations, const int n_threads)
  {
    allocInter(true);

    dVector2D times(3);
    times[0].resize(iterations);
    times[1].resize(iterations);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows, A.rows, A.columns,
                  1.0, A.data, A.rows, A.data, A.rows, 0.0, M.data, M.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[0][it] = std::chrono::duration<double>(t1 - t0).count();
    }

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M.rows, B.columns, M.columns,
                  1.0, M.data, M.rows, B.data, B.rows, 0.0, X.data, X.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[1][it] = std::chrono::duration<double>(t1 - t0).count();
    }
    times[2].push_back(lamb::medianVector(times[0]) + lamb::medianVector(times[1]));

    freeInter();
    return times;
  }

  dVector2D AATB::alg4Isolated(const int iterations, const int n_threads)
  {
    allocInter(false);

    dVector2D times(3);
    times[0].resize(iterations);
    times[1].resize(iterations);

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M.rows, M.columns, B.rows,
                  1.0, A.data, A.rows, B.data, B.rows, 0.0, M.data, M.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[0][it] = std::chrono::duration<double>(t1 - t0).count();
    }

    for (int it = 0; it < iterations; ++it)
    {
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      lamb::cacheFlush(n_threads);
      auto t0 = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows, M.columns, M.rows,
                  1.0, A.data, A.rows, M.data, M.rows, 0.0, X.data, X.rows);
      auto t1 = std::chrono::high_resolution_clock::now();
      times[1][it] = std::chrono::duration<double>(t1 - t0).count();
    }
    times[2].push_back(lamb::medianVector(times[0]) + lamb::medianVector(times[1]));

    freeInter();
    return times;
    ;
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
   * @brief Changes the dimensions of the input matrices and the results one.
   *
   * This function is only used when the original set of dimensions is replaced by a new
   * set of dimensions.
   */
  void AATB::resizeInput()
  {
    A.rows = dims[0];
    A.columns = dims[1];

    B.rows = dims[0];
    B.columns = dims[2];

    X.rows = A.rows;
    X.columns = B.columns;
  }

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

    X.name = "X";
    X.rows = A.rows;
    X.columns = B.columns;
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
   * @brief Allocates memory for the input matrices and the result matrix.
   *
   * This function allocates memory for the input matrices and initialises them with
   * random values in the range [0,1). The result matrix is initialised with zeroes.
   */
  void AATB::allocInput()
  {
    A.data = (double *)mkl_malloc(A.rows * A.columns * sizeof(double), ALIGN);
    for (unsigned i = 0; i < A.rows * A.columns; ++i)
      A.data[i] = drand48();

    B.data = (double *)mkl_malloc(B.rows * B.columns * sizeof(double), ALIGN);
    for (unsigned i = 0; i < B.rows * B.columns; ++i)
      B.data[i] = drand48();

    X.data = (double *)mkl_malloc(X.rows * X.columns * sizeof(double), ALIGN);
    for (unsigned i = 0; i < X.rows * X.columns; ++i)
      X.data[i] = 0.0;
  }

  /**
   * @brief Allocates memory for the intermediate matrix.
   *
   * This function allocates memory for the intermediate matrix and initialises it
   * with zeroes. The actual sizes of this matrix depend on the algorithm to be solved.
   *
   * @param first_left Determines whether the leftmost pair is computed.
   */
  void AATB::allocInter(const bool first_left)
  {
    if (first_left)
    {
      M.rows = A.rows;
      M.columns = A.rows;
    }
    else
    {
      M.rows = A.columns;
      M.columns = B.columns;
    }
    M.data = (double *)mkl_malloc(M.rows * M.columns * sizeof(double), ALIGN);
    for (unsigned i = 0; i < M.rows * M.columns; ++i)
      M.data[i] = 0.0;
  }

  /**
   * @brief Free memory for the input matrices.
   */
  void AATB::freeInput()
  {
    mkl_free(A.data);
    mkl_free(B.data);
    mkl_free(X.data);
  }

  /**
   * @brief Frees the memory allocated to the matrices within the input vector.
   *
   * @param matrices Vector with matrices of which the memory is freed.
   */
  void AATB::freeInter()
  {
    mkl_free(M.data);
  }

  void AATB::copyHalfInterm()
  {
    for (unsigned j = 0; j < M.columns; ++j)
    {
      for (unsigned i = j + 1; i < M.rows; ++i)
      {
        M.data[j * M.rows + i] = M.data[i * M.rows + j];
      }
    }
  }

} // end namespace aatb