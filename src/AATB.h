#ifndef __AATB_H__
#define __AATB_H__

#include <string>
#include <vector>

#include "common.h"

namespace aatb
{

  class AATB
  {
  private:
    iVector1D dims;

    Matrix A, B; // Input matrices, A and B
    Matrix M;    // Intermediate result matrix
    Matrix X;    // Output matrix

    std::vector<unsigned long> FLOPs;

  public:
    /**
     * @brief Construct a new AATB object. Default constructor.
     */
    AATB();

    /**
     * @brief Additional constructor with a set of dimensions.
     *
     * This constructor uses the input set of dimensions to create empty input and intermediate
     * matrices that are used to compute the corresponding cost in terms of FLOPs.
     *
     * @param dimensions input vector with the set of dimensions that represent the problem.
     */
    AATB(const iVector1D &dimensions);

    /**
     * @brief Additional constructor with individual dimensions.
     *
     * @param m first dimension  - rows of A
     * @param k second dimension - columns of A
     * @param n third dimension  - columns of B
     */
    AATB(const int m, const int k, const int n);

    /**
     * @brief Destroy the AATB object. Default destructor.
     */
    ~AATB();

    /**
     * @brief Returns the #FLOPs for all the algorithms
     *
     * @return Vector with the #FLOPs for each algorithm.
     */
    std::vector<unsigned long> getFLOPs();

    /**
     * @brief Returns the #FLOPs for all the algorithms and each operation. Returns a 2D vector
     * where each 1D vector corresponds to an algorithm. Inside each 1D vector there is an element
     * for each operation and, at the end, another element for the total number of FLOPs for that
     * algorithm.
     *
     * @return 2D vector where each element corresponds to an algorithm.
     */
    std::vector<std::vector<unsigned long>> getFLOPsInd();

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
    dVector2D executeAll(const int iterations, const int n_threads);

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
    dVector3D executeAllInd(const int iterations, const int n_threads);

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
    dVector3D executeAllIsolated(const int iterations, const int n_threads);

    /**
     * @brief Sets the set of dimensions of the problem to be the one given as an argument.
     *
     * @param dimensions input vector with the set of dimensions that represent the new problem.
     */
    void setDims(const iVector1D &dimensions);

    /**
     * @brief Sets the set of dimensions of the problem to be the one given as an argument.
     *
     * @param m first dimension  - rows of A
     * @param k second dimension - columns of A
     * @param n third dimension  - columns of B
     */
    void setDims(const int m, const int k, const int n);

    /**
     * @brief Get the dimensions of the object.
     *
     * @return vector<int> with the dimensions.
     */
    iVector1D getDims() const;

    /**
     * @brief returns the number of algorithms (5).
     */
    unsigned getNumAlgs() const;

  private:
    /**
     * @brief Computes algorithm 0 and returns the execution times of all the iterations.
     *
     * @param iterations Number of times the algorithm is executed.
     * @param n_threads  Number of threads to use in the algorithm.
     * @return dVector1D with the times of every iteration.
     */
    dVector1D alg0(const int iterations, const int n_threads);
    dVector1D alg1(const int iterations, const int n_threads);
    dVector1D alg2(const int iterations, const int n_threads);
    dVector1D alg3(const int iterations, const int n_threads);
    dVector1D alg4(const int iterations, const int n_threads);

    /**
     * @brief Computes algorithm 0 and returns execution times for all the operations and
     * the iterations.
     *
     * @param iterations Number of times the algorithm is executed.
     * @param n_threads  Number of threads to use in the algorithm.
     * @return dVector2D with the times for each operation and iteration. {operation, iteration}
     */
    dVector2D alg0Ind(const int iterations, const int n_threads);
    dVector2D alg1Ind(const int iterations, const int n_threads);
    dVector2D alg2Ind(const int iterations, const int n_threads);
    dVector2D alg3Ind(const int iterations, const int n_threads);
    dVector2D alg4Ind(const int iterations, const int n_threads);

    /**
     * @brief Computes algorithm 0 and returns execution times without cache effects for
     * the single operations.
     *
     * @param iterations Number of times the algorithm is executed.
     * @param n_threads  Number of threads to use in the algorithm.
     * @return dVector2D with the times without cache effects for each operation and iteration.
     * {operation, iteration}
     */
    dVector2D alg0Isolated(const int iterations, const int n_threads);
    dVector2D alg1Isolated(const int iterations, const int n_threads);
    dVector2D alg2Isolated(const int iterations, const int n_threads);
    dVector2D alg3Isolated(const int iterations, const int n_threads);
    dVector2D alg4Isolated(const int iterations, const int n_threads);

    /**
     * @brief Computes the cost in terms of FLOPs for each algorithm.
     */
    void computeFLOPs();

    /**
     * @brief Computes #FLOP for algorithm 0.
     *
     * @return FLOP count.
     */
    unsigned long flopsAlg0();

    /**
     * @brief Computes #FLOP for algorithm 1.
     *
     * @return FLOP count.
     */
    unsigned long flopsAlg1();

    /**
     * @brief Computes #FLOP for algorithm 2.
     *
     * @return FLOP count.
     */
    unsigned long flopsAlg2();

    /**
     * @brief Computes #FLOP for algorithm 3.
     *
     * @return FLOP count.
     */
    unsigned long flopsAlg3();

    /**
     * @brief Computes #FLOP for algorithm 4.
     *
     * @return FLOP count.
     */
    unsigned long flopsAlg4();

    /**
     * @brief Computes the #FLOPs for each operation in Algorithm 0
     *
     * @return std::vector<unsigned long>
     */
    std::vector<unsigned long> flopsIndAlg0();

    /**
     * @brief Computes the #FLOPs for each operation in Algorithm 1
     *
     * @return std::vector<unsigned long>
     */
    std::vector<unsigned long> flopsIndAlg1();

    /**
     * @brief Computes the #FLOPs for each operation in Algorithm 2
     *
     * @return std::vector<unsigned long>
     */
    std::vector<unsigned long> flopsIndAlg2();

    /**
     * @brief Computes the #FLOPs for each operation in Algorithm 3
     *
     * @return std::vector<unsigned long>
     */
    std::vector<unsigned long> flopsIndAlg3();

    /**
     * @brief Computes the #FLOPs for each operation in Algorithm 4
     *
     * @return std::vector<unsigned long>
     */
    std::vector<unsigned long> flopsIndAlg4();

    /**
     * @brief Changes the dimensions of the input matrices.
     *
     * This function is only used when the original set of dimensions is replaced by a new
     * set of dimensions.
     */
    void resizeInput();

    /**
     * @brief Generates empty input matrices.
     *
     * These matrices have names and sizes. However,
     * there is no memory allocation performed within this function.
     */
    void genEmptyInput();

    /**
     * @brief Generates empty intermediate matrix.
     *
     * This matrix has a name, however, it does not have sizes nor allocated memory, since
     * those depend on the actual algorithm to be executed.
     */
    void genEmptyInter();

    /**
     * @brief Allocates memory for the input matrices.
     *
     * This function allocates memory for the input matrices and initialises them with
     * random values in the range [0,1).
     */
    void allocInput();

    /**
     * @brief Allocates memory for the intermediate matrix.
     *
     * This function allocates memory for the intermediate matrix and initialises it
     * with zeroes. The actual sizes of this matrix depend on the algorithm to be solved.
     *
     * @param first_left Determines whether the leftmost pair is computed.
     */
    void allocInter(const bool first_left);

    /**
     * @brief Free memory for the input matrices.
     *
     */
    void freeInput();

    /**
     * @brief Frees the memory allocated to the matrices within the input vector.
     *
     * @param matrices Vector with matrices of which the memory is freed.
     */
    void freeInter();

    /**
     * @brief Copies the upper triangle in M into the lower triangle (only for algorithm 1).
     * 
     */
    void copyHalfInterm();
  };

} // end namespace aatb

#endif