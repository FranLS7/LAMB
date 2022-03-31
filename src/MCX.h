#ifndef __MCX_H__
#define __MCX_H__

#include <string>
#include <vector>

#include "common.h"

namespace mcx {

class MCX {
  public:
    /**
     * @brief Default constructor.
     */ 
    MCX();

    /**
     * @brief Additional constructor with a set of dimensions.
     * 
     * This constructor uses the input set of dimensions to create empty input and intermediate 
     * matrices that are used to generate all the possible algorithms to solve that Matrix Chain 
     * and the corresponding cost in terms of FLOPs.
     * 
     * @param dimensions input vector with the set of dimensions that represent the problem.
     */
    MCX(const iVector1D& dimensions);

    /**
     * @brief Default destructor.
     */
    ~MCX();

    /**
     * @brief Sets the set of dimensions of the problem to be the one given as an argument.
     * 
     * Depending on the current size of the object, this function may perform the same operations
     * the additional constructor does (if the object is empty) or just resize the input matrices 
     * and recompute the #FLOPs of each algorithm.
     * 
     * @param dimensions input vector with the set of dimensions that represent the new problem.
     */
    void setDims (const iVector1D& dimensions);

    /**
     * @brief Generates all the algoritms that solve the current Matrix Chain problem.
     * 
     * This function generates all the algorithms that solve the Matrix Chain problem of 
     * a certain size, given by the number of matrices previously created. This function 
     * works with pointers to these matrices, making a given algorithm to actually be
     * a set of vectors with pointers to matrices. The first two pointers are the input
     * arguments to GEMM and the third one is where the result is stored. This function uses
     * a recursive function to generate all the possible algorithms.
     * 
     * @return A vector with all the algorithms where each algorithm is a 2D vector of pointers
     * to Matrix structs.
     */
    std::vector<std::vector<std::vector<Matrix*>>> generateAlgorithms();

    /**
     * @brief Returns the #FLOPs for all the previously generated algorithms.
     * 
     * @return A vector that contains the number of FLOPs for each algorithm.
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
     * @brief Executes a set of algorithms.
     * 
     * The main purpose of this function is to measure execution times of the algorithms 
     * specified in the alg_ids variable. The input matrices are allocated within this 
     * function, in such a way that these matrices are only allocated once. As expected,
     * these matrices are freed at the end of the function. This function uses its
     * homonym to execute each of the algorithms.
     * 
     * @param alg_ids Vector with the set of algorithms IDs to execute.
     * @param iterations Integer that specifies the number of times the algorithm is executed.
     * @param n_theads Integer with the number of threads to be used.
     * 
     * @return A 2D vector with all the execution times for the specified algorithms.
     */
    dVector2D execute(const std::vector<unsigned>& alg_ids, 
        const int iterations, const int n_threads);

    /**
     * @brief Executes all the algorithms that have been generated.
     * 
     * The purpose is to measure execution times of all the algorithms that have been generated
     * for the Matrix Chain problem. The input matrices are allocated within this 
     * function, in such a way that these matrices are only allocated once. As expected,
     * these matrices are freed at the end of the function. This function uses the 'execute'
     * function to execute each of the algorithms.
     * 
     * @param iterations Integer that specifies the number of times the algorithm is executed.
     * @param n_theads Integer with the number of threads to be used.
     * 
     * @return A 2D vector with all the execution times for the algorithms.
     */
    dVector2D executeAll(const int iterations, const int n_threads);

    dVector3D executeAllInd(const int iterations, const int n_threads);

    dVector3D executeAllIsolated(const int iterations, const int n_threads);

    /**
     * @brief returns the set of dimensions.
     */
    iVector1D getDims() const;

    /**
     * @brief returns the number of algorithms that have been generated.
     */
    unsigned getNumAlgs() const;

    /**
     * @brief Function that generates solutions in the form of sizes for GEMM operations.
     * 
     * @return All the algorithms in the form of vectors of vectors of int, where the 
     * sizes are specified.
     */
    std::vector<std::vector<std::vector<int>>> genDims();

    // @TODO ERASE THESE FUNCTIONS
    std::vector<Matrix> getInMat ();

  private:

    iVector1D dims;

    std::vector<Matrix> input_matrices;
    std::vector<Matrix> inter_matrices;
    
    std::vector<unsigned long> FLOPs;

    std::vector<std::vector<std::vector<Matrix*>>> algorithms;



    // ==================================================================
    //   - - - - - - - - - - - Private functions  - - - - - - - - - - -
    // ==================================================================

    /**
     * @brief Executes a certain algorithm.
     * 
     * This function executes an algorithm based on its ID as many times as specified by
     * the iterations variable and using as many threads as the n_threads variable indicates. 
     * The main purpose of this function is to measure the execution time the specified algorithm
     * takes to solve the Matrix Chain. The input matrices are supposed to have been allocated 
     * beforehand, whereas the intermediate matrices are allocated within this function because 
     * their sizes depend on the actual algorithm to be executed. 
     * 
     * @param alg_id Unsigned that identifies the algorithm to be executed.
     * @param iterations Integer that specifies the number of times the algorithm is executed.
     * @param n_theads Integer with the number of threads to be used.
     * 
     * @return A vector with the execution times for each iteration.
     */
    dVector1D execute(const unsigned alg_id, const int iterations, 
        const int n_threads);

    /**
    * @brief Executes a certain algorithm and measures each GEMM's execution time. The 2D output vector
    * has a vector for each operation and another one for the entire execution time. These 1D vectors
    * contain repetitions of the same operation.
    * 
    * @param alg_id Unsigned that identifies the algorithm to be executed.
    * @param iterations Integer that specifies how many times the algorithm is executed.
    * @param n_threads Integer with the number of threads to be used.
    *
    * @return dVector2D with the execution time of all kernels - total execution in the element.
    */
    dVector2D executeInd(const unsigned alg_id, const int iterations, const int n_threads);

    /**
     * @brief Executes a certain algorithm and measures each GEMM's execution time without caching
     * effects. This means the cache is flushed before each GEMM's execution. The output 2D vector
     * contains a vector for each operation. These 1D vectors (each) contain repetitions of the 
     * same operation.
     * 
     * @param alg_id 
     * @param iterations 
     * @param n_threads 
     * @return dVector2D with the execution time of all kernels without cache effects.
     */
    dVector2D executeIsolated(const unsigned alg_id, const int iterations, const int n_threads);

    /**
     * @brief Recursive function that, for a set of input matrices and intermediate matrices,
     * generates all the algorithms in the form of pointers to matrices.
     * 
     * This recursive function generates all the possible algorithms that solve a given 
     * Matrix Chain problem. The inputs are vectors of pointers to Matrix structs and the 
     * level of depth within the tree-like structure. 
     * 
     * @param expression     Vector of pointers to matrices where the actual expression of which a 
     *    solution must be found is represented.
     * @param inter_matrices Vector of pointers to the intermediate matrices.
     * @param depth          Level of depth within the tree-like structure.
     */
    std::vector<std::vector<std::vector<Matrix*>>> recursiveGen (std::vector<Matrix*> expression,
        std::vector<Matrix*> out_matrices, const int depth);

    /**
     * @brief Recursive function that generates all the algorithms in the form of set of dimensions.
     * 
     * @param dimensions The set of dimensions of which solutions must be generated.
     * 
     * @return All the algorithms in the form of vectors of vectors of int, where the 
     * sizes are specified.
     */
    std::vector<std::vector<std::vector<int>>> recursiveGeneration (iVector1D dimensions);

    /**
     * @brief Computes the cost in terms of FLOPs for each algorithm.
     */
    void computeFLOPs();

    /**
     * @brief Generates input matrices.
     * 
     * This function takes into account the number of dimensions in the problem
     * and generates Matrix structs with names, dimensions and proper data allocation.
     */
    void genInputMatrices();

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
     * These matrices have names, sizes and are placed in the corresponding vector. However, 
     * there is no memory allocation performed within this function.
     */
    void genEmptyInput();
    
    /**
     * @brief Generates empty intermediate matrices.
     * 
     * These matrices have names and are placed in the corresponding vector. However, 
     * they do not have sizes nor allocated memory, since those depend on the actual
     * algorithm to be executed.
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
     * @brief Allocates memory for the intermediate matrices.
     * 
     * This function allocates memory for the intermediate matrices and initialises them 
     * with zeroes. The actual sizes of these matrices depend on the algorithm to be solved.
     * 
     * @param alg 2D vector with pointers to matrices -- an algorithm to solve the Matrix Chain.
     */
    void allocInter(const std::vector<std::vector<Matrix*>>& alg);

    /**
     * @brief Frees the memory allocated to the matrices within the input vector.
     * 
     * @param matrices Vector with matrices of which the memory is freed.
     */ 
    void freeMatrices(std::vector<Matrix>& matrices);




};

} // namespace mcx


#endif