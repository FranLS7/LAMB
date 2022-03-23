#ifndef MC3_FUNC
#define MC3_FUNC

#include <vector>

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
    const int iterations, const int n_threads);

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
    const int parenth, const int iterations, const int n_threads);

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
    const int iterations, const int n_threads);

std::vector<double> MC3_p1 (std::vector<int> &dims, std::vector<double*> imatrices,
    const int iterations, const int n_threads);

/**
 * Computes the amount of FLOPs for one parenthesisation.
 *
 * @param dims    The vector with the matrices dimensions.
 * @param parenth The parenthesisation for which #FLOPs is computed.
 * @return        The number of FLOPs for a given parenthesisation.
 */
unsigned long long int MC3_flops (std::vector<int> dims, int parenth);

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The vector with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions and parenthesisation.
 */
unsigned long long int MC3_flops_0 (std::vector<int> dims);

unsigned long long int MC3_flops_1 (std::vector<int> dims);

} // namespace mc3

#endif