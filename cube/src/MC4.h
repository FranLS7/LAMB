#ifndef MC4_FUNC
#define MC4_FUNC

#include <vector>

namespace mc4 {
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
std::vector<std::vector<double>> MC4_execute_par (std::vector<int> dims, std::vector<int> parenths,
    const int iterations, const int n_threads);

/**
 * Executes a certain parenthesisation of the Matrix Chain of length 4 problem. Takes as input 
 * the dimensions of the problem, the input matrices already allocated and initialised, the
 * parenthesisation to compute, the number of iterations to measure the time, and the number
 * of threads to use. Returns the execution times in the form of a vector.
 * 
 * @param dims              Vector containing the dimensions of the problem.
 * @param input_matrices    Vector containing the input matrices already allocated and initialised.
 * @param parenth           Indicates which parenthesisation to evaluate.
 * @param iterations        Number of iterations to execute the problem.
 * @param n_threads         Number of threads to use during computation.
 * @return                  Execution times in the form of a vector.
 */
std::vector<double> MC4_execute (std::vector<int> dims, std::vector<double*> input_matrices,
    const int parenth, const int iterations, const int n_threads);

/**
 * Each of these functions computes a certain MC4 parenthesisation and returns
 * execution times. These functions assume the initial matrices have been allocated 
 * and initialised.
 *
 * @param dims              The vector with the matrices dimensions.
 * @param input_matrices    Vector containing pointers to memory allocated for matrices.
 * @param iterations        The number of iterations to compute.
 * @param nthreads          The number of threads to used during computation.
 * @return                  The execution times in the form of a vector.
 */
std::vector<double> MC4_p0 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads);

std::vector<double> MC4_p1 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads);

std::vector<double> MC4_p2 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads);

std::vector<double> MC4_p3 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads);

std::vector<double> MC4_p4 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads);

std::vector<double> MC4_p5 (std::vector<int> &dims, std::vector<double*> input_matrices,
    const int iterations, const int n_threads);

/**
 * Computes the amount of FLOPs for one parenthesisation.
 *
 * @param dims    The vector with the matrices dimensions.
 * @param parenth The parenthesisation for which #FLOPs is computed.
 * @return        The number of FLOPs for a given parenthesisation.
 */
unsigned long long int MC4_flops (std::vector<int> dims, int parenth);

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The array with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions.
 */
unsigned long long int MC4_flops_0 (std::vector<int> dims);

unsigned long long int MC4_flops_1 (std::vector<int> dims);

unsigned long long int MC4_flops_2 (std::vector<int> dims);

unsigned long long int MC4_flops_3 (std::vector<int> dims);

unsigned long long int MC4_flops_4 (std::vector<int> dims);


/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been allocated nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_0 (int dims[], double times[], const int iterations, 
    const int n_measures, const int nthreads);

void MC4_parenth_1 (int dims[], double times[], const int iterations, 
    const int n_measures, const int nthreads);

void MC4_parenth_2 (int dims[], double times[], const int iterations, 
    const int n_measures, const int nthreads);

void MC4_parenth_3 (int dims[], double times[], const int iterations, 
    const int n_measures, const int nthreads);

void MC4_parenth_4 (int dims[], double times[], const int iterations, 
    const int n_measures, const int nthreads);

void MC4_parenth_5 (int dims[], double times[], const int iterations, 
    const int n_measures, const int nthreads);

/**
 * Computes the amount of FLOPs for one parenthesisation.
 *
 * @param dims    The array with the matrices dimensions.
 * @param parenth The parenthesisation for which FLOPs is computed.
 * @return        The number of FLOPs for a given parenthesisation.
 */
unsigned long long int MC4_flops (int dims[], int parenth);

/**
 * Each of these functions computes #FLOPs for a certain parenthesisation.
 *
 * @param dims  The array with the matrices dimensions.
 * @return      The #FLOPs that have been computed for those dimensions.
 */
unsigned long long int MC4_flops_0 (int dims[]);
unsigned long long int MC4_flops_1 (int dims[]);
unsigned long long int MC4_flops_2 (int dims[]);
unsigned long long int MC4_flops_3 (int dims[]);
unsigned long long int MC4_flops_4 (int dims[]);
} // namespace mc4

#endif