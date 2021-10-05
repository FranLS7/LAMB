#ifndef MC4_FUNC
#define MC4_FUNC

#include <vector>

/**
 * Computes a certain MC4 parenthesisation. This function assumes ALL initial
 * matrices have been declared and initialised (inside matrices).
 *
 * @param parenth     Number that indicates the parenthesisation to compute.
 * @param matrices    The array that points to the initial matrices
 * @param dims        The array that contains the problem's dimensions.
 * @param times       The array that stores the execution times.
 * @param iterations  The number of times each solution must be computed.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4 (int parenth, double** matrices, int dims[], double times[],
  const int iterations, const int n_measures, const int nthreads);

/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume ALL initial matrices have been
 * declared and initialised (inside matrices).
 *
 * @param matrices    The array that points to the initial matrices
 * @param dims        The array that contains the problem's dimensions.
 * @param times       The array that stores the execution times.
 * @param iterations  The number of times each solution must be computed.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to use during computation.
 * @return            Execution times stored in times.
 */
void MC4_parenth0 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads);
void MC4_parenth1 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads);
void MC4_parenth2 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads);
void MC4_parenth3 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads);
void MC4_parenth4 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads);
void MC4_parenth5 (double** matrices, int dims[], double times[], const int iterations,
  const int n_measures, const int nthreads);

std::vector<double*> generateMatrices (std::vector<int> &dims);

std::vector<std::vector<double>> MC4_execute_par (std::vector<int> &dims, std::vector<int> &parenths,
  const int iterations, const int n_threads);

std::vector<double> MC4_execute (std::vector<int> &dims, std::vector<double*> imatrices,
  const int parenth, const int iterations, const int n_threads);

std::vector<double> MC4_p0 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads);
std::vector<double> MC4_p1 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads);
std::vector<double> MC4_p2 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads);
std::vector<double> MC4_p3 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads);
std::vector<double> MC4_p4 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads);
std::vector<double> MC4_p5 (std::vector<int> &dims, std::vector<double*> imatrices,
  const int iterations, const int n_threads);

/**
 * Each of these functions computes a certain MC4 parenthesisation and stores
 * execution times. These functions assume no prior matrix has been declared nor
 * initialised.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array that contains the obtained execution times.
 * @param iterations  The number of iterations to compute.
 * @param n_measures  Variable used to measure all the steps: set to 1.
 * @param nthreads    The number of threads to used during computation.
 * @return            The execution times stored in times.
 */
void MC4_parenth_0 (int dims[], double times[], const int iterations, const int n_measures, const int nthreads);
void MC4_parenth_1 (int dims[], double times[], const int iterations, const int n_measures, const int nthreads);
void MC4_parenth_2 (int dims[], double times[], const int iterations, const int n_measures, const int nthreads);
void MC4_parenth_3 (int dims[], double times[], const int iterations, const int n_measures, const int nthreads);
void MC4_parenth_4 (int dims[], double times[], const int iterations, const int n_measures, const int nthreads);
void MC4_parenth_5 (int dims[], double times[], const int iterations, const int n_measures, const int nthreads);

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

// These functions are the same but are applied upon vectors.
unsigned long long int MC4_flops (std::vector<int> dims, int parenth);
unsigned long long int MC4_flops_0 (std::vector<int> dims);
unsigned long long int MC4_flops_1 (std::vector<int> dims);
unsigned long long int MC4_flops_2 (std::vector<int> dims);
unsigned long long int MC4_flops_3 (std::vector<int> dims);
unsigned long long int MC4_flops_4 (std::vector<int> dims);

#endif