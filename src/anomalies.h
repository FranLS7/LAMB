#ifndef ANOMALIES_FUNC
#define ANOMALIES_FUNC

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace lamb{
/**
 * @struct Anomaly
 * @brief This structure represents an anomaly, which is given by a set of dimensions,
 * the algorithms involved, and the number of threads used. With these values,
 * we obtain the flops_score and time_score.
 * 
 * @param dims        Vector containing the dimensions of the anomalous problem.
 * @param algs        Vector containing the algorithms of the anomaly.
 * @param n_threads   Number of threads used.
 * @param flops_score Score in terms of #FLOPs.
 * @param time_score  Score in terms of time.
 * @param isAnomaly   Boolean field indicating whether it's an anomaly or not.
 */
struct Anomaly {
  std::vector<int> dims; 
  std::vector<int> algs;
  int n_threads;
  double flops_score, time_score;
  bool isAnomaly;
};

/**
 * @struct exploratorySpace
 * @brief This structure contains the queue and hashTable used during the iterative search for
 * anomalies (== the exploration of volumes).
 * 
 * @param queue Deque containing points to be checked in the future. 
 * @param checked HashTable containing points that have already been checked. The keys in this
 *    hashTable are formed serialising the dimensions that identify each problem.
 */
struct exploratorySpace {
  std::deque<std::vector<int>> queue;
  std::unordered_map<std::string, Anomaly> checked;
};

/**
 * Checks whether there is at least one dimension within the threshold.
 * This way we increase the possibilities of finding anomalies.
 *
 * @param dims      The array with the matrices dimensions.
 * @param threshold The max size for at least one dimension.
 * @return          True if at least one dimension is smaller than threshold.
 * False, otherwise.
 */
bool computation_decision (std::vector<int> dims, const int threshold = 500);

// @TODO -- write comments for this function.
Anomaly analysePoint(dVector2D& times, std::vector<unsigned long>& flops, const double min_margin);

/**
 * Checks whether there is an anomaly in the computed results.
 *
 * @param candidate   anomaly struct that might be an anomaly.
 * @param times       The array containing all the execution times.
 * @param iterations  The number of times each algorithm has been computed.
 * @param lo_margin   Minimum difference between median values to consider a
 * point as an anomaly.
 * @param ratio       The percentage of total executions for which the
 * algorithm with more FLOPs must be faster.
 */
void search_anomaly_pair(Anomaly &candidate, std::vector<std::vector<double>> times,
    const double ratio);

/**
 * Looks for anomalies and returns them within a vector.
 * 
 * @param dims        Vector with the problem's dimensions.
 * @param times       2D vector with all the algorithms execution times.
 * @param algs        Vector with all the algorithms involved.
 * @param lo_margin   Minimum difference between median values to consider a
 *                    point as an anomaly.
 * @param ratio       The percentage of total executions for which the
 *                    algorithm with more FLOPs must be faster.
 * @param n_threads   Number of threads used for computation.
 * @return            The anomalies found.
 */
std::vector<Anomaly> search_anomaly (std::vector<int> dims, std::vector<std::vector<double>> times, 
    std::vector<int> algs, const double lo_margin, const double ratio, const int n_threads);

std::vector<Anomaly> search_anomaly_MCX (const std::vector<int>& dims, 
    const std::vector<unsigned long>& flops, std::vector<std::vector<double>>& times, 
    const std::vector<int>& algs, const double lo_margin, 
    const double ratio, const int n_threads);

/**
 * Checks whether there is an anomaly between one pair of algorithms.
 *
 * @param flops_a     The #FLOPs for the first algorithm.
 * @param flops_b     The #FLOPs for the second algorithm.
 * @param times_a     The execution times for the first algorithm.
 * @param times_b     The execution times for the second algorithm.
 * @param iterations  The number of times each algorithm is computed.
 * @param ratio       The percentage of total executions for which the
 *                      algorithm with more FLOPs must be faster.
 * @return            Whether there is an anomaly for this pair.
 */
bool is_anomaly (const unsigned long long int flops_a, const unsigned long long int flops_b,
  const std::vector<double> times_a, const std::vector<double> times_b, const double ratio);

/**
 * Explores a certain limited hyperspace around a hit (anomaly)
 *  
 * @param hit         The hit around which the region is determined.
 * @param iterations  The number of iterations for each algorithm.
 * @param span        The maximum modification in each dimension (greater/smaller).
 * @param jump        The difference between contiguous points in a given dimension.
 * @param ratio       Relative number of times a alg must be faster than the other one.
 * @param ofile       The output file where results are printed.
 */
std::deque<Anomaly> gridExploration (Anomaly &hit, const int iterations, 
    const int span, const int jump, const double ratio);

/**
 * Explores a hyperspace around a hit, adding the neighbours of each anomaly found in the process
 * to the list of points to be checked. The process stops when there are no more points to be 
 * checked or when max_points is reached.
 * 
 * @param hit           The hit around which the region is explored.
 * @param iterations    The number of iterations for each algorithm.
 * @param jump          The difference between contiguous points in a given dimension.
 * @param min_margin    Minimum relative difference in time to be considered an anomaly.
 * @param max_points    The total number of points to be checked.
 */
std::unordered_map<std::string, Anomaly> iterativeExplorationMCX(const Anomaly &hit, 
    const int iterations, const int jump, const double min_margin, const unsigned max_points);

std::vector<std::vector<DataPoint>> arrowMCX(const Anomaly& hit, const int iterations, 
    const int jump, const double min_margin, const int max_out, 
    std::vector<lamb::Anomaly>& summary);

std::unordered_map<std::string, Anomaly> iterativeExplorationAATB(const Anomaly &hit, 
    const int iterations, const int jump, const double min_margin, const unsigned max_points);

/**
 * Explores the space around a certain initial anomaly (hit) by modifying a single dimension
 * independently. This means that only one dimension is modified at a given step in the process.
 * Therefore, the result of modifying multiple dimensions is not explored, but individual changes.
 * The results are printed in the output file.
 * 
 * @param hit         The hit around which the dimensions are individually explored.
 * @param iterations  The number of iterations for each algorithm.
 * @param span        The maximum change in each dimension (greater/smaller).
 * @param jump        The difference between contiguous points in a given dimension.
 * @param ratio       Relative number of times a alg must be faster than the other one.
 * @param ofile       The output file where results are printed.
 */
std::deque<Anomaly> dimExploration (Anomaly &hit, const int iterations, 
    const int span, const int jump, const double ratio);

/**
 * Individually explores all dimensions of a given anomaly, checking all the neighbours that 
 * only differ in one single jump in a dimension. In doing so, we have to check whether the 
 * new points to add to the queue already exist in such queue or have already been checked.
 * 
 * @param space   Struct that contains the queue and list of points already checked.
 * @param a       Initial anomaly of which neighbours are explored.
 * @param jump    The difference between contiguous points in a given dimension.
 */
void addNeighbours (exploratorySpace &space, const Anomaly &a, const int jump);

/**
 * Adds the headers to an output file for anomalies. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score |
 *
 * @param ofile   Output file manager already opened.
 * @param ndim    The number of dimensions in the problem.
 */
void print_header_anomalies (std::ofstream &ofile, const int ndim);

/**
 * Adds a line to an output file/stream for anomalies. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score |
 *
 * @param ofile Output manager already opened.
 * @param an    The anomaly to be printed in the file.
 */
void print_anomaly (std::ostream &ofile, const Anomaly& an);

/**
 * Prints the headers to an output validation file for anomalies. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score || old_time_score |
 *
 * @param ofile   Output file manager already opened.
 * @param ndim    The number of dimensions in the problem.
 */
void print_header_validation (std::ofstream &ofile, const int ndim);

/**
 * Prints a line in the output anomaly validation file. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score || old_time_score |
 *
 * @param ofile           Output file manager already opened.
 * @param an              The anomaly to be printed in the file.
 * @param new_time_score  The old time score validated.
 */
void print_validation (std::ofstream &ofile, const Anomaly& an, const float old_time_score);

/**
 * Adds the headers to an output file in validation phase. Format:
 * | ndim dims || alg || nsamples samples |
 *
 * @param ofile     The output file manager, which has been previously opened.
 * @param ndim      The number of problem dimensions.
 * @param nsamples  The number of samples that will be computed.
 */
void print_header_val_time (std::ofstream &ofile, const int ndim, const int nsamples);

/**
 * Adds a line in the already opened ofile for validation phase. Format:
 * | dims || n_alg || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      The array with the problem dimensions.
 * @param ndim      The number of dimensions in the problem.
 * @param times     The array that contains the execution times (already computed).
 * @param nsamples  The number of samples to store in the output file.
 * @param alg        Value that indicates which algorithm is stored.
 */
void print_val_time (std::ofstream &ofile, const int dims[], const int ndim, 
    const double times[], const int nsamples, const int alg);

} // namespace lamb
#endif