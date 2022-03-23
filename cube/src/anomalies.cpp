#include "anomalies.h"

#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mkl.h"

#include "common.h"
#include "MCX.h"
#include "MC4.h"
#include "operation.h"

namespace lamb {
/**
 * Checks whether there is at least one dimension within the threshold.
 * This way we increase the possibilities of finding anomalies.
 *
 * @param dims      The array with the matrices dimensions.
 * @param threshold The max size for at least one dimension.
 * @return          True if at least one dimension is smaller than threshold. False, otherwise.
 */
bool computation_decision (std::vector<int> dims, const int threshold) {
  bool compute = false;

  for (unsigned i = 0; i < dims.size(); i++){
    if (dims[i] <= threshold){
      compute = true;
      break;
    }
  }
  return compute;
}

Anomaly analysePoint(dVector2D& times, std::vector<unsigned long>& flops, const double min_margin) {
  Anomaly aux;
  aux.isAnomaly = false;
  // aux.algs = {-1, -1};
  aux.flops_score = 0.0;
  aux.time_score = 0.0;
  dVector1D medians;

  for (unsigned i = 0; i < times.size(); ++i) medians.push_back(medianVector<double>(times[i]));

  unsigned id_min_flops = idxMinVector(flops);
  unsigned id_min_times = idxMinVector(times);
  aux.algs = {static_cast<int>(id_min_flops), static_cast<int>(id_min_flops)};

  // double time_score = score(medians[id_min_flops], medians[id_min_times]);
  if (id_min_flops == id_min_times || flops[id_min_flops] == flops[id_min_times]){
    aux.isAnomaly = false;
  }
  else if (medians[id_min_times] < ((1 - min_margin) * medians[id_min_flops])) {
    aux.isAnomaly = true;
    aux.algs = {static_cast<int>(id_min_times), static_cast<int>(id_min_flops)};
    aux.flops_score = score(flops[id_min_flops], flops[id_min_times]);
    aux.time_score = score(medians[id_min_flops], medians[id_min_times]);
  }
  std::cout << "INSIDE ANALYSEPOINT ...\n";
  std::cout << "isAnomaly: " << aux.isAnomaly << std::endl;
  std::cout << "algs: " << aux.algs[0] << ", " << aux.algs[1] << std::endl;
  std::cout << "flops_score: " << aux.flops_score << std::endl;
  std::cout << "time_score:  " << aux.time_score  << std::endl;

  return aux;
}

/**
 * Checks whether there is an anomaly in the computed results.
 * 
 * These results correspond to just one pair of algorithms, therefore, times is 
 * a vector with 2 vectors of doubles, where each of these doubles belongs to a certain
 * execution of one of the algorithms. The dimensions, algorithms and number of 
 * threads to use are included within the /anomaly/ variable.
 *
 * @param candidate   anomaly struct that might be an anomaly.
 * @param times       The array containing the execution times of both algorithms.
 * @param ratio       The percentage of total executions for which the
 * algorithm with more FLOPs must be faster.
 */
void search_anomaly_pair(Anomaly &candidate, std::vector<std::vector<double>> times,
    const double ratio) {

  std::vector<unsigned long long int> flops;
  std::vector<double> medians;

  unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  for (unsigned p = 0; p < candidate.algs.size(); p++){
    flops.push_back (mc4::MC4_flops(candidate.dims, candidate.algs[p]));
    medians.push_back (medianVector (times[p]));
    shuffle (times[p].begin(), times[p].end(), std::default_random_engine(seed));
  }

  candidate.time_score = score (medians[0], medians[1]);

  if (is_anomaly (flops[0], flops[1], times[0], times[1], ratio)) {
    candidate.flops_score =   score (flops[0], flops[1]);
    candidate.isAnomaly = true;
  } else {
    candidate.flops_score = - score (flops[0], flops[1]);
    candidate.isAnomaly = false;
  }
}

/**
 * Looks for anomalies and returns them within a vector.
 * 
 * @param dims        Vector with the problem's dimensions.
 * @param times       2D vector with all the algorithms execution times.
 * @param algs        Vector with all the algorithms involved.
 * @param min_margin  Minimum difference between median values to consider a
 *                    point as an anomaly.
 * @param ratio       The percentage of total executions for which the
 *                    algorithm with more FLOPs must be faster.
 * @param n_threads   Number of threads used for computation.
 * @return            The anomalies found in the form of a vector.
 */
std::vector<Anomaly> search_anomaly (std::vector<int> dims, std::vector<std::vector<double>> times, 
    std::vector<int> algs, const double min_margin, const double ratio, const int n_threads) {
  std::vector<Anomaly> anomalies_found;
  Anomaly aux_anomaly;

  std::vector<unsigned long long int> flops;
  std::vector<double> medians;
  double time_score;

  unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (unsigned p = 0; p < algs.size(); p++) {
    flops.push_back (mc4::MC4_flops(dims, algs[p]));
    medians.push_back (medianVector(times[p]));
    shuffle (times[p].begin(), times[p].end(), std::default_random_engine(seed));
  }

  for (unsigned i = 0; i < algs.size(); i++) {
    for (unsigned j = i + 1; j < algs.size(); j++) {
      time_score = score<double> (medians[i], medians[j]);

      if ((time_score >= min_margin) && is_anomaly(flops[i], flops[j], times[i], times[j], ratio)) {
        aux_anomaly.dims = dims;
        aux_anomaly.algs = {int(i), int(j)};
        aux_anomaly.n_threads = n_threads;
        aux_anomaly.flops_score = score<unsigned long long int> (flops[i], flops[j]);
        aux_anomaly.time_score = time_score;
        aux_anomaly.isAnomaly = true;
        anomalies_found.push_back (aux_anomaly);
      }
    }
  }
  return anomalies_found;
}

std::vector<Anomaly> search_anomaly_MCX (const std::vector<int>& dims, 
    const std::vector<unsigned long>& flops, std::vector<std::vector<double>>& times, 
    const std::vector<int>& algs, const double min_margin, 
    const double ratio, const int n_threads) {

  std::vector<Anomaly> anomalies_found;
  Anomaly aux_anomaly;

  std::vector<double> medians;
  double time_score;

  unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (unsigned p = 0; p < algs.size(); p++) {
    medians.push_back (medianVector(times[p]));
    shuffle (times[p].begin(), times[p].end(), std::default_random_engine(seed));
  }

  for (unsigned i = 0; i < algs.size(); i++) {
    for (unsigned j = i + 1; j < algs.size(); j++) {
      time_score = score<double> (medians[i], medians[j]);

      if ((time_score >= min_margin) && is_anomaly(flops[i], flops[j], times[i], times[j], ratio)) {
        aux_anomaly.dims = dims;
        aux_anomaly.algs = {int(i), int(j)};
        aux_anomaly.n_threads = n_threads;
        aux_anomaly.flops_score = score<unsigned long> (flops[i], flops[j]);
        std::cout << "FLOPs score: " << aux_anomaly.flops_score << std::endl;
        aux_anomaly.time_score = time_score;
        aux_anomaly.isAnomaly = true;
        anomalies_found.push_back (aux_anomaly);
      }
    }
  }
  return anomalies_found;
}

/**
 * Checks whether there is an anomaly between one pair of algorithms.
 *
 * @param flops_a     The #FLOPs for the first algorithm.
 * @param flops_b     The #FLOPs for the second algorithm.
 * @param times_a     The execution times for the first algorithm.
 * @param times_b     The execution times for the second algorithm.
 * @param iterations  The number of times each algorithm is computed.
 * @param ratio       The percentage of total executions for which the
 *    algorithm with more FLOPs must be faster.
 * @return            Whether there is an anomaly for this pair.
 */
bool is_anomaly (const unsigned long long int flops_a, const unsigned long long int flops_b,
  const std::vector<double> times_a, const std::vector<double> times_b, const double ratio){
  int n_anomalies = 0;

  if (flops_a < flops_b) {
    for (unsigned i = 0; i < times_a.size(); i++){
      if (times_b[i] < times_a[i])
        n_anomalies++;
    }
  } else if (flops_b < flops_a){
    for (unsigned i = 0; i < times_b.size(); i++){
      if (times_a[i] < times_b[i])
        n_anomalies++;
    }
  }
  return n_anomalies >= int(times_a.size() * ratio);
}

/**
 * Explores a limited hyperspace around a hit (anomaly).
 *  
 * @param hit         The hit around which the region is determined.
 * @param iterations  The number of iterations for each algorithm.
 * @param span        The maximum modification in each dimension (greater/smaller).
 * @param jump        The difference between contiguous points in a given dimension.
 * @param ratio       Relative number of times an alg must be faster than the other one.
 * @param ofile       The output file where results are printed.
 */
std::deque<Anomaly> gridExploration (Anomaly &hit, const int iterations, 
    const int span, const int jump, const double ratio) {

  std::deque<Anomaly> anomalies_found;

  int ticks = int (span / jump) * 2 + 1;

  // helper variable that will be reused for different points in the search space.
  Anomaly candidate;
  candidate.algs = hit.algs;
  candidate.n_threads = hit.n_threads;

  std::vector<int> base_dims;
  for (auto x : hit.dims)
    base_dims.push_back (x - int(span / jump) * jump);
  candidate.dims = base_dims;

  std::vector<std::vector<double>> times;
  for (int d0 = 0; d0 < ticks; d0++) {
    candidate.dims[0] = base_dims[0] + d0 * jump;
    for (int d1 = 0; d1 < ticks; d1++) {
      candidate.dims[1] = base_dims[1] + d1 * jump;
      for (int d2 = 0; d2 < ticks; d2++) {
        candidate.dims[2] = base_dims[2] + d2 * jump;
        for (int d3 = 0; d3 < ticks; d3++) {
          candidate.dims[3] = base_dims[3] + d3 * jump;
          for (int d4 = 0; d4 < ticks; d4++) {
            candidate.dims[4] = base_dims[4] + d4 * jump;
            std::cout << "About to compute {";
            for (auto &x : candidate.dims)
              std::cout << x << ",";
            std::cout << "}" << '\n';
            times = mc4::MC4_execute_par (candidate.dims, candidate.algs, iterations, candidate.n_threads);
            search_anomaly_pair(candidate, times, ratio);
            anomalies_found.push_back(candidate);
          }
        }
      }
    }
  }
  return anomalies_found;
}

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
    const int iterations, const int jump, const double min_margin, const unsigned max_points) {

  exploratorySpace space;
  addNeighbours (space, hit, jump);

  // helper vars
  std::string key;
  dVector2D times;
  std::vector<unsigned long> flops;
  Anomaly candidate;
  mcx::MCX chain (hit.dims);
  candidate.n_threads = hit.n_threads;


  while (!space.queue.empty() && space.checked.size() < max_points) {
    chain.setDims(space.queue.front());
    space.queue.pop_front();

    times = chain.executeAll(iterations, hit.n_threads);
    flops = chain.getFLOPs();

    candidate = analysePoint(times, flops, min_margin);
    candidate.dims = chain.getDims();
    candidate.n_threads = hit.n_threads;
    key = serialiseVector<int>(candidate.dims);
    space.checked[key] = candidate;


    // If it is an anomaly look at the vicinity and add them if not present already in the queue
    if (candidate.isAnomaly) 
      addNeighbours(space, candidate, jump);
    
  }
  return space.checked;
}

std::vector<std::vector<DataPoint>> arrowMCX(const Anomaly& hit, const int iterations, 
    const int jump, const double min_margin, const int max_out, 
    std::vector<lamb::Anomaly>& summary) {
  
  std::vector<std::vector<DataPoint>> results;
  DataPoint point;
  point.samples.resize(hit.dims.size() - 1);

  bool keepExploring = true;
  bool in = true;  // keeps track of whether we are out of the region.
  int count_out = 0; // counter with the number of consecutive non-anomalous points.
  int count_in = 0;  // counter with the number of jumps to the transition point.

  srand(time(NULL));
  int dim_id = static_cast<int>( static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 
  static_cast<double>(hit.dims.size()) );

  iVector1D dims = hit.dims;
  mcx::MCX chain(dims);
  dVector3D times;
  dVector2D algs_times (chain.getNumAlgs());
  std::vector<unsigned long> flops;
  std::vector<std::vector<unsigned long>> flopsInd;
  Anomaly anomaly;
  results.resize(chain.getNumAlgs());

  // ========================= print information =========================
  std::cout << ">> Initial dimensions: {";
  for (const auto& d : hit.dims)
    std::cout << d << ',';
  std::cout << "}\n";
  std::cout << ">> Exploring dimension " << dim_id << "..." << std::endl;
  // =====================================================================

  while (keepExploring) { // values in one direction
    std::cout << ">> Computing point: {";
    for (const auto& d : dims)
      std::cout << d << ',';
    std::cout << "}...\n";

    chain.setDims(dims);
    flops = chain.getFLOPs();

    times = chain.executeAllInd(iterations, hit.n_threads);
    for (int i = 0; i < chain.getNumAlgs(); ++i)
      algs_times[i] = times[i].back();

    anomaly = analysePoint(algs_times, flops, min_margin);
    anomaly.dims = dims;
    anomaly.n_threads = hit.n_threads;
    summary.push_back(anomaly);

    flopsInd = chain.getFLOPsInd();
    // Store information to return it later
    point.dims = dims;
    for (int i = 0; i < chain.getNumAlgs(); ++i) {
      for (int j = 0; j < times[i].size(); ++j) {
        point.samples[j] = lamb::medianVector(times[i][j]);
      }
      point.flops = flopsInd[i];
      results[i].push_back(point);
    }

    if (in) {
      if (anomaly.isAnomaly) {
        count_in = count_in + 1 + count_out;
        count_out = 0; 
      }
      else {
        ++count_out;
      }
    }
    else
      ++count_out;

    if (count_out >= max_out)
      in = false;

    if (!in && count_out >= count_in)
      keepExploring = false;

    dims[dim_id] -= jump;
    if (dims[dim_id] <= 0) keepExploring = false;
    std::cout << "\tcount_in:  " << count_in << std::endl;
    std::cout << "\tcount_out: " << count_out << std::endl;
  }

  std::cout << ">> GOING BACKWARDS NOW\n";
  keepExploring = true;
  in = true;
  count_out = 0;
  count_in = 0;
  
  dims = hit.dims;

  while (keepExploring) { // values in the other direction
  // PRINT VALUES AND COUNTERS 
    dims[dim_id] += jump;

    std::cout << ">> Computing point: {";
    for (const auto& d : dims)
      std::cout << d << ',';
    std::cout << "}...\n";

    chain.setDims(dims);
    flops = chain.getFLOPs();

    times = chain.executeAllInd(iterations, hit.n_threads);
    for (int i = 0; i < chain.getNumAlgs(); ++i)
      algs_times[i] = times[i].back();

    anomaly = analysePoint(algs_times, flops, min_margin);
    anomaly.dims = dims;
    anomaly.n_threads = hit.n_threads;
    summary.push_back(anomaly);

    flopsInd = chain.getFLOPsInd();
    // Store information to return it later
    point.dims = dims;
    for (int i = 0; i < chain.getNumAlgs(); ++i) {
      for (int j = 0; j < times[i].size(); ++j) {
        point.samples[j] = lamb::medianVector(times[i][j]);
      }
      point.flops = flopsInd[i];
      results[i].push_back(point);
    }

    if (in) {
      if (anomaly.isAnomaly) {
        count_in = count_in + 1 + count_out;
        count_out = 0;
      }
      else {
        ++count_out;
      }
    }
    else
      ++count_out;

    if (count_out >= max_out)
      in = false;

    if (!in && count_out >= count_in)
      keepExploring = false;
    
    std::cout << "\tcount_in:  " << count_in << std::endl;
    std::cout << "\tcount_out: " << count_out << std::endl;
  }
  return results;
}

std::unordered_map<std::string, Anomaly> iterativeExplorationAATB(const Anomaly &hit, 
    const int iterations, const int jump, const double min_margin, const unsigned max_points) {

  iVector1D dims;
  exploratorySpace space;
  addNeighbours (space, hit, jump);

  // helper vars
  std::string key;
  dVector2D times;
  std::vector<unsigned long> flops;
  Anomaly candidate;
  candidate.n_threads = hit.n_threads;


  while (!space.queue.empty() && space.checked.size() < max_points) {
    // chain.setDims(space.queue.front());
    dims = space.queue.front();
    space.queue.pop_front();

    times = executeAll(dims);
    flops = flopsAll(dims);

    candidate = analysePoint(times, flops, min_margin);
    candidate.dims = dims;
    candidate.n_threads = hit.n_threads;
    key = serialiseVector<int>(candidate.dims);
    space.checked[key] = candidate;


    // If it is an anomaly look at the vicinity and add them if not present already in the queue
    if (candidate.isAnomaly) 
      addNeighbours(space, candidate, jump);
  }
  return space.checked;
}

/**
 * Explores the space around a certain initial anomaly (hit) by modifying a single dimension
 * independently. This means that only one dimension is modified at a given step in the process.
 * Therefore, the result of modifying multiple dimensions is not explored, but individual changes.
 * 
 * @param hit         The hit around which the dimensions are individually explored.
 * @param iterations  The number of iterations for each algorithm.
 * @param span        The maximum change in each dimension (greater/smaller).
 * @param jump        The difference between contiguous points in a given dimension.
 * @param ratio       Relative number of times a alg must be faster than the other one.
 * @param ofile       The output file where results are printed.
 * @returns           A queue with the anomalies that have been found.
 */
std::deque<Anomaly> dimExploration (Anomaly &hit, const int iterations, 
    const int span, const int jump, const double ratio) {

  std::deque<Anomaly> anomalies_found;

  mkl_set_num_threads (hit.n_threads);
  unsigned ticks = int (span / jump) * 2 + 1;
  std::vector<std::vector<double>> times;
  Anomaly candidate = hit;

  for (unsigned d = 0; d < hit.dims.size(); d++) {
    candidate.dims = hit.dims;
    for (unsigned i = 0; i < ticks; i++) {
      // candidate.dims[d] = hit.dims[d] - int(span / jump) * jump + i * jump;
      candidate.dims[d] = hit.dims[d] + (i - int(span / jump)) * jump;
      if (candidate.dims[d] > 0){
        for (auto &x : candidate.dims)
          std::cout << x << ',';
        std::cout << '\n';

        times = mc4::MC4_execute_par (candidate.dims, candidate.algs, iterations, candidate.n_threads);
        search_anomaly_pair(candidate, times, ratio);
        anomalies_found.push_back(candidate);
      }
    }
  }
  return anomalies_found;
}

/**
 * Individually explores all dimensions of a given anomaly, checking all the neighbours that 
 * only differ in one single jump in a dimension. In doing so, we have to check whether the 
 * new points to add to the queue already exist in such queue or have already been checked.
 * 
 * @param space   Struct that contains the queue and list of points already checked.
 * @param a       Initial anomaly of which neighbours are explored.
 * @param jump    The difference between contiguous points in a given dimension.
 */
void addNeighbours (exploratorySpace &space, const Anomaly &a, const int jump) {
  std::vector<int> nearDims;
  std::string key;

  for (unsigned i = 0; i < a.dims.size(); i++) {
    nearDims = a.dims;

    if (a.dims[i] - jump > 0) {
      nearDims[i] = a.dims[i] - jump;
      key = serialiseVector (nearDims);
      if (std::find (space.queue.begin(), space.queue.end(), nearDims) == space.queue.end() &&
          space.checked.find (key) == space.checked.end())
        space.queue.push_back (nearDims);
    }

    nearDims[i] = a.dims[i] + jump;
    key = serialiseVector (nearDims);
    if (std::find (space.queue.begin(), space.queue.end(), nearDims) == space.queue.end() &&
        space.checked.find (key) == space.checked.end())
      space.queue.push_back (nearDims);
  }
}

/**
 * Adds the headers to an output file for anomalies. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score |
 *
 * @param ofile   Output file manager already opened.
 * @param ndim    The number of dimensions in the problem.
 */
void print_header_anomalies (std::ofstream &ofile, const int ndim) {
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  ofile << "n_threads,alg_i,alg_j,flops_score,time_score" << '\n';
}

/**
 * Adds a line to an output file/stream for anomalies. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score |
 *
 * @param ofile Output file manager already opened.
 * @param an    The anomaly to be printed in the file.
 */
void print_anomaly (std::ostream &ofile, const Anomaly& an) {
  for (auto &d : an.dims)
    ofile << d << ',';

  ofile << an.n_threads << ',';

  for (auto &p : an.algs)
    ofile << p << ',';

  ofile << an.flops_score << ',' << an.time_score << '\n';
}

/**
 * Prints the headers to an output validation file for anomalies. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score || old_time_score |
 *
 * @param ofile   Output file manager already opened.
 * @param ndim    The number of dimensions in the problem.
 */
void print_header_validation (std::ofstream &ofile, const int ndim) {
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  ofile << "n_threads,alg_i,alg_j,flops_score,time_score,old_time_score" << '\n';
}

/**
 * Prints a line in the output anomaly validation file. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score || old_time_score |
 *
 * @param ofile           Output file manager already opened.
 * @param an              The anomaly to be printed in the file.
 * @param old_time_score  The old time score validated.
 */
void print_validation (std::ofstream &ofile, const Anomaly& an, const float old_time_score) {
  for (auto &d : an.dims)
    ofile << d << ',';

  ofile << an.n_threads << ',';

  for (auto &p : an.algs)
    ofile << p << ',';

  ofile << an.flops_score << ',' << an.time_score << ',' << old_time_score << '\n';
}

/**
 * Adds the headers to an output file in validation phase. Format:
 * | ndim dims || alg || nsamples samples |
 *
 * @param ofile     The output file manager, which has been previously opened.
 * @param ndim      The number of problem dimensions.
 * @param nsamples  The number of samples that will be computed.
 */
void print_header_val_time (std::ofstream &ofile, const int ndim, const int nsamples){
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  ofile << "alg" << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << "Sample_" << i;

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

/**
 * Adds a line in the already opened ofile for validation phase. Format:
 * | dims || n_alg || samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param dims      The array with the problem dimensions.
 * @param ndim      The number of dimensions in the problem.
 * @param times     The array that contains the execution times (already computed).
 * @param nsamples  The number of samples to store in the output file.
 * @param alg   Value that indicates which algorithm is stored.
 */
void print_val_time (std::ofstream &ofile, const int dims[], const int ndim, 
    const double times[], const int nsamples, const int alg) {
  for (int i = 0; i < ndim; i++)
    ofile << dims[i] << ',';

  ofile << alg << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << std::fixed << std::setprecision(10) << times[i];

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

} // namespace lamb


