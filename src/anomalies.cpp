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

#include "AATB.h"
#include "common.h"
#include "MCX.h"

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
  dVector1D medians;
  
  for (unsigned i = 0; i < times.size(); ++i) medians.push_back(medianVector<double>(times[i]));
  
  return analysePoint(medians, flops, min_margin);
}

Anomaly analysePoint(const dVector1D& median_times, std::vector<unsigned long>& flops, 
    const double min_margin) {
  Anomaly aux;
  aux.isAnomaly = false;
  aux.flops_score = 0.0;
  aux.time_score = 0.0;

  unsigned id_cheapest = getFastestCheap(median_times, flops);
  unsigned id_fastest = idxMinVector(median_times);

  aux.algs = {static_cast<int>(id_cheapest), static_cast<int>(id_cheapest)};

  if (id_cheapest == id_fastest) {
    aux.isAnomaly = false;
  }
  else if (median_times[id_fastest] < ((1 - min_margin) * median_times[id_cheapest])) {
    aux.isAnomaly = true;
    aux.algs = {static_cast<int>(id_fastest), static_cast<int>(id_cheapest)};
    aux.flops_score = score(flops[id_cheapest], flops[id_fastest]);
    aux.time_score = score(median_times[id_cheapest], median_times[id_fastest]);
  }
  return aux;
}

unsigned getFastestCheap (const dVector1D& median_times, std::vector<unsigned long>& flops) {
  unsigned long min = flops[0];
  unsigned id_fast_cheap = 0; // index of the fastest amongst the cheapest
  for (unsigned i = 0; i < flops.size(); ++i) {
    if (flops[i] < min) {
      min = flops[i];
      id_fast_cheap = i;
    }
    else if (flops[i] == min && median_times[i] < median_times[id_fast_cheap])
      id_fast_cheap = i;
  }
  return id_fast_cheap;
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
  // To be implemented for MCX.h and AATB.h
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
    const int iterations, const int jump, const double min_margin, const unsigned max_points,
    const int max_dim) {

  exploratorySpace space;
  addNeighbours (space, hit, jump, max_dim);

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
      addNeighbours(space, candidate, jump, max_dim);
    
  }
  return space.checked;
}

std::vector<std::vector<DataPoint>> arrowMCX(const Anomaly& hit, const int iterations, 
    const int jump, const double min_margin, const int max_out, const int dim_id, const int max_dim,
    std::vector<lamb::Anomaly>& summary) {
  
  std::vector<std::vector<DataPoint>> results;
  DataPoint point;
  point.samples.resize(hit.dims.size() - 1);

  bool keepExploring = true;
  bool in = true;  // keeps track of whether we are out of the region.
  int count_out = 0; // counter with the number of consecutive non-anomalous points.
  int count_in = 0;  // counter with the number of jumps to the transition point.

  // srand(time(NULL));
  // int dim_id = static_cast<int>( static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 
  // static_cast<double>(hit.dims.size()) );

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
    
    if (dims[dim_id] > max_dim) keepExploring = false;
    std::cout << "\tcount_in:  " << count_in << std::endl;
    std::cout << "\tcount_out: " << count_out << std::endl;
  }
  return results;
}

std::vector<std::vector<DataPoint>> executeMCXNoCache(const std::deque<lamb::Anomaly> &queue_points,
    const int iterations, const double min_margin, std::vector<lamb::Anomaly>& summary) {
  
  std::vector<std::vector<DataPoint>> results;
  
  DataPoint data_point;
  data_point.samples.resize(queue_points[0].dims.size() - 1);

  mcx::MCX chain (queue_points[0].dims);
  dVector3D times;
  dVector1D times_algs (chain.getNumAlgs());

  std::vector<unsigned long> flops;
  std::vector<std::vector<unsigned long>> flopsInd;

  Anomaly anomaly;

  results.resize(chain.getNumAlgs());

  for (const auto& point : queue_points) {
    chain.setDims(point.dims);

    flops = chain.getFLOPs();
    flopsInd = chain.getFLOPsInd();

    times = chain.executeAllIsolated(iterations, point.n_threads);

    data_point.dims = point.dims;
    for (unsigned i = 0; i < chain.getNumAlgs(); ++i) {
      double median = 0;
      for (unsigned j = 0; j < times[i].size() - 1; ++j) {
        data_point.samples[j] = lamb::medianVector(times[i][j]);
        median += data_point.samples[j];
      }
      times_algs[i] = median;
      data_point.samples[times[i].size() - 1] = median;
      data_point.flops = flopsInd[i];
      results[i].push_back(data_point);
    }

    anomaly = analysePoint(times_algs, flops, min_margin);
    anomaly.dims = point.dims;
    anomaly.n_threads = point.n_threads;
    summary.push_back(anomaly);
  }
  return results;
}


std::unordered_map<std::string, Anomaly> iterativeExplorationAATB(const Anomaly &hit, 
    const int iterations, const int jump, const double min_margin, const unsigned max_points, 
    const int max_dim) {

  iVector1D dims;
  exploratorySpace space;
  addNeighbours (space, hit, jump, max_dim);

  // helper vars
  std::string key;
  dVector2D times;
  std::vector<unsigned long> flops;
  Anomaly candidate;
  candidate.n_threads = hit.n_threads;
  aatb::AATB expression (hit.dims);

  while (!space.queue.empty() && space.checked.size() < max_points) {
    expression.setDims(space.queue.front());
    space.queue.pop_front();

    times = expression.executeAll(iterations, hit.n_threads);
    flops = expression.getFLOPs();

    candidate = analysePoint(times, flops, min_margin);
    candidate.dims = expression.getDims();
    candidate.n_threads = hit.n_threads;
    key = serialiseVector<int>(candidate.dims);
    space.checked[key] = candidate;

    // If it is an anomaly look at the vicinity and add them if not present already in the queue
    if (candidate.isAnomaly) 
      addNeighbours(space, candidate, jump, max_dim);

  }
  return space.checked;
}


std::vector<std::vector<DataPoint>> arrowAATB(const Anomaly& hit, const int iterations, 
    const int jump, const double min_margin, const int max_out, const int dim_id, const int max_dim,
    std::vector<lamb::Anomaly>& summary) {
  
  int diff = jump;
  std::vector<std::vector<DataPoint>> results;
  DataPoint point;
  point.samples.resize(hit.dims.size());

  bool forward = true;
  bool keepExploring = true;
  bool in = true;  // keeps track of whether we are out of the region.
  int count_out = 0; // counter with the number of consecutive non-anomalous points.
  int count_in = 0;  // counter with the number of jumps to the transition point.

  iVector1D dims = hit.dims;
  aatb::AATB expression (dims);
  dVector3D times;
  dVector1D algs_times (expression.getNumAlgs());
  std::vector<unsigned long> flops;
  std::vector<std::vector<unsigned long>> flopsInd;
  Anomaly anomaly;
  results.resize(5);

  // ========================= print information =========================
  std::cout << ">> Initial dimensions: {";
  for (const auto& d : hit.dims)
    std::cout << d << ',';
  std::cout << "}\n";
  std::cout << ">> Exploring dimension " << dim_id << "..." << std::endl;
  // =====================================================================

  while (keepExploring) { // values in one direction
    // std::cout << ">> Computing point: {";
    // for (const auto& d : dims)
    //   std::cout << d << ',';
    // std::cout << "}...\n";

    expression.setDims(dims);
    flops = expression.getFLOPs();

    times = expression.executeAllInd(iterations, hit.n_threads);
    for (int i = 0; i < expression.getNumAlgs(); ++i)
      algs_times[i] = lamb::medianVector<double>(times[i].back());

    anomaly = analysePoint(algs_times, flops, min_margin);
    anomaly.dims = dims;
    anomaly.n_threads = hit.n_threads;
    summary.push_back(anomaly);

    flopsInd = expression.getFLOPsInd();

    // Store information to return it later
    point.dims = dims;
    for (int i = 0; i < expression.getNumAlgs(); ++i) {
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

    dims[dim_id] -= diff;
    if (dims[dim_id] <= 0 || dims[dim_id] > max_dim) keepExploring = false;
    // std::cout << "\tcount_in:  " << count_in << std::endl;
    // std::cout << "\tcount_out: " << count_out << std::endl;

    if (!keepExploring && forward) {
      std::cout << ">> GOING BACKWARDS NOW\n";
      keepExploring = true;
      forward = false;
      in = true;
      diff *= -1;
      count_in = 0;
      count_out = 0;
      dims = hit.dims;
      dims[dim_id] -= diff;
    }
  }

  return results;
}

std::vector<std::vector<DataPoint>> executeAATBNoCache(const std::deque<lamb::Anomaly> &queue_points,
    const int iterations, const double min_margin, std::vector<lamb::Anomaly>& summary) {
  
  std::vector<std::vector<DataPoint>> results; // {algorithm, instance}
  
  DataPoint data_point;
  data_point.samples.resize(queue_points[0].dims.size());

  aatb::AATB expression(queue_points[0].dims);
  dVector3D times;
  dVector1D times_algs (expression.getNumAlgs());

  std::vector<unsigned long> flops;
  std::vector<std::vector<unsigned long>> flopsInd;

  Anomaly anomaly;

  results.resize(expression.getNumAlgs());

  for (const auto& point : queue_points) {
    expression.setDims(point.dims);

    flops = expression.getFLOPs();
    flopsInd = expression.getFLOPsInd();

    times = expression.executeAllIsolated(iterations, point.n_threads);

    data_point.dims = point.dims;
    for (unsigned i = 0; i < expression.getNumAlgs(); ++i) {
      double median = 0;
      for (unsigned j = 0; j < times[i].size() - 1; ++j) {
        data_point.samples[j] = lamb::medianVector(times[i][j]);
        median += data_point.samples[j];
      }
      times_algs[i] = times[i].back()[0];
      data_point.samples.back() = times[i].back()[0];
      data_point.flops = flopsInd[i];
      results[i].push_back(data_point);
    }

    anomaly = analysePoint(times_algs, flops, min_margin);
    anomaly.dims = point.dims;
    anomaly.n_threads = point.n_threads;
    summary.push_back(anomaly);
  }
  return results;
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
  // To be implemented for MCX.h and AATB.h
}

/**
 * Individually explores all dimensions of a given anomaly, checking all the neighbours that 
 * only differ in one single jump in a dimension. In doing so, we have to check whether the 
 * new points to add to the queue already exist in such queue or have already been checked.
 * 
 * @param space   Struct that contains the queue and list of points already checked.
 * @param a       Initial anomaly of which neighbours are explored.
 * @param jump    The difference between contiguous points in a given dimension.
 * @param max_dim Max size any dimension can take.
 */
void addNeighbours (exploratorySpace &space, const Anomaly &a, const int jump, const int max_dim) {
  std::vector<int> nearDims;
  std::string key;

  for (unsigned i = 0; i < a.dims.size(); i++) {
    nearDims = a.dims;

    if (a.dims[i] - jump > 0) {
      nearDims[i] = a.dims[i] - jump;
      key = serialiseVector (nearDims);
      if (std::find(space.queue.begin(), space.queue.end(), nearDims) == space.queue.end() &&
          space.checked.find(key) == space.checked.end())
        space.queue.push_back(nearDims);
    }

    if (a.dims[i] + jump < max_dim) {
      nearDims[i] = a.dims[i] + jump;
      key = serialiseVector (nearDims);
      if (std::find(space.queue.begin(), space.queue.end(), nearDims) == space.queue.end() &&
          space.checked.find(key) == space.checked.end())
        space.queue.push_back (nearDims);
    }
  }
}

/**
 * Adds the headers to an output file for anomalies. Format:
 * | dims || n_threads || alg_i || alg_j || flops_score || times_score |
 *
 * @param ofile   Output file manager already opened.
 * @param ndim    The number of dimensions in the problem.
 */
void printHeaderAnomaly (std::ofstream &ofile, const int ndim) {
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
void printAnomaly (std::ostream &ofile, const Anomaly& an) {
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
void printHeaderVal (std::ofstream &ofile, const int ndim) {
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
void printVal(std::ofstream &ofile, const Anomaly& an, const float old_time_score) {
  for (auto &d : an.dims)
    ofile << d << ',';

  ofile << an.n_threads << ',';

  for (auto &p : an.algs)
    ofile << p << ',';

  ofile << an.flops_score << ',' << an.time_score << ',' << old_time_score << '\n';
}

} // namespace lamb


