#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <deque>
#include "mkl.h"

#include <cube.h>
#include <common.h>
#include <MC4.h>

using namespace std;

// ----------------- CONSTANTS ----------------- //
const int NPAR = 2;
const int NDIM = 5;


// ---------------- DATA TYPES ----------------- //
struct exploratorySpace {
  std::deque<std::vector<int>> queue;
  std::unordered_map<std::string, anomaly> checked;
};


// ----------------- FUNCTIONS ----------------- //
extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/**
 * Checks whether there is at least one dimension within the threshold.
 * This way we increase the possibilities of finding anomalies.
 *
 * @param dims      The array with the matrices dimensions.
 * @param ndim      The number of total dimensions in the problem.
 * @param threshold The max size for at least one dimension.
 * @return          True if at least one dimension is smaller than threshold.
 * False, otherwise.
 */
bool computation_decision (int dims[], const int ndim, const int threshold);

/**
 * Checks whether there is an anomaly in the computed results.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array containing all the execution times.
 * @param nparenth    The number of parenthesisations that have been computed.
 * @param iterations  The number of times each parenthesisation has been computed.
 * @param lo_margin   Minimum difference between median values to consider a
 * point as an anomaly.
 * @ratio             The percentage of total executions for which the
 * parenthesisation with more FLOPs must be faster.
 * @return            Whether there is an anomaly in the computed results.
 */
void check_anomaly(anomaly &candidate, std::vector<std::vector<double>> &times,
    const int iterations, const double ratio);

void gridExploration (anomaly &hit, const int iterations, const int span, const int jump,
    const double ratio, std::ofstream &ofile);

void iterativeExploration (anomaly &hit, const int iterations, const int jump, const double ratio,
    std::ofstream &ofile);

void dimExploration (anomaly &hit, const int iterations, const int span,
    const int jump, const double ratio, std::ofstream &ofile);

std::string serialiseVector (std::vector<int> v);
void addNeighbours (exploratorySpace &space, const anomaly &a, const int jump);


int main (int argc, char **argv) {

  std::vector <anomaly> anomalies;

  int iterations, jump, span;
  double ratio;

  string root_dir, filename_an;
  std::ifstream anomalyFile;
  std::ofstream regionFile;

  // Helper vars
  string line;

  if (argc != 7) {
    std::cout << "Execution: " << argv[0] << " iterations ratio jump span root_dir"
      " anomaly_file" << std::endl;
    exit (-1);
  } else {
    iterations  = atoi (argv[1]);
    ratio       = atof (argv[2]);
    jump        = atoi (argv[3]);
    span        = atoi (argv[4]);
    root_dir.append (argv[5]);
    filename_an = root_dir + argv[6];
  }

  // Open anomalies file
  anomalyFile.open (filename_an, std::ifstream::in);
  if (anomalyFile.fail()) {
    std::cout << "Error opening file with anomalies" << std::endl;
    exit (-1);
  }
  std::getline (anomalyFile, line); // to read the headers

  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  mkl_set_dynamic (0);
  initialise_mkl ();

  auto start = std::chrono::high_resolution_clock::now();

  anomaly hit;
  hit.dims.resize (NDIM);
  hit.parenths.resize (NPAR);

  // read first to get EOF at the end of the script
  std::getline (anomalyFile, line);

  while (!anomalyFile.eof()) {
    sscanf (line.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f", &hit.dims[0], &hit.dims[1],
    &hit.dims[2], &hit.dims[3], &hit.dims[4], &hit.n_threads,
    &hit.parenths[0], &hit.parenths[1],
    &hit.flops_score, &hit.time_score);

    anomalies.push_back (hit);
    add_line_anomalies (std::cout, hit);

    std::getline (anomalyFile, line);
  }

  std::cout << "anomalies.size: " << anomalies.size() << std::endl;
  anomalyFile.close();

  // helper to name output files
  int aux = 0;

  for (auto &hit : anomalies) {
    regionFile.open (root_dir + string("dims_") + to_string(aux) + string(".csv"));
    if (regionFile.fail()){
      std::cout << ">> ERROR: opening output file for " << aux << "anomaly." << std::endl;
      exit(-1);
    }
    add_headers_anomalies (regionFile, NDIM);
    // gridExploration (hit, iterations, span, jump, ratio, regionFile);
    // iterativeExploration (hit, iterations, jump, ratio, regionFile);
    dimExploration (hit, iterations, span, jump, ratio, regionFile);
    regionFile.close();
    aux++;
    break;
  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());

  return 0;
}

/**
 * Checks whether there is at least one dimension within the threshold.
 * This way we increase the possibilities of finding anomalies.
 *
 * @param dims      The array with the matrices dimensions.
 * @param ndim      The number of total dimensions in the problem.
 * @param threshold The max size for at least one dimension.
 * @return          True if at least one dimension is smaller than threshold.
 * False, otherwise.
 */
bool computation_decision (int dims[], const int ndim, const int threshold){
  bool compute = false;

  for (int i = 0; i < ndim; i++){
    if (dims[i] <= threshold)
      compute = true;
  }
  return compute;
}

/**
 * Checks whether there is an anomaly in the computed results.
 *
 * @param candidate   anomaly struct that might be an anomaly.
 * @param times       The array containing all the execution times.
 * @param iterations  The number of times each parenthesisation has been computed.
 * @param lo_margin   Minimum difference between median values to consider a
 * point as an anomaly.
 * @param ratio       The percentage of total executions for which the
 * parenthesisation with more FLOPs must be faster.
 * @return            Whether there is an anomaly in the computed results.
 */
void check_anomaly(anomaly &candidate, std::vector<std::vector<double>> &times,
  const int iterations, const double ratio){

  std::vector<unsigned long long int> flops;
  std::vector<double> medians;

  for (unsigned p = 0; p < candidate.parenths.size(); p++){
    flops.push_back (MC4_flops(candidate.dims, candidate.parenths[p]));
    medians.push_back (median_vector (times[p]));
  }

  // Shuffle the timings belonging to each parenthesisation
  unsigned seed;
  for (unsigned p = 0; p < candidate.parenths.size(); p++){
    seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    shuffle (times[p].begin(), times[p].end(), default_random_engine (seed));
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


// std::vector<anomaly> gridExploration (anomaly &hit, int ndim, int span, int jump){
void gridExploration (anomaly &hit, const int iterations, const int span, const int jump,
    const double ratio, std::ofstream &ofile){

  int ticks = int (span / jump) * 2 + 1;

  // helper variable that will be reused for different points in the search space.
  anomaly candidate;
  candidate.parenths = hit.parenths;
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
            std::cout << "}" << std::endl;
            times = MC4_execute_par (candidate.dims, candidate.parenths, iterations, candidate.n_threads);
            check_anomaly (candidate, times, iterations, ratio);
            add_line_anomalies (ofile, candidate);
          }
        }
      }
    }
  }
}

void iterativeExploration (anomaly &hit, const int iterations, const int jump, const double ratio,
    std::ofstream &ofile){

  exploratorySpace space;

  addNeighbours (space, hit, jump);

  // helper vars
  std::string key;
  std::vector<std::vector<double>> times;
  anomaly candidate;
  candidate.parenths = hit.parenths;
  candidate.n_threads = hit.n_threads;

  unsigned total_points = 5000;

  while (!space.queue.empty() && space.checked.size() < total_points) {
    // Execute first point in the queue
    std::cout << ">> #Checked: " << space.checked.size() << std::endl;
    std::cout << "\tQueue size: " << space.queue.size() << std::endl;
    candidate.dims = space.queue.front();
    space.queue.pop_front();

    times = MC4_execute_par (candidate.dims, candidate.parenths, iterations, candidate.n_threads);
    check_anomaly (candidate, times, iterations, ratio);
    key = serialiseVector (candidate.dims);
    space.checked[key] = candidate;

    // If it is an anomaly look at the vicinity and add them if not present already in the queue
    if (candidate.isAnomaly)
      addNeighbours(space, candidate, jump);
  }
  for (auto const &an : space.checked) {
    if (an.second.isAnomaly)
      add_line_anomalies (ofile, an.second);
  }
}

std::string serialiseVector (std::vector<int> v){
  std::string buffer;

  for (auto const &x : v){
    buffer.append (std::to_string(x) + "/");
  }
  return buffer;
}

void addNeighbours (exploratorySpace &space, const anomaly &a, const int jump) {
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


void dimExploration (anomaly &hit, const int iterations, const int span,
    const int jump, const double ratio, std::ofstream &ofile) {

  mkl_set_num_threads (hit.n_threads);
  unsigned ticks = int (span / jump) * 2 + 1;
  std::vector<std::vector<double>> times;
  anomaly candidate = hit;

  for (unsigned d = 0; d < hit.dims.size(); d++) {
    candidate.dims = hit.dims;
    for (unsigned i = 0; i < ticks; i++) {
      // candidate.dims[d] = hit.dims[d] - int(span / jump) * jump + i * jump;
      candidate.dims[d] = hit.dims[d] + (i - int(span / jump)) * jump;
      if (candidate.dims[d] > 0){
        for (auto &x : candidate.dims)
        std::cout << x << ',';
        std::cout << std::endl;

        times = MC4_execute_par (candidate.dims, candidate.parenths, iterations, candidate.n_threads);
        check_anomaly (candidate, times, iterations, ratio);
        print_anomaly (ofile, candidate);
      }
    }
  }
}
