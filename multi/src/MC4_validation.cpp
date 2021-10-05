#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <deque>
#include <chrono>
#include <random>
#include <algorithm>

#include "mkl.h"

#include <common.h>
#include <MC4.h>

// ----------------- CONSTANTS ----------------- //
const int NPAR = 2;
const int NDIM = 5;

// ----------------- FUNCTIONS ----------------- //
extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void check_anomaly(anomaly &candidate, std::vector<std::vector<double>> &times,
  const int iterations, const double ratio);


int main (int argc, char **argv) {
  std::deque<anomaly> queue_anomalies;

  int iterations;
  double ratio;

  std::string root_dir, anomaly_filename, validation_filename;
  std::ifstream anomaly_file;
  std::ofstream validation_file;

  // Helper var
  std::string line;

  if (argc != 6) {
    std::cout << "Execution: " << argv[0] << " iterations ratio root_dir anomaly_filename"
      " validation_filename" << std::endl;
    exit (-1);
  } else {
    iterations = atoi (argv[1]);
    ratio      = atof (argv[2]);
    root_dir.append (argv[3]);
    anomaly_filename = root_dir + argv[4];
    validation_filename = root_dir + argv[5];
  }


  // Open file with anomalies to validate.
  anomaly_file.open (anomaly_filename, std::ifstream::in);
  if (anomaly_file.fail()) {
    std::cout << "Error opening the file with anomalies to validate." << std::endl;
    exit (-1);
  }
  std::getline (anomaly_file, line); // to read the headers

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
  std::getline (anomaly_file, line);

  while (!anomaly_file.eof()) {
    sscanf (line.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f", &hit.dims[0], &hit.dims[1],
    &hit.dims[2], &hit.dims[3], &hit.dims[4], &hit.n_threads,
    &hit.parenths[0], &hit.parenths[1],
    &hit.flops_score, &hit.time_score);

    queue_anomalies.push_back(hit);
    // add_line_anomalies (std::cout, hit);

    std::getline (anomaly_file, line);
  }

  std::cout << "anomalies.size: " << queue_anomalies.size() << std::endl;
  anomaly_file.close();

  // Open output file and add headers.
  validation_file.open (validation_filename, std::ofstream::out);
  if (validation_file.fail()) {
    std::cout << "Error opening the file for validation." << std::endl;
    exit (-1);
  }
  print_header_validation (validation_file, NDIM);

  // helper vars
  std::vector<std::vector<double>> times;
  anomaly an;
  float old_time_score;

  while (!queue_anomalies.empty()) {
    an = queue_anomalies.front();
    queue_anomalies.pop_front();
    for (auto &d : an.dims)
      std::cout << d << ',';
    std::cout << std::endl;
    old_time_score = an.time_score;

    mkl_set_num_threads (an.n_threads);

    times = MC4_execute_par (an.dims, an.parenths, iterations,
                             an.n_threads);
    check_anomaly (an, times, iterations, ratio);
    print_validation (validation_file, an, old_time_score);
  }
  validation_file.close();

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());

  return 0;
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
    std::shuffle (times[p].begin(), times[p].end(), std::default_random_engine (seed));
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