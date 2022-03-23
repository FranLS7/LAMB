#include <stdlib.h>

#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mkl.h"

#include "anomalies.h"
#include "common.h"
#include "MC4.h"

// ----------------- CONSTANTS ----------------- //
const int NPAR = 2;
const int NDIM = 5;
// --------------------------------------------- //

int main (int argc, char **argv) {
  std::deque<lamb::Anomaly> queue_anomalies;

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
  MKL_Set_Dynamic (0);
  lamb::initialiseMKL ();

  auto start = std::chrono::high_resolution_clock::now();

  lamb::Anomaly hit;
  hit.dims.resize (NDIM);
  hit.algs.resize (NPAR);

  // read first to get EOF at the end of the script
  std::getline (anomaly_file, line);

  while (!anomaly_file.eof()) {
    sscanf (line.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f", &hit.dims[0], &hit.dims[1],
    &hit.dims[2], &hit.dims[3], &hit.dims[4], &hit.n_threads,
    &hit.algs[0], &hit.algs[1],
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
  lamb::print_header_validation (validation_file, NDIM);

  // helper vars
  std::vector<std::vector<double>> times;
  lamb::Anomaly an;
  float old_time_score;

  while (!queue_anomalies.empty()) {
    an = queue_anomalies.front();
    queue_anomalies.pop_front();
    for (auto &d : an.dims)
      std::cout << d << ',';
    std::cout << std::endl;
    old_time_score = an.time_score;

    MKL_Set_Num_Threads (an.n_threads);

    times = mc4::MC4_execute_par (an.dims, an.algs, iterations,
                             an.n_threads);
    lamb::search_anomaly_pair(an, times, ratio);
    lamb::print_validation (validation_file, an, old_time_score);
  }
  validation_file.close();

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());

  return 0;
}
