#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mkl.h"

#include "anomalies.h"
#include "common.h"
#include "MC4.h"

// ----------------- CONSTANTS ----------------- //
const int NPAR = 6;
const int NDIM = 5;
// --------------------------------------------- //

int main (int argc, char **argv) {
  std::vector<int> dims (NDIM, 0);
  int min_size, max_size, iterations, n_anomalies, n_threads, threshold;
  double lo_margin, ratio;

  std::string root_dir;
  std::ofstream ofiles[NPAR];
  std::ofstream ofile_an;

  if (argc != 10){
    std::cout << "Execution: " << argv[0] << " min_size max_size iterations n_anomalies n_threads"
      " threshold lo_margin ratio root_dir" << '\n';
    exit(-1);
  } else {
    min_size    = atoi (argv[1]); // min size a dimension can take.
    max_size    = atoi (argv[2]); // max size a dimension can take.
    iterations  = atoi (argv[3]); // number of times each parenth is computed.
    n_anomalies = atoi (argv[4]); // number of anomalies to be found.
    n_threads   = atoi (argv[5]); // number of threads to use during computation.
    threshold   = atoi (argv[6]); // at least one dim must be less than this.
    lo_margin   = atof (argv[7]); // min score to be considered as anomaly.
    ratio       = atof (argv[8]); // percentage number of times a parenth
        // must be faster than the other one.
    root_dir.append (argv[9]); // base filename to create output files.
  }

  // ==================================================================
  //   - - - - - - - - - - Opening output files - - - - - - - - - - -
  // ==================================================================
  // TODO: modify how names are generated - make it dependant on features of the execution.
  for (unsigned i = 0; i < NPAR; i++){
    ofiles[i].open (root_dir + std::string("parenth_") + std::to_string(i) + 
        std::string(".csv"));
    if (ofiles[i].fail()) {
      std::cout << "Error opening the file for parenthesisation " << i << std::endl;
      exit(-1);
    }
    lamb::printHeaderTime(ofiles[i], NDIM, iterations);
  }

  ofile_an.open (root_dir + std::string("summary.csv"));
  if (ofile_an.fail()) {
    std::cout << "Error opening the file for anomalies summary" << std::endl;
    exit(-1);
  } else {
    lamb::print_header_anomalies (ofile_an, NDIM);
  }

  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  MKL_Set_Dynamic (0);
  MKL_Set_Num_Threads (n_threads);
  lamb::initialiseBLAS();

  std::vector<int> parenths;
  for (unsigned i = 0; i < NPAR; i++)
    parenths.push_back (i);
  std::vector<std::vector<double>> times; // vector with different vectors, each for the 
  // corresponding parenthesisation.
  std::vector<lamb::Anomaly> anomalies_found;

  auto start = std::chrono::high_resolution_clock::now();
  unsigned long long int computed_points = 0;

  // Main loop.
  for (int found = 0; found < n_anomalies; ){
    for (auto &d : dims) // generate random values for each dimension.
      d = int(drand48() * (max_size - min_size)) + min_size;

    if (lamb::computation_decision (dims, threshold)) {
      times = mc4::MC4_execute_par (dims, parenths, iterations, n_threads);
      anomalies_found = lamb::search_anomaly (dims, times, parenths, lo_margin, ratio, n_threads);
      computed_points++;

      std::cout << ">> " << computed_points << " points, " << found << " anomalies -- {";
      for (const auto d : dims) 
        std::cout << d << ", ";
      std::cout << "}\n";

      if (!anomalies_found.empty()) {
        found++;
        std::cout << "Anomaly! So far we have found " << found << '\n';
        for (int i = 0; i < NPAR; i++)
          lamb::printTime(ofiles[i], dims, times[i]);
        for (const auto &an : anomalies_found) 
          lamb::print_anomaly(ofile_an, an);
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());
  printf("TOTAL computed points: %llu\n", computed_points);

  // Close output files.
  for (int i = 0; i < NPAR; i++){
    ofiles[i].close();
  }
  ofile_an.close();

  return 0;
}

