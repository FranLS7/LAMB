#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mkl.h"

#include "anomalies.h"
#include "common.h"
#include "MCX.h"


int main (int argc, char **argv) {
  int ndim;
  int min_size, max_size, n_anomalies;

  std::string root_dir;
  std::ofstream ofile_an;

  // std::vector<int> dims (NDIM, 0);

  if (argc != 6){
    std::cout << "Execution: " << argv[0] << " ndim min_size max_size n_anomalies root_dir\n";
    exit(-1);
  } else {
    ndim        = atoi (argv[1]); // number of dimensions;
    min_size    = atoi (argv[2]); // min size a dimension can take.
    max_size    = atoi (argv[3]); // max size a dimension can take.
    // iterations  = atoi (argv[4]); // number of times each parenth is computed.
    n_anomalies = atoi (argv[4]); // number of anomalies to be found.
    // n_threads   = atoi (argv[6]); // number of threads to use during computation.
    // threshold   = atoi (argv[7]); // at least one dim must be less than this.
    // min_margin  = atof (argv[8]); // min score to be considered an anomaly.
    // ratio       = atof (argv[9]); // percentage number of times a parenth
                                  // must be faster than the other one.
    root_dir.append (argv[5]);   // base filename to create output files.
  }

  std::vector<int> dims (ndim, 0);
  mcx::MCX chain (dims);
  unsigned n_algs = chain.getNumAlgs();
  std::ofstream ofiles[n_algs];

  std::vector<int> parenths;
  for (unsigned i = 0; i < n_algs; i++)
    parenths.push_back (i);

  // ==================================================================
  //   - - - - - - - - - - Opening output files - - - - - - - - - - -
  // ==================================================================
  // @TODO: modify how names are generated - make it dependant on features of the execution.
  for (unsigned i = 0; i < n_algs; i++){
    ofiles[i].open (root_dir + std::string("alg_") + std::to_string(i) + 
        std::string(".csv"));
    if (ofiles[i].fail()) {
      std::cout << "Error opening the file for parenthesisation " << i << std::endl;
      exit(-1);
    }
    lamb::printHeaderTime (ofiles[i], ndim, BENCH_REPS);
  }

  ofile_an.open (root_dir + std::string("summary.csv"));
  if (ofile_an.fail()) {
    std::cout << "Error opening the file for anomalies summary" << std::endl;
    exit(-1);
  } else {
    lamb::print_header_anomalies (ofile_an, ndim);
  }

  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  std::vector<unsigned long> flops;
  dVector2D times; // vector with different vectors, each for the 
                                          // corresponding parenthesisation.
  lamb::Anomaly anomaly;

  auto start = std::chrono::high_resolution_clock::now();
  unsigned long long int computed_points = 0;

  srand(time(NULL));
  // Main loop.
  for (int found = 0; found < n_anomalies; ){
    for (auto &d : dims) // generate random values for each dimension.
      d = int((double(rand())/RAND_MAX) * (max_size - min_size)) + min_size;


    if (lamb::computation_decision (dims, THRESHOLD)) {
      chain.setDims(dims);
      flops = chain.getFLOPs();
      times = chain.executeAll(BENCH_REPS, N_THREADS);

      // anomalies_found = lamb::search_anomaly_MCX(dims, flops, times, parenths, lo_margin, 
      //     ratio, n_threads);
      anomaly = lamb::analysePoint(times, flops, MARGIN_AN);
      computed_points++;
      std::cout << ">> " << computed_points << " points, " << found << " anomalies -- {";
      for (const auto d : dims) 
        std::cout << d << ", ";
      std::cout << "}\n";

      if (anomaly.isAnomaly) {
        found++;
        std::cout << "Anomaly! So far we have found " << found << '\n';
        for (int i = 0; i < n_algs; i++)
          lamb::printTime(ofiles[i], dims, times[i]);
        anomaly.dims = dims;
        anomaly.n_threads = N_THREADS;
        lamb::print_anomaly(ofile_an, anomaly);
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());
  printf("TOTAL computed points: %llu\n", computed_points);

  // Close output files.
  for (int i = 0; i < n_algs; i++){
    ofiles[i].close();
  }
  ofile_an.close();

  return 0;
}

