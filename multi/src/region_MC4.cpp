#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
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

  std::vector <lamb::Anomaly> anomalies;

  int iterations, jump, span, max_points;
  double ratio;

  std::string root_dir, filename_an;
  std::ifstream anomalyFile;
  std::ofstream regionFile;

  // Helper vars
  std::string line;

  if (argc != 8) {
    std::cout << "Execution: " << argv[0] << " iterations ratio jump span max_points root_dir"
      " anomaly_file" << std::endl;
    exit (-1);
  } else {
    iterations  = atoi (argv[1]);
    ratio       = atof (argv[2]);
    jump        = atoi (argv[3]);
    span        = atoi (argv[4]);
    max_points  = atoi (argv[5]);
    root_dir.append (argv[6]);
    filename_an = root_dir + argv[7];
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
  lamb::initialiseBLAS();

  auto start = std::chrono::high_resolution_clock::now();

  lamb::Anomaly hit;
  hit.dims.resize (NDIM);
  hit.algs.resize (NPAR);

  // read first to get EOF at the end of the script
  std::getline (anomalyFile, line);

  while (!anomalyFile.eof()) {
    sscanf (line.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f", &hit.dims[0], &hit.dims[1],
    &hit.dims[2], &hit.dims[3], &hit.dims[4], &hit.n_threads,
    &hit.algs[0], &hit.algs[1],
    &hit.flops_score, &hit.time_score);

    anomalies.push_back (hit);
    lamb::print_anomaly(std::cout, hit);

    std::getline (anomalyFile, line);
  }

  std::cout << "anomalies.size: " << anomalies.size() << std::endl;
  anomalyFile.close();

  // helper to name output files
  int aux = 0;
  std::unordered_map<std::string, lamb::Anomaly> region;

  for (auto &hit : anomalies) {
    regionFile.open (root_dir + std::string("region_") + std::to_string(aux) + std::string(".csv"));
    if (regionFile.fail()){
      std::cout << ">> ERROR: opening output file for " << aux << "anomaly." << std::endl;
      exit(-1);
    }
    lamb::print_header_anomalies(regionFile, NDIM);
    // Choose type of exploration: 
    //  - gridExploration (hit, iterations, span, jump, ratio, regionFile);
    //  - iterativeExploration (hit, iterations, jump, ratio, regionFile);
    //  - dimExploration (hit, iterations, span, jump, ratio, regionFile);
    // region = lamb::iterativeExploration(hit, iterations, jump, ratio, max_points);
    // MODIFICATION ON THE METHOD --> FILE NOT DOING ANYTHING AS OF 14/3/22

    for (auto const &an : region) 
      lamb::print_anomaly(regionFile, an.second);

    regionFile.close();
    aux++;
    break; // Just to stop after exploring around the first hit. Should disappear.
  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());

  return 0;
}
