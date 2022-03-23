/**
 * @file arrow_MCX.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief 
 * @version 0.1
 * @date 2022-03-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "anomalies.h"
#include "common.h"

int main(int argc, char** argv) {
  std::deque<lamb::Anomaly> queue_anomalies;
  int ndim = 5;
  int jump, max_out;
  double margin_anomaly;

  std::string root_dir, anomaly_filename, out_template;
  std::ofstream ofile_algs, ofile_summary;
  std::ifstream ifile_anomalies;

  if (argc != 7) {
    std::cerr << "Execution: " << argv[0] << " jump max_out margin_anomaly root_dir anomaly_filename"
    " out_template\n";
    exit(-1);
  } else {
    jump = atoi(argv[1]);
    max_out = atoi(argv[2]);
    margin_anomaly = atof(argv[3]);
    root_dir.append(argv[4]);
    anomaly_filename = root_dir + argv[5];
    out_template = root_dir + argv[6];
  }

  std::string line;
  // Open file with anomalies.
  ifile_anomalies.open (anomaly_filename, std::ifstream::in);
  if (ifile_anomalies.fail()) {
    std::cout << "Error opening the file with anomalies to validate." << std::endl;
    exit (-1);
  }
  std::getline (ifile_anomalies, line); // to read the headers

  lamb::Anomaly hit;
  hit.dims.resize(ndim);
  hit.algs.resize(2);

  std::getline(ifile_anomalies, line);

  while (!ifile_anomalies.eof()) {
    sscanf (line.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f", &hit.dims[0], &hit.dims[1],
      &hit.dims[2], &hit.dims[3], &hit.dims[4], &hit.n_threads,
      &hit.algs[0], &hit.algs[1],
      &hit.flops_score, &hit.time_score);
    
      queue_anomalies.push_back(hit);
      std::getline(ifile_anomalies, line);
  }
  ifile_anomalies.close();

  std::cout << "Number of hits: " << queue_anomalies.size() << std::endl;

  srand(time(NULL));
  int anomaly_id = static_cast<int>( static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * 
  static_cast<double>(queue_anomalies.size()) );
  hit = queue_anomalies[anomaly_id];

  std::vector<lamb::Anomaly> summary;
  auto results = lamb::arrowMCX(hit, BENCH_REPS, jump, margin_anomaly, max_out, summary);

  for (int i = 0; i < results.size(); ++i) {
    ofile_algs.open (out_template + std::to_string(i) + ".csv");
    if (ofile_algs.fail()) {
      std::cerr << "Error opening the output file " << i << std::endl;
      exit(-1);
    }
    lamb::printHeaderTime(ofile_algs, ndim, 1, ndim - 2, true);

    for (const auto& point : results[i]) {
      lamb::printTime(ofile_algs, point.dims, point.samples, point.flops);
    }

    ofile_algs.close();

  }

  ofile_summary.open(out_template + "summary.csv");
  if (ofile_summary.fail()) {
    std::cerr << "Error opening the summary file\n";
    exit(-1);
  }
  lamb::print_header_anomalies(ofile_summary, ndim);

  for (const auto& anomaly : summary)
    lamb::print_anomaly(ofile_summary, anomaly);

  return 0;
}