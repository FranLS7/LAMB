/**
 * @file arrow_AATB.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief 
 * @version 0.1
 * @date 2022-03-31
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
  int ndim = 3;
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
    sscanf (line.c_str(), "%d,%d,%d,%d,%d,%d,%lf,%lf", &hit.dims[0], &hit.dims[1],
      &hit.dims[2], &hit.n_threads,
      &hit.algs[0], &hit.algs[1],
      &hit.flops_score, &hit.time_score);
    
      queue_anomalies.push_back(hit);
      std::getline(ifile_anomalies, line);
  }
  ifile_anomalies.close();

  std::cout << "Number of hits: " << queue_anomalies.size() << std::endl;

  int max_dim = 1200;
  
  for (unsigned anomaly_id = 0; anomaly_id < queue_anomalies.size(); ++anomaly_id) {
    hit = queue_anomalies[anomaly_id];
    for (unsigned dim_id = 0; dim_id < hit.dims.size(); ++dim_id) {
      std::string base_out = out_template + "anomaly" + std::to_string(anomaly_id) + "_dim" + 
                         std::to_string(dim_id);
      std::vector<lamb::Anomaly> summary;
      auto results = lamb::arrowAATB(hit, BENCH_REPS, jump, margin_anomaly, 
                                    max_out, dim_id, max_dim, summary);
      
      for (int i = 0; i < results.size(); ++i) {
        ofile_algs.open (base_out + "_alg" + std::to_string(i) + ".csv");
        if (ofile_algs.fail()) {
          std::cerr << "Error opening the output file " << i << std::endl;
          exit(-1);
        }
        lamb::printHeaderTime(ofile_algs, ndim, 1, ndim - 1, true);

        for (const auto& point : results[i]) {
          lamb::printTime(ofile_algs, point.dims, point.samples, point.flops);
        }

        ofile_algs.close();

      }

      ofile_summary.open(base_out + "_summary.csv");
      if (ofile_summary.fail()) {
        std::cerr << "Error opening the summary file\n";
        exit(-1);
      }
      lamb::print_header_anomalies(ofile_summary, ndim);

      for (const auto& anomaly : summary)
        lamb::print_anomaly(ofile_summary, anomaly);
      
      ofile_summary.close();
    }
  }
  


  return 0;
}