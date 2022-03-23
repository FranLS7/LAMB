/**
 * @file volume_MCX.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief 
 * @version 0.1
 * @date 2022-03-14
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
  int jump;

  std::string root_dir, anomaly_filename, out_filename;
  std::ofstream ofile_an;
  std::ifstream anomaly_file;

  if (argc != 5) {
    std::cout << "Execution: " << argv[0] << " jump root_dir anomaly_filename out_filename\n";
    exit(-1);
  } else {
    jump = atoi(argv[1]);
    root_dir.append(argv[2]);   // base filename to create output files.
    anomaly_filename = root_dir + argv[3];
    out_filename = root_dir + argv[4];
  }

  // ofile_an.open (root_dir + std::string("volume.csv"));
  // if (ofile_an.fail()) {
  //   std::cout << "Error opening the file for anomalies volume" << std::endl;
  //   exit(-1);
  // } else {
  //   lamb::print_header_anomalies (ofile_an, ndim);
  // }

  std::string line;
  // Open file with anomalies.
  anomaly_file.open (anomaly_filename, std::ifstream::in);
  if (anomaly_file.fail()) {
    std::cout << "Error opening the file with anomalies to validate." << std::endl;
    exit (-1);
  }
  std::getline (anomaly_file, line); // to read the headers

  lamb::Anomaly hit;
  hit.dims.resize(ndim);
  hit.algs.resize(2);

  std::getline(anomaly_file, line);

  while (!anomaly_file.eof()) {
    sscanf (line.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%f,%f", &hit.dims[0], &hit.dims[1],
      &hit.dims[2], &hit.dims[3], &hit.dims[4], &hit.n_threads,
      &hit.algs[0], &hit.algs[1],
      &hit.flops_score, &hit.time_score);
    
      queue_anomalies.push_back(hit);
      std::getline(anomaly_file, line);
  }
  anomaly_file.close();

  std::cout << "Number of hits: " << queue_anomalies.size() << std::endl;


  std::unordered_map<std::string, lamb::Anomaly> volume;
  int explored_hit = 0;
  unsigned long long num_anomalies = 0;
  auto start = std::chrono::high_resolution_clock::now();

  while (!queue_anomalies.empty()) {
    ofile_an.open(out_filename + std::to_string(explored_hit) + ".csv", std::ofstream::out);
    if (ofile_an.fail()) {
      std::cerr << "Error opening the file with volume." << std::endl;
      exit(-1);
    }
    lamb::print_header_anomalies(ofile_an, ndim);

    hit = queue_anomalies.front();
    queue_anomalies.pop_front();

    std::cout << "Computing hit number " << explored_hit << "...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    volume = lamb::iterativeExplorationMCX(hit, BENCH_REPS, jump, MARGIN_AN, 1000);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\tTotal points checked for hit " << explored_hit << ": " 
              << volume.size() << std::endl;
    
    std::cout << "\tTime computing: " << std::chrono::duration<double>(t1 - t0).count() << std::endl;

    for (const auto& instance : volume) {
      if (instance.second.isAnomaly) {
        num_anomalies++;
        lamb::print_anomaly(ofile_an, instance.second);
      }
    }


    ofile_an.close();

    ++explored_hit;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Total anomalies found: " << num_anomalies << std::endl;
  std::cout << "Total Time consumed: " << std::chrono::duration<double>(end - start).count() 
            << std::endl;

  return 0;  
}