/**
 * @file anomaly_AATB.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Benchmark to find anomalies in the operation: X = A*trans(A)*B
 * @version 0.1
 * @date 2022-01-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "AATB.h"
#include "anomalies.h"
#include "common.h"

int main(int argc, char **argv)
{
  const int ndim = 3;
  int min_size, max_size, iterations, target_anomalies, n_threads, threshold;
  double min_margin, ratio;

  std::string root_dir, filename;
  std::ofstream ofile;

  if (argc != 6)
  {
    std::cout << "Execution: " << argv[0] << " min_size max_size target_anomalies root_dir filename\n";
    exit(-1);
  }
  else
  {
    min_size = atoi(argv[1]);
    max_size = atoi(argv[2]);
    target_anomalies = atoi(argv[3]);
    root_dir = argv[4];
    filename = argv[5];
  }

  iVector1D dims(ndim);
  dVector2D temp_times;
  std::vector<unsigned long> temp_flops;
  lamb::Anomaly anomaly;

  std::vector<std::vector<unsigned long>> flops;
  dVector3D times;
  std::vector<lamb::Anomaly> summary;

  aatb::AATB expression(1, 1, 1);

  auto t0 = std::chrono::high_resolution_clock::now();
  unsigned long long int computed_points = 0;

  srand(time(NULL));
  for (int found = 0; found < target_anomalies;)
  {
    for (auto &d : dims)
      d = int((double(rand()) / RAND_MAX) * (max_size - min_size)) + min_size;

    // if (lamb::computation_decision(dims, THRESHOLD)) {

    std::cout << '{';
    for (const auto &dim : dims)
      std::cout << dim << ',';
    std::cout << "}\n";

    expression.setDims(dims);

    temp_flops = expression.getFLOPs();
    temp_times = expression.executeAll(BENCH_REPS, N_THREADS);
    ++computed_points;

    anomaly = lamb::analysePoint(temp_times, temp_flops, MARGIN_AN);

    if (anomaly.isAnomaly)
    {
      anomaly.dims = dims;
      anomaly.n_threads = N_THREADS;
      summary.push_back(anomaly);

      times.push_back(temp_times);
      flops.push_back(temp_flops);

      ++found;
      std::cout << "We've found " << found << " so far!\n";
    }
    // }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  auto execution_time = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "We've computed " << computed_points << " points."
            << "We've consumed " << execution_time << " seconds.\n";

  // write results to files
  for (unsigned int alg = 0; alg < times[0].size(); ++alg)
  {
    ofile.open(root_dir + filename + "_alg" + std::to_string(alg) + std::string(".csv"));
    if (ofile.fail())
    {
      std::cerr << ">> ERROR: opening the result files for alg " << alg << std::endl;
      exit(-1);
    }

    lamb::printHeaderTime(ofile, ndim, BENCH_REPS, true);
    for (unsigned point = 0; point < times.size(); ++point)
      lamb::printTime(ofile, summary[point].dims, times[point][alg], flops[point][alg]);
    ofile.close();
  }

  // write the summary of anomalies
  ofile.open(root_dir + filename + std::string("_summary.csv"));
  lamb::print_header_anomalies(ofile, ndim);
  for (const auto &an : summary)
    lamb::print_anomaly(ofile, an);
  ofile.close();

  return 0;
}