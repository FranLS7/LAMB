/**
 * @file arrows_AATB_no_cache.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief
 * @version 0.1
 * @date 2022-04-02
 *
 * @copyright Copyright (c) 2022
 *
 */

// COMPLETE THIS FILE AND THE CORRESPONDING FUNCTION IN ANOMALIES.CPP/.H
// REVIEW AATB CLASS TO ENSURE RESULTS WILL BE CORRECT

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "anomalies.h"
#include "common.h"

int main(int argc, char **argv)
{
  int ndim;
  int n_anomalies;
  double margin_anomaly;

  std::string root_dir, points_filename, out_template;
  std::ofstream ofile_algs, ofile_summary;
  std::ifstream ifile_points;

  if (argc != 7)
  {
    std::cerr << "Execution: " << argv[0] << " ndim n_anomalies margin_anomaly root_dir points_filename"
                                             " out_template\n";
    exit(-1);
  }
  else
  {
    ndim = atoi(argv[1]);
    n_anomalies = atoi(argv[2]);
    margin_anomaly = atof(argv[3]);
    root_dir = argv[4];
    points_filename = root_dir + argv[5];
    out_template = root_dir + argv[6];
  }

  std::deque<lamb::Anomaly> queue_points;
  std::vector<lamb::Anomaly> summary;

  for (unsigned anomaly_id = 0; anomaly_id < n_anomalies; ++anomaly_id)
  {
    for (unsigned dim_id = 0; dim_id < ndim; ++dim_id)
    {
      std::cout << ">> [anomaly,dim] : [" << anomaly_id << "," << dim_id << "]\n";

      std::string base_in = points_filename + "anomaly" + std::to_string(anomaly_id) + "_dim" +
                            std::to_string(dim_id);
      std::string base_out = out_template + "anomaly" + std::to_string(anomaly_id) + "_dim" +
                             std::to_string(dim_id);
      queue_points.clear();

      std::string line;
      // Open file with anomalies.
      ifile_points.open(base_in + "_summary.csv", std::ifstream::in);
      if (ifile_points.fail())
      {
        std::cout << "Error opening the file with points to evaluate for anomaly " << anomaly_id << " dim " << dim_id << std::endl;
        exit(-1);
      }
      std::getline(ifile_points, line); // to read the headers

      lamb::Anomaly hit;
      hit.dims.resize(ndim);
      hit.algs.resize(2);

      std::getline(ifile_points, line);

      while (!ifile_points.eof())
      {
        sscanf(line.c_str(), "%d,%d,%d,%d,%d,%d,%lf,%lf", &hit.dims[0], &hit.dims[1],
               &hit.dims[2], &hit.n_threads,
               &hit.algs[0], &hit.algs[1],
               &hit.flops_score, &hit.time_score);

        queue_points.push_back(hit);
        std::getline(ifile_points, line);
      }
      ifile_points.close();

      summary.clear();

      auto results = lamb::executeAATBNoCache(queue_points, BENCH_REPS, margin_anomaly, summary);

      for (int i = 0; i < results.size(); ++i)
      {
        ofile_algs.open(base_out + "_alg" + std::to_string(i) + ".csv");
        if (ofile_algs.fail())
        {
          std::cerr << "Error opening the output file " << i << std::endl;
          exit(-1);
        }
        lamb::printHeaderTime(ofile_algs, ndim, 1, ndim - 1, true);

        for (const auto &point : results[i])
        {
          lamb::printTime(ofile_algs, point.dims, point.samples, point.flops);
        }

        ofile_algs.close();
      }

      ofile_summary.open(base_out + "_summary.csv");
      if (ofile_summary.fail())
      {
        std::cerr << "Error opening the summary file\n";
        exit(-1);
      }
      lamb::printHeaderAnomaly(ofile_summary, ndim);

      for (const auto &anomaly : summary)
        lamb::printAnomaly(ofile_summary, anomaly);

      ofile_summary.close();
    }
  }

  return 0;
}