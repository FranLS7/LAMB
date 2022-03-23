#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "anomalies.h"
#include "common.h"
#include "MCX.h"

void print_header_winner (std::ofstream &ofile, const int ndim);

void print_winner (std::ofstream &ofile, std::vector<int> dims, const int winner_flops,
    const int winner_time);

void gridExploration2D(const std::vector<int> dims, const int iterations, const int n_threads,
    const int span, const int jump, std::ofstream &ofile);

void printHeaderWinner (std::ofstream &ofile, const int ndim);

void printWinner (std::ofstream &ofile, std::vector<int> dims, const int n_threads, 
    const int winner_flops, const int winner_time, const int isAnomaly, 
    const double flops_score, const double time_score);

void searchSimpleAnomaly(std::vector<unsigned long> flops, 
    std::vector<std::vector<double>> times, lamb::Anomaly& info);

void gridExplorationAllDims (const std::vector<int> dims, const int iterations, const int n_threads,
    const int span, const int jump, std::string result_dir);


int main (int argc, char **argv) {
  int ndim = 5;
  std::vector<int> initial_dims (ndim, 0);

  int iterations, n_threads, jump, span;

  std::string root_dir;
  // std::ofstream regionFile;

  if (argc != 11) {
    std::cout << "Execution: " << argv[0] << " d0 d1 d2 d3 d4 iterations n_threads jump span "
    "root_dir " << std::endl;
    exit(-1);
  }
  else {
    for (unsigned i = 0; i < initial_dims.size(); i++)
      initial_dims[i] = atoi(argv[i + 1]);
    iterations  = atoi (argv[6]);
    n_threads   = atoi (argv[7]);
    jump        = atoi (argv[8]);
    span        = atoi (argv[9]);
    root_dir.append (argv[10]);
  }
  auto start = std::chrono::high_resolution_clock::now();

  // regionFile.open(root_dir + std::string("region_10_1000.csv"));
  // if (regionFile.fail()) {
  //   std::cout << ">> ERROR: opening output file for region anomaly.\n";
  //   exit(-1);
  // }
  // print_header_winner(regionFile, ndim);

  // gridExploration2D(initial_dims, iterations, n_threads, span, jump, regionFile);
  gridExplorationAllDims(initial_dims, iterations, n_threads, span, jump, root_dir);

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());

  return 0;
}




void print_header_winner (std::ofstream &ofile, const int ndim) {
  for (int i = 0; i < ndim; i++) 
    ofile << 'd' << i << ',';
  
  ofile << "winner_flops,winner_time\n";
}

void printHeaderWinner (std::ofstream &ofile, const int ndim) {
  for (int i = 0; i < ndim; i++) 
    ofile << 'd' << i << ',';

  ofile << "n_threads,winner_flops,winner_time,isAnomaly,flops_score,time_score\n";
}

void print_winner (std::ofstream &ofile, std::vector<int> dims, const int winner_flops,
    const int winner_time) {
  for (auto const &d : dims)
    ofile << d << ',';

  ofile << winner_flops << ',' << winner_time << '\n';
}

void printWinner (std::ofstream &ofile, std::vector<int> dims, const int n_threads, 
    const int winner_flops, const int winner_time, const int isAnomaly, 
    const double flops_score, const double time_score) {
  for (auto const &d : dims)
    ofile << d << ',';

  ofile << n_threads << ',' << winner_flops << ',' << winner_time << ',' << isAnomaly
    << ',' << flops_score << ',' << time_score << '\n';
}

void gridExploration2D(const std::vector<int> dims, const int iterations, const int n_threads,
    const int span, const int jump, std::ofstream &ofile) {

  std::vector<std::vector<double>> times;
  std::vector<double> median_times;
  std::vector<unsigned long> flops;
  int ticks = int (span / jump) * 2 + 1;
  int winner_flops, winner_time;

  std::vector<int> base_dims;
  for (auto x : dims)
    base_dims.push_back(x - int(span / jump) * jump);

  
  mcx::MCX chain(dims);
  times = chain.executeAll(iterations, n_threads);
  median_times.resize(chain.getNumAlgs());

  for (unsigned i = 0; i < median_times.size(); i++)
    median_times[i] = lamb::medianVector<double>(times[i]);

  winner_flops = lamb::idxMinVector<unsigned long>(chain.getFLOPs());
  winner_time = lamb::idxMinVector(median_times);

  // print_winner(ofile, dims, winner_flops, winner_time);

  
  auto dims_expl = dims;

  for (int d3 = 0; d3 < ticks; d3++) {
    dims_expl[1] = base_dims[1] + d3 * jump;
    for (int d4 = 0; d4 < ticks; d4++) {
      dims_expl[4] = base_dims[4] + d4 * jump;
      std::cout << ">> {";
      for (const auto &d : dims_expl)
        std::cout << d << ',';
      std::cout << "}\n";
      chain.setDims(dims_expl);
      auto flops = chain.getFLOPs();
      // for (int i = 0; i < flops.size(); i++)
      //   std::cout << "Alg" << i << " -- " << flops[i] << " FLOPs\n";
      times = chain.executeAll(iterations, n_threads);

      for (unsigned i = 0; i < median_times.size(); i++)
        median_times[i] = lamb::medianVector<double>(times[i]);
      
      winner_flops = lamb::idxMinVector<unsigned long>(chain.getFLOPs());
      winner_time = lamb::idxMinVector(median_times);

      print_winner(ofile, dims_expl, winner_flops, winner_time);
    }
  }

}



void searchSimpleAnomaly(std::vector<unsigned long> flops, 
    std::vector<std::vector<double>> times, lamb::Anomaly& info) {
  
  std::vector<double> median_times;
  median_times.resize(flops.size());

  int winner_flops = lamb::idxMinVector<unsigned long>(flops);

  for (unsigned i = 0; i < median_times.size(); i++)
    median_times[i] = lamb::medianVector<double>(times[i]);

  int winner_time  = lamb::idxMinVector<double>(median_times);

  info.algs = {winner_flops, winner_time};


  if (winner_flops != winner_time) {
    info.isAnomaly = 1;
    // std::cout << "Fdfasdfasdfasdfasdfasdfa" << std::endl;
    // std::cout << "winner_flops: " << winner_flops << std::endl;
    // std::cout << "winner_time: " << winner_time << std::endl;
    // std::cout << median_times[winner_flops] << ' ' << median_times[winner_time] << std::endl;
    info.time_score = lamb::score(median_times[winner_flops], median_times[winner_time]);
    info.flops_score = lamb::score(flops[winner_flops], flops[winner_time]);
  }
  else {
    info.isAnomaly = 0;
    info.time_score = 0.0;
    info.flops_score = 0.0;
  }
}



void gridExplorationAllDims (const std::vector<int> dims, const int iterations, const int n_threads,
    const int span, const int jump, std::string result_dir) {
  std::ofstream ofile;
  std::vector<std::vector<double>> times;
  std::vector<unsigned long> flops;
  int ticks = int (span / jump) * 2 + 1;

  std::vector<int> base_dims;
  for (auto x : dims)
    base_dims.push_back(x - int(span / jump) * jump);

  mcx::MCX chain(dims);

  // Iterate over all the pairs of dimensions.
  for (unsigned i = 0; i < dims.size() - 1; i++) {
    for (unsigned j = i + 1; j < dims.size(); j++) {
      std::cout << "{" << i << j << "}\n";
      auto dims_expl = dims;

      ofile.open(result_dir + std::string("region_") + std::to_string(i) + std::string("_") + 
          std::to_string(j) + std::string(".csv"));
      if (ofile.fail()) {
        std::cout << ">> ERROR: opening output file for region " << i << ' ' << j << '\n';
        exit(-1);
      }
      printHeaderWinner(ofile, dims.size());

      for (int d_i = 0; d_i < ticks; d_i++) {
        dims_expl[i] = base_dims[i] + d_i * jump;
        for (int d_j = 0; d_j < ticks; d_j++) {
          dims_expl[j] = base_dims[j] + d_j * jump;

          std::cout << ">> {";
          for (const auto &d : dims_expl)
            std::cout << d << ',';
          std::cout << "}\n";

          lamb::Anomaly info;
          chain.setDims(dims_expl);
          auto flops = chain.getFLOPs();

          times = chain.executeAll(iterations, n_threads);
          searchSimpleAnomaly(flops, times, info);

          printWinner(ofile, dims_expl, n_threads, info.algs[0], info.algs[1], 
            info.isAnomaly, info.flops_score, info.time_score);
        }
      }
      ofile.close();
    }
  }
}


