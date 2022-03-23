#include <iostream>
#include <vector>

#include "MCX.h"
#include "common.h"

int main(int argc, char **argv) {
  int iterations, n_threads;

  if (argc != 3){
    std::cout << "Execution: " << argv[0] << " iterations n_threads\n";
    exit(-1);
  }
  else {
    iterations = atoi (argv[1]);
    n_threads  = atoi (argv[2]);
  }


  std::vector<int> dimensions {350, 225, 315, 1000, 400};

  mcx::MCX chain (dimensions);

  // std::vector<mcx::Matrix> input_matrices = chain.getInMat();
  // for (auto const &mat : input_matrices) {
  //   std::cout << "name: " << mat.name << std::endl;
  //   std::cout << "rows: " << mat.rows << std::endl;
  //   std::cout << "cols: " << mat.columns << std::endl;
  // }
  

  std::vector<unsigned long> flops = chain.getFLOPs();
  for (unsigned i = 0; i < flops.size(); i++)
    std::cout << "Algorithm " << i << " : " << flops[i] << " FLOPs\n";

  std::vector<std::vector<std::vector<Matrix*>>> result = chain.generateAlgorithms();

  for (unsigned i = 0; i < result.size(); i++) {
    std::cout << " >> Algorithm" << i << " :" << '\n';
    for (unsigned j = 0; j < result[i].size(); j++) {
      std::cout << "\tGEMM " << j << ": [";
      for (auto const &x : result[i][j])
        std::cout << x->name << " ";
      std::cout << "]" << std::endl;
    }
  }

  std::vector<double> median_values;
  std::vector<std::vector<double>> times = chain.executeAll(iterations, n_threads);

  for (unsigned alg_id = 0; alg_id < chain.getNumAlgs(); alg_id++) {
    median_values.push_back(lamb::medianVector<double>(times[alg_id]));
    std::cout << ">> Alg" << alg_id << " with " << flops[alg_id] << " FLOPs and a median time of: " << median_values[alg_id] << '\n';
  }


  
  return 0;
}