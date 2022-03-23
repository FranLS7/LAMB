#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>


std::vector<std::vector<std::vector<int>>> recursiveGeneration(std::vector<int> dimensions) {
  if (dimensions.size() == 3) 
    // do sth to finish branch
    return std::vector<std::vector<std::vector<int>>> {{dimensions}};
  
  else {
    std::vector<int> gemm_sizes, aux;
    std::vector<std::vector<std::vector<int>>> result;

    for (int i = 1; i < dimensions.size() - 1; i++) {
      aux = dimensions;
      gemm_sizes = std::vector<int> (&dimensions[i - 1], &dimensions[i + 2]);
      aux.erase(aux.begin() + i);

      auto temp_result = recursiveGeneration(aux);

      for (auto &x : temp_result) {
        x.push_back(gemm_sizes);
      }
      result.insert(result.end(), temp_result.begin(), temp_result.end());
    }
    return result;
  }
}

std::vector<std::vector<std::vector<int>>> generateAlgorithms(std::vector<int> dims) {
  std::vector<std::vector<std::vector<int>>> result = recursiveGeneration(dims);
  
  for (auto &alg : result)
    std::reverse(alg.begin(), alg.end());
  return result;
}

std::vector<unsigned long> getFLOPs(std::vector<std::vector<std::vector<int>>> parenthesisations) {
  std::vector<unsigned long> flops;
  unsigned long flops_gemm = 1, flops_parenth = 0;

  for (auto const &parenth : parenthesisations) {
    flops_parenth = 0;
    for (auto const &gemm : parenth) {
      flops_gemm = 1;
      for (auto const &dim : gemm) {
        flops_gemm *= uint64_t(dim);
      }
      flops_parenth += flops_gemm;
    }
    flops.push_back(2 * flops_parenth);
  }

  return flops;
}

int main (int argc, char **argv){

  int ndim;
  if (argc != 2) {
    std::cout << "Execution: " << argv[0] << " ndim" << '\n';
    exit(-1);
  }
  else {
    ndim = atoi (argv[1]);
  }
  std::vector<int> foo;
  for (int i = 0; i < ndim; i++)
    foo.push_back(i);
  std::cout << "The number of dimensions is: " << foo.size() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  auto result = generateAlgorithms(foo);
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration<double>(end - start).count();
  std::cout << result.size() << " algorithms generated in " << time << " seconds" << std::endl;

  auto flops = getFLOPs(result);

  // for (unsigned i = 0; i < result.size(); i++) {
  //   std::cout << ">> Algorithm" << i << " with " << flops[i] << " FLOPs: \n";
  //   for (unsigned j = 0; j < result[i].size(); j++) {
  //     std::cout << "\tGEMM " << j << ": ";
  //     for (auto const &x : result[i][j])
  //       std::cout << x << ", ";
  //     std::cout << std::endl;
  //   }
  // }


}