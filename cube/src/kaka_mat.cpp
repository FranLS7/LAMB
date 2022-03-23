#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

struct Matrix{
  double* data;
  int rows, columns;
  std::string name;
};

std::vector<std::vector<std::vector<Matrix*>>> recursive (std::vector<Matrix*> expression, 
    std::vector<Matrix*> out_matrices, const int depth) {
  if (expression.size() == 2){
    std::vector<std::vector<std::vector<Matrix*>>> gemm = {{expression}};
    gemm[0][0].push_back(out_matrices[depth]);
    return gemm;
  }
  else {
    std::vector<Matrix*> sub_expression;
    std::vector<Matrix*> gemm_matrices;
    std::vector<std::vector<std::vector<Matrix*>>> result;

    for (unsigned i = 0; i < expression.size() - 1; i++){
      // re-initialise the sub-expression in order to modify it later by combining two matrices into one.
      sub_expression = expression;

      // GET the GEMM call
      gemm_matrices = std::vector<Matrix*> (&expression[i], &expression[i + 2]);
      gemm_matrices.push_back(out_matrices[depth]);

      sub_expression.erase(sub_expression.begin() + i);
      sub_expression[i] = out_matrices[depth];
      // make sub_expression[i] point to the corresponding interm_matrix.
      // get the corresponding matrix that'll be used in the gemm call.
      auto temp_result = recursive(sub_expression, out_matrices, depth + 1);

      for (auto &x : temp_result)
        x.push_back(gemm_matrices);
      
      result.insert(result.end(), temp_result.begin(), temp_result.end());
    }
    return result;
  }
}

std::vector<std::vector<std::vector<Matrix*>>> genAlgs (std::vector<Matrix> expression, 
    std::vector<Matrix> out_matrices) {

  std::vector<Matrix*> expression_ptr;
  std::vector<Matrix*> out_matrices_ptr;

  for (unsigned i = 0; i < expression.size(); i++)
    expression_ptr.push_back(&expression[i]);
  
  for (unsigned i = 0; i < out_matrices.size(); i++)
    out_matrices_ptr.push_back(&out_matrices[i]);

  std::vector<std::vector<std::vector<Matrix*>>> result = recursive(expression_ptr, 
    out_matrices_ptr, 0);

  for (auto &alg : result)
    std::reverse(alg.begin(), alg.end());
  return result;
}

int main(int argc, char **argv) {

  int ndim;

  if (argc != 2) {
    std::cout << "Execution: " << argv[0] << " ndim" << '\n';
    exit(-1);
  }
  else {
    ndim = atoi (argv[1]);
  }

  std::vector<int> dimensions;
  for (int i = 0; i < ndim; i++)
    dimensions.push_back(i);


  std::vector<Matrix> input_matrices;
  std::vector<Matrix> out_matrices;

  for (unsigned i = 0; i < dimensions.size() - 1; i++) {
    input_matrices.push_back(Matrix());
    input_matrices[i].rows = dimensions[i];
    input_matrices[i].columns = dimensions[i + 1];
    input_matrices[i].name = std::string("A") + std::to_string(i);
  }

  for (unsigned i = 0; i < dimensions.size() - 2; i++) {
    out_matrices.push_back(Matrix());
    out_matrices[i].name = std::string("M") + std::to_string(i);
  }


  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<std::vector<Matrix*>>> result = genAlgs(input_matrices, out_matrices);
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration<double>(end - start).count();
  std::cout << result.size() << " algorithms generated in " << time << " seconds" << std::endl;


  // for (unsigned i = 0; i < result.size(); i++) {
  //   std::cout << " >> Algorithm" << i << " :" << '\n';
  //   for (unsigned j = 0; j < result[i].size(); j++) {
  //     std::cout << "\tGEMM " << j << ": [";
  //     for (auto const &x : result[i][j])
  //       std::cout << x->name << " ";
  //     std::cout << "]" << std::endl;
  //   }
  // }


}