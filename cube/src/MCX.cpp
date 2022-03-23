#include "MCX.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "mkl.h"

#include "common.h"

const int ALIGN = 64;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

namespace mcx {

/**
 * @brief Default constructor.
 */ 
MCX::MCX () {}

/**
 * @brief Additional constructor with a set of dimensions.
 * 
 * This constructor uses the input set of dimensions to create empty input and intermediate 
 * matrices that are used to generate all the possible algorithms to solve that Matrix Chain 
 * and the corresponding cost in terms of FLOPs.
 * 
 * @param dimensions input vector with the set of dimensions that represent the problem.
 */
MCX::MCX (const iVector1D& dimensions) {
  dims = dimensions;
  genEmptyInput();
  genEmptyInter();
  algorithms = generateAlgorithms();
  computeFLOPs();
}

/**
 * @brief Default destructor.
 */
MCX::~MCX() {}

/**
 * @brief Sets the set of dimensions of the problem to be the one given as an argument.
 * 
 * Depending on the current size of the object, this function may perform the same operations
 * the additional constructor does (if the object is empty) or just resize the input matrices 
 * and recompute the #FLOPs of each algorithm.
 * 
 * @param dimensions input vector with the set of dimensions that represent the new problem.
 */
void MCX::setDims (const iVector1D& dimensions) {
  if (dims.empty()) {
    dims = dimensions;
    genEmptyInput();
    genEmptyInter();
    algorithms = generateAlgorithms();
    computeFLOPs();
  }
  else if (dims.size() == dimensions.size()) {
    dims = dimensions;
    resizeInput();
    computeFLOPs();
  }
}

/**
 * @brief Generates all the algoritms that solve the current Matrix Chain problem.
 * 
 * This function generates all the algorithms that solve the Matrix Chain problem of 
 * a certain size, given by the number of matrices previously created. This function 
 * works with pointers to these matrices, making a given algorithm to actually be
 * a set of vectors with pointers to matrices. The first two pointers are the input
 * arguments to GEMM and the third one is where the result is stored. This function uses
 * a recursive function to generate all the possible algorithms.
 * 
 * @return A vector with all the algorithms where each algorithm is a 2D vector of pointers
 * to Matrix structs.
 */
std::vector<std::vector<std::vector<Matrix*>>> MCX::generateAlgorithms() {
  std::vector<Matrix*> expression_ptr;
  std::vector<Matrix*> interm_ptr;

  for (unsigned i = 0; i < input_matrices.size(); i++) 
    expression_ptr.push_back(&input_matrices[i]);

  for (unsigned i = 0; i < inter_matrices.size(); i++)
    interm_ptr.push_back(&inter_matrices[i]);

  std::vector<std::vector<std::vector<Matrix*>>> result = recursiveGen(expression_ptr, 
    interm_ptr, 0);

  for (auto &alg : result)
    std::reverse(alg.begin(), alg.end());
  
  return std::move(result);
}

/**
 * @brief Recursive function that, for a set of input matrices and intermediate matrices,
 * generates all the algorithms in the form of pointers to matrices.
 * 
 * This recursive function generates all the possible algorithms that solve a given 
 * Matrix Chain problem. The inputs are vectors of pointers to Matrix structs and the 
 * level of depth within the tree-like structure. 
 * 
 * @param expression     Vector of pointers to matrices where the actual expression of which a 
 *    solution must be found is represented.
 * @param inter_matrices Vector of pointers to the intermediate matrices.
 * @param depth          Level of depth within the tree-like structure.
 */
std::vector<std::vector<std::vector<Matrix*>>> MCX::recursiveGen (std::vector<Matrix*> expression,
    std::vector<Matrix*> inter_matrices, const int depth) {
  if (expression.size() == 2){
    std::vector<std::vector<std::vector<Matrix*>>> gemm = {{expression}};
    gemm[0][0].push_back(inter_matrices[depth]);
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
      gemm_matrices.push_back(inter_matrices[depth]);

      sub_expression.erase(sub_expression.begin() + i);
      sub_expression[i] = inter_matrices[depth];

      auto temp_result = recursiveGen(sub_expression, inter_matrices, depth + 1);

      for (auto &x : temp_result)
        x.push_back(gemm_matrices);
      
      result.insert(result.end(), temp_result.begin(), temp_result.end());
    }
    return result;
  }
}

/**
 * @brief Computes the cost in terms of FLOPs for each algorithm.
 */
void MCX::computeFLOPs() {
  FLOPs.clear();
  unsigned long flops_alg = 0;

  for (auto const &alg : algorithms) {
    flops_alg = 0;
    for (auto const &op : alg) {
      flops_alg += uint64_t(op[0]->rows) * uint64_t(op[0]->columns) * uint64_t(op[1]->columns);
      op[2]->rows = op[0]->rows;
      op[2]->columns = op[1]->columns;
    }

    FLOPs.push_back(2 * flops_alg);
  }
}

/**
 * @brief Returns the #FLOPs for all the previously generated algorithms.
 * 
 * @return A vector that contains the number of FLOPs for each algorithm.
 */
std::vector<unsigned long> MCX::getFLOPs() {
  return FLOPs;
}

/**
 * @brief Returns the #FLOPs for all the algorithms and each operation. Returns a 2D vector
 * where each 1D vector corresponds to an algorithm. Inside each 1D vector there is an element
 * for each operation and, at the end, another element for the total number of FLOPs for that 
 * algorithm.
 * 
 * @return 2D vector where each element corresponds to an algorithm.
 */
std::vector<std::vector<unsigned long>> MCX::getFLOPsInd() {
  std::vector<std::vector<unsigned long>> flops_all;
  std::vector<unsigned long> flops_alg;

  for (int i = 0; i < algorithms.size(); ++i) {
    flops_alg.clear();
    for (auto const& op : algorithms[i]) {
      flops_alg.push_back(uint64_t(2) * uint64_t(op[0]->rows) * uint64_t(op[0]->columns) * 
                          uint64_t(op[1]->columns));
      op[2]->rows = op[0]->rows;
      op[2]->columns = op[1]->columns;
    }
    flops_alg.push_back(FLOPs[i]);
    flops_all.push_back(flops_alg);
  }
  return flops_all;
}

/**
 * @brief Executes a certain algorithm.
 * 
 * This function executes an algorithm based on its ID as many times as specified by
 * the iterations variable and using as many threads as the n_threads variable indicates. 
 * The main purpose of this function is to measure the execution time the specified algorithm
 * takes to solve the Matrix Chain. The input matrices are supposed to have been allocated 
 * beforehand, whereas the intermediate matrices are allocated within this function because 
 * their sizes depend on the actual algorithm to be executed. 
 * 
 * @param alg_id Unsigned that identifies the algorithm to be executed.
 * @param iterations Integer that specifies the number of times the algorithm is executed.
 * @param n_theads Integer with the number of threads to be used.
 * 
 * @return A vector with the execution times for each iteration.
 */
dVector1D MCX::execute(const unsigned alg_id, const int iterations, const int n_threads) {
  dVector1D times;
  auto alg = algorithms[alg_id];
  allocInter(alg);
  char trans = 'N';
  double one = 1.0;
  
  mkl_set_dynamic(false);
  mkl_set_num_threads(n_threads);

  for (int it = 0; it < iterations; it++){
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (auto const &op : alg) {
      dgemm_(&trans, &trans, &op[0]->rows, &op[1]->columns, &op[0]->columns, 
          &one, op[0]->data, &op[0]->rows, 
                op[1]->data, &op[1]->rows, 
          &one, op[2]->data, &op[2]->rows);
    }
    auto end = std::chrono::high_resolution_clock::now();
    times.push_back (std::chrono::duration<double>(end-start).count());
  }
  freeMatrices(inter_matrices);
  return times;
}

/**
 * @brief Executes a certain algorithm and measures each GEMM's execution time. The 2D output vector
 * has a vector for each operation and another one for the entire execution time. These 1D vectors
 * contain repetitions of the same operation.
 * 
 * @param alg_id Unsigned that identifies the algorithm to be executed.
 * @param iterations Integer that specifies how many times the algorithm is executed.
 * @param n_threads Integer with the number of threads to be used.
 *
 * @return dVector2D with the execution time of all kernels - total execution in the element.
 */
dVector2D MCX::executeInd(const unsigned alg_id, const int iterations, const int n_threads) {
  auto alg = algorithms[alg_id];
  allocInter(alg);
  char trans = 'N';
  double one = 1.0;
  
  dVector2D times;
  times.resize(alg.size() + 1); // we reserve n+1 1D vectors, where n is the number of operations
                                 // in the algorithm.
  for (auto& v : times)
    v.reserve(iterations); // for each of the 1D vectors we reserve as many elements as iterations
                           // will be computed. This is done so the vectors are not resized.

  mkl_set_dynamic(false);
  mkl_set_num_threads(n_threads);

  for (int it = 0; it < iterations; it++) {
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);
    lamb::cacheFlush(n_threads);
    int op_id = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (auto const &op : alg) {
      auto before = std::chrono::high_resolution_clock::now();
      dgemm_(&trans, &trans, &op[0]->rows, &op[1]->columns, &op[0]->columns, 
          &one, op[0]->data, &op[0]->rows, 
                op[1]->data, &op[1]->rows, 
          &one, op[2]->data, &op[2]->rows);
      auto after = std::chrono::high_resolution_clock::now();
      times[op_id].push_back(std::chrono::duration<double>(after-before).count());
      ++op_id;
    }
    auto end = std::chrono::high_resolution_clock::now();
    times[alg.size()].push_back(std::chrono::duration<double>(end-start).count());
  }
  freeMatrices(inter_matrices);
  return times;
}

/**
 * @brief Executes a set of algorithms.
 * 
 * The main purpose of this function is to measure execution times of the algorithms 
 * specified in the alg_ids variable. The input matrices are allocated within this 
 * function, in such a way that these matrices are only allocated once. As expected,
 * these matrices are freed at the end of the function. This function uses its
 * homonym to execute each of the algorithms.
 * 
 * @param alg_ids Vector with the set of algorithms IDs to execute.
 * @param iterations Integer that specifies the number of times the algorithm is executed.
 * @param n_theads Integer with the number of threads to be used.
 * 
 * @return A 2D vector with all the execution times for the specified algorithms.
 */
dVector2D MCX::execute(const std::vector<unsigned>& alg_ids, 
    const int iterations, const int n_threads) {
  allocInput();
  dVector2D times;

  for (auto const &alg_id : alg_ids)
    times.push_back(execute(alg_id, iterations, n_threads));
  
  freeMatrices(input_matrices);
  return times;
}

/**
 * @brief Executes all the algorithms that have been generated.
 * 
 * The purpose is to measure execution times of all the algorithms that have been generated
 * for the Matrix Chain problem. The input matrices are allocated within this 
 * function, in such a way that these matrices are only allocated once. As expected,
 * these matrices are freed at the end of the function. This function uses the 'execute'
 * function to execute each of the algorithms.
 * 
 * @param iterations Integer that specifies the number of times the algorithm is executed.
 * @param n_theads Integer with the number of threads to be used.
 * @param individual bool used to decide whether each GEMM must be timed on its own.
 * 
 * @return A 2D vector with all the execution times for the algorithms.
 */
dVector2D MCX::executeAll(const int iterations, const int n_threads) {
  dVector2D times;
  allocInput();

  for (unsigned alg_id = 0; alg_id < algorithms.size(); ++alg_id)
    times.push_back(execute(alg_id, iterations, n_threads));
  
  freeMatrices(input_matrices);

  return times;
}

dVector3D MCX::executeAllInd(const int iterations, const int n_threads) {
  dVector3D times;
  allocInput();

  for (unsigned alg_id = 0; alg_id < algorithms.size(); ++alg_id)
    times.push_back(executeInd(alg_id, iterations, n_threads));
  
  freeMatrices(input_matrices);

  return times;
}

/**
 * @brief returns the set of dimensions.
 */
iVector1D MCX::getDims() const {
  return dims;
}

/**
 * @brief returns the number of algorithms that have been generated.
 */
unsigned MCX::getNumAlgs() const {
  return algorithms.size();
}

/**
 * @brief Function that generates solutions in the form of sizes for GEMM operations.
 * 
 * @return All the algorithms in the form of vectors of vectors of int, where the 
 * sizes are specified.
 */
std::vector<std::vector<std::vector<int>>> MCX::genDims() {
  std::vector<int> __dims;
  for (unsigned i = 0; i < dims.size(); i++) __dims.push_back(i);

  std::vector<std::vector<std::vector<int>>> result = recursiveGeneration(__dims);
  
  for (auto &alg : result)
    std::reverse(alg.begin(), alg.end());
  return result;
}

/**
 * @brief Recursive function that generates all the algorithms in the form of set of dimensions.
 * 
 * @param dimensions The set of dimensions of which solutions must be generated.
 * 
 * @return All the algorithms in the form of vectors of vectors of int, where the 
 * sizes are specified.
 */
std::vector<std::vector<std::vector<int>>> MCX::recursiveGeneration(std::vector<int> dimensions) {
  if (dimensions.size() == 3) 
    // do sth to finish branch
    return std::vector<std::vector<std::vector<int>>> {{dimensions}};
  
  else {
    std::vector<int> gemm_sizes, aux;
    std::vector<std::vector<std::vector<int>>> result;

    for (unsigned i = 1; i < dimensions.size() - 1; i++) {
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

/**
 * @brief Generates input matrices.
 * 
 * This function takes into account the number of dimensions in the problem
 * and generates Matrix structs with names, dimensions and proper data allocation.
 */
void MCX::genInputMatrices() {
  Matrix mat;

  for (unsigned i = 0; i < (dims.size() - 1); i++) {
    mat.name = std::string("A") + std::to_string(i);

    mat.rows = dims[i];
    mat.columns = dims[i + 1];

    mat.data = (double*)mkl_malloc(dims[i] * dims[i + 1] * sizeof(double), ALIGN);
    for (int j = 0; j < dims[i] * dims[i + 1]; j++)
      mat.data[j] = drand48();

    // mat.isAllocated = true;
    input_matrices.push_back(mat);
  }
}

/**
 * @brief Changes the dimensions of the input matrices. 
 * 
 * This function is only used when the original set of dimensions is replaced by a new
 * set of dimensions.
 */
void MCX::resizeInput() {
  for (unsigned i = 0; i < (dims.size() - 1); i++) {
    input_matrices[i].rows = dims[i];
    input_matrices[i].columns = dims[i + 1];
  }
}

/**
 * @brief Generates empty input matrices.
 * 
 * These matrices have names, sizes and are placed in the corresponding vector. However, 
 * there is no memory allocation performed within this function.
 */
void MCX::genEmptyInput() {
  Matrix mat;
  for (unsigned i = 0; i < dims.size() - 1; i++) {
    mat.name = std::string("A") + std::to_string(i);
    mat.rows = dims[i];
    mat.columns = dims[i + 1];
    input_matrices.push_back(mat);
  }
}

/**
 * @brief Generates empty intermediate matrices.
 * 
 * These matrices have names and are placed in the corresponding vector. However, 
 * they do not have sizes nor allocated memory, since those depend on the actual
 * algorithm to be executed.
 */
void MCX::genEmptyInter() {
  Matrix mat;
  for (unsigned i = 0; i < dims.size() - 2; i++) {
    mat.name = std::string("M") + std::to_string(i);
    inter_matrices.push_back(mat);
  }
}

/**
 * @brief Allocates memory for the input matrices.
 * 
 * This function allocates memory for the input matrices and initialises them with
 * random values in the range [0,1).
 */
void MCX::allocInput() {
  for (auto &mat : input_matrices) {
    mat.data = (double*)mkl_malloc(mat.rows * mat.columns * sizeof(double), ALIGN);
    for (int j = 0; j < mat.rows * mat.columns; j++)
      mat.data[j] = drand48();
    // mat.isAllocated = true;
  }
}

/**
 * @brief Allocates memory for the intermediate matrices.
 * 
 * This function allocates memory for the intermediate matrices and initialises them 
 * with zeroes. The actual sizes of these matrices depend on the algorithm to be solved.
 * 
 * @param alg 2D vector with pointers to matrices -- an algorithm to solve the Matrix Chain.
 */
void MCX::allocInter(const std::vector<std::vector<Matrix*>>& alg) {
  for (unsigned i = 0; i < alg.size(); i++){
    inter_matrices[i].rows = alg[i][0]->rows;
    inter_matrices[i].columns = alg[i][1]->columns;

    inter_matrices[i].data = (double*)mkl_malloc(inter_matrices[i].rows *
                                                 inter_matrices[i].columns *
                                                 sizeof(double), ALIGN);
    for (int j = 0; j < inter_matrices[i].rows * inter_matrices[i].columns; j++)
      inter_matrices[i].data[j] = 0.0f;
    // inter_matrices[i].isAllocated = true;
  }
}

/**
 * @brief Frees the memory allocated to the matrices within the input vector.
 * 
 * @param matrices Vector with matrices of which the memory is freed.
 */ 
void MCX::freeMatrices(std::vector<Matrix>& matrices) {
  for (auto &mat : matrices) {
    mkl_free(mat.data);
    // mat.isAllocated = false;
  }
}

// @TODO: erase this function.
std::vector<Matrix> MCX::getInMat (){
  return input_matrices;
}


} // namespace mcx

