#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include "mkl.h"

#include <cube.h>
#include <common.h>
#include <MC4.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

void exec_parenths(int parenths[], int nparenth, int dims[], double times[],
  const int iterations, const int n_measures, const int nthreads, std::ofstream &ofile);

int main (int argc, char **argv){

  int ndim = 5;
  int nparenth = 2;

  int iterations, nthreads;
  double *times;
  double lo_margin;

  string line;
  string input_file;// = "../timings/MC4_mt/";
  std::ifstream ifile;
  string output_file;// = "../timings/MC4_mt/";
  std::ofstream ofile;

  if (argc != 6){
    printf("Execution: %s iterations nthreads lo_margin in_file out_file\n", argv[0]);
    exit(-1);
  } else {
    iterations = atoi (argv[1]);
    nthreads = atoi (argv[2]);
    lo_margin = atof (argv[3]);
    input_file = argv[4];
    output_file = argv[5];
  }
  int n_measures = 1;
  times = (double*)malloc(n_measures * iterations * sizeof(double)); // we use 4
  // because we are going to measure the execution time for each GEMM and the
  // entire operation (matrix chain)
  printf("The number of iterations to perform is %d and the number of threads is %d\n", iterations, nthreads);

  float dims_[ndim];
  int dims[ndim];
  float parenths_[nparenth];
  int parenths[nparenth];
  float flops_diff, score;

  ifile.open(input_file, std::ifstream::in);
  if (ifile.fail()){
    printf("Error opening the file with anomalies...\n");
    exit(-1);
  }
  std::getline (ifile, line); // to read the headers


  ofile.open(output_file, std::ofstream::out);
  if (ofile.fail()){
    printf("Error opening the output file...\n");
    exit(-1);
  }
  add_headers_val (ofile, ndim, iterations * n_measures);

  mkl_set_dynamic(false);
  mkl_set_num_threads(nthreads);
  initialise_BLAS();

  while (!ifile.eof()){

    std::getline (ifile, line);
    // cout << line << endl;

    sscanf (line.c_str(), "%f,%f,%f,%f,%f,%f,%f,%f,%f", &dims_[0], &dims_[1], &dims_[2], &dims_[3], &dims_[4],
                                                &parenths_[0], &parenths_[1], &flops_diff, &score);
    // printf("The dimensions are: {%f, %f, %f, %f, %f}\n", dims[0], dims[1], dims[2], dims[3], dims[4]);
    // printf("The parenthesisations are: %f and %f\n", parenths[0], parenths[1]);
    // printf("Flops_diff and score: %f & %f\n", flops_diff, score);
    printf("score: %f\n", score);
    if (score >= lo_margin){
      printf("\t>> validating...\n");
      for (int i = 0; i < ndim; i++)
        dims[i] = int(dims_[i]);

      parenths[0] = int(parenths_[0]);
      parenths[1] = int(parenths_[1]);

      for (int p = 0; p < nparenth; p++){
        if (parenths[p] == 0)
          MC4_parenth_0 (dims, times, iterations, n_measures, nthreads);
        else if (parenths[p] == 1)
          MC4_parenth_1 (dims, times, iterations, n_measures, nthreads);
        else if (parenths[p] == 2)
          MC4_parenth_2 (dims, times, iterations, n_measures, nthreads);
        else if (parenths[p] == 3)
          MC4_parenth_3 (dims, times, iterations, n_measures, nthreads);
        else if (parenths[p] == 4)
          MC4_parenth_4 (dims, times, iterations, n_measures, nthreads);
        else if (parenths[p] == 5)
          MC4_parenth_5 (dims, times, iterations, n_measures, nthreads);
        add_line_val (ofile, dims, ndim, times, iterations * n_measures, parenths[p]);
      }
    }
  }

  free(times);
  ofile.close();
  ifile.close();
}

void exec_parenths(int parenths[], int nparenth, int dims[], double times[],
  const int iterations, const int n_measures, const int nthreads, std::ofstream &ofile){
  for (int p = 0; p < nparenth; p++){
    if (parenths[p] == 0)
      MC4_parenth_0 (dims, &times[p * iterations], iterations, n_measures, nthreads);
    else if (parenths[p] == 1)
      MC4_parenth_1 (dims, &times[p * iterations], iterations, n_measures, nthreads);
    else if (parenths[p] == 2)
      MC4_parenth_2 (dims, &times[p * iterations], iterations, n_measures, nthreads);
    else if (parenths[p] == 3)
      MC4_parenth_3 (dims, &times[p * iterations], iterations, n_measures, nthreads);
    else if (parenths[p] == 4)
      MC4_parenth_4 (dims, &times[p * iterations], iterations, n_measures, nthreads);
    else if (parenths[p] == 5)
      MC4_parenth_5 (dims, &times[p * iterations], iterations, n_measures, nthreads);
  }
}








