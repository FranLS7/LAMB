#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <mkl.h>

#include <common.h>
#include <omp.h>

using namespace std;

int main (int argc, char** argv){
  int ndim = 2;
  double one = 1.0;
  int align = 64;
  std::vector<int> points = {20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500,
    600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000};
  int iterations, nthreads;
  string output_file;

  if (argc != 4){
    cout << "Execution: " << argv[0] << " iterations nthreads output_file" << endl;
    return(-1);
  }
  else {
    iterations = atoi (argv[1]);
    nthreads = atoi (argv[2]);
    output_file = argv[3];
  }

  std::vector<int> dims (ndim);
  std::vector<double> times (iterations);
  std::ofstream ofile;

  ofile.open (output_file + string(".csv"));
  if (ofile.fail()){
    cout << "Error opening output file" << endl;
    return(-1);
  }
  lamb::printHeaderTime(ofile, ndim, iterations);

  auto start = std::chrono::high_resolution_clock::now();
  double *A, *B;
  mkl_set_dynamic (false);
  mkl_set_num_threads (nthreads);

  for (auto m : points){
    // for (auto k : points){
      int n = m;
      cout << "Executing with {" << m << "," << n << "}" << endl;
      dims = {m, n};

      A = static_cast<double*>(mkl_malloc(m * m * sizeof(double), align));
      for (int i = 0; i < m * m; i++)
        A[i] = drand48();

      B = static_cast<double*>(mkl_malloc(m * n * sizeof(double), align));
      for (int i = 0; i < m * n; i++)
        B[i] = drand48();

      for (int it = 0; it < iterations; it++){
        lamb::cacheFlush(nthreads);
        lamb::cacheFlush(nthreads);
        lamb::cacheFlush(nthreads);

        auto time1 = std::chrono::high_resolution_clock::now();
        cblas_dtrmm (CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                      m, n, one, A, m, B, n);
        auto time2 = std::chrono::high_resolution_clock::now();

        times[it] = std::chrono::duration<double>(time2 - time1).count();
      }
      lamb::printTime(ofile, dims, times);

      mkl_free(A);
      mkl_free(B);
    // }
  }

  auto end = std::chrono::high_resolution_clock::now();
  cout << "Execution time: " << std::chrono::duration<double>(end - start).count()
       << " seconds" << endl;

  ofile.close();


  return 0;
}