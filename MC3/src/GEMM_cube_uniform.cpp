#include <float.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <string>

#include <cube.h>
#include <common.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

int main (int argc, char **argv){

  int iterations, max_size, jump_size, points, m, k, n;
  double *A, *B, *C;
  double one = 1.0;

  double *times;

  char transpose = 'N';

  std::ofstream ofile;
  string filename = "../timings/3D/GEMM_m";

  if (argc != 4){
    cout << "Execution: " << argv[0] << " max_size jump_size iterations" << endl;
    return(-1);
  } else {
    max_size = atoi(argv[1]);
    jump_size = atoi(argv[2]);
    iterations = atoi(argv[3]);
  }

  times = (double*)malloc(iterations * sizeof(double));

  points = int(max_size / jump_size);

  for (m = jump_size; m <= max_size; m += jump_size){ //jump between slices
    ofile.open(filename + to_string(m) + ".csv", std::ios::out);
    cout << "Filename: " << filename << m << ".csv" << endl;
    if(ofile.fail()){
      cout << "HOLY SMOKES!" << endl;
      return -1;
    }
    ofile << "k/n, ";

    for (int i = 0; i < points; i++){
      if (i == points - 1)
        ofile << (i + 1) * jump_size << '\n';
      else
        ofile << (i + 1) * jump_size << ", ";
    }

    for (k = jump_size; k <= max_size; k += jump_size){ //jump between rows within same file
      ofile << k << ", ";
      for (n = jump_size; n <= max_size; n += jump_size){ //jump between contiguous elements within row

        A = (double*)malloc(m * k * sizeof(double));
        B = (double*)malloc(k * n * sizeof(double));
        C = (double*)malloc(m * n * sizeof(double));

        for (int i = 0; i < m * k; i++)
          A[i] = drand48();

        for (int i = 0; i < k * n; i++)
          B[i] = drand48();

        for (int i = 0; i < m * n; i++)
          C[i] = drand48();

        for (int it = 0; it <= iterations; it++){
          auto time1 = std::chrono::high_resolution_clock::now();
          dgemm_(&transpose, &transpose, &m, &n, &k, &one, A, &m, B, &k, &one, C, &m);
          auto time2 = std::chrono::high_resolution_clock::now();

          times[it] = std::chrono::duration<double>(time2 - time1).count();
        }

        ofile << std::fixed << std::setprecision(10) << mean_array(times, iterations);
        if (n == max_size)
          ofile << '\n';
        else
          ofile << ", ";

        cout << "Finished GEMM with (m=" << m << ", k=" << k << ", n=" << n << ")" << endl;

        free(A);
        free(B);
        free(C);
      } // n loop
    } // k loop
    ofile.close();
  } // m loop

  free(times);

  return 0;
}















