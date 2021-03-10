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

  int nsamples, max_size, jump_size, m, k, n;
  double *A, *B, *C;
  double one = 1.0;
  int ndim = 3;

  char transpose = 'N';

  std::ofstream ofile;
  string filename = "../timings/3D/";

  if (argc != 5){
    cout << "Execution: " << argv[0] << " max_size jump_size nsamples filename" << endl;
    return(-1);
  } else {
    max_size = atoi(argv[1]);
    jump_size = atoi(argv[2]);
    nsamples = atoi(argv[3]);
    filename.append(argv[4]);
  }

  ofile.open (filename, std::ios::out);

  if (ofile.fail()){
    cout << "Error opening the file.." << endl;
    return -1;
  }

  add_headers(ofile, ndim, nsamples);

  initialise_BLAS();
  for (m = jump_size; m <= max_size; m += jump_size){ //jump between slices
    for (k = jump_size; k <= max_size; k += jump_size){ //jump between rows within same file
      for (n = jump_size; n <= max_size; n += jump_size){ //jump between contiguous elements within row
        ofile << m << ", " << k << ", " << n << ", ";

        A = (double*)malloc(m * k * sizeof(double));
        B = (double*)malloc(k * n * sizeof(double));
        C = (double*)malloc(m * n * sizeof(double));

        for (int i = 0; i < m * k; i++)
          A[i] = drand48();

        for (int i = 0; i < k * n; i++)
          B[i] = drand48();

        for (int i = 0; i < m * n; i++)
          C[i] = drand48();

        for (int it = 0; it < nsamples; it++){
          cache_flush();
          auto time1 = std::chrono::high_resolution_clock::now();
          dgemm_(&transpose, &transpose, &m, &n, &k, &one, A, &m, B, &k, &one, C, &m);
          auto time2 = std::chrono::high_resolution_clock::now();

          ofile << std::fixed << std::setprecision(10) << std::chrono::duration<double>(time2 - time1).count();
          if (it == nsamples - 1)
            ofile << '\n';
          else
            ofile << ", ";
        }

        cout << "Finished GEMM with (m=" << m << ", k=" << k << ", n=" << n << ")" << endl;

        free(A);
        free(B);
        free(C);
      } // n loop
    } // k loop
  } // m loop
  ofile.close();

  return 0;
}







