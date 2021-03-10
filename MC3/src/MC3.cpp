#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>

#include <cube.h>
#include <common.h>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

// void initialise_BLAS();
// double min_array (double* a, int size);
// double mean_array (double* a, int size);
void compare_algs (int d1, int d2, int d3, int d4, float margin, bool* ex_alg);

// void cache_flush();


int main (int argc, char **argv){

  int n_algs = 2;
  int iterations, d1, d2, d3, d4;
  int max_size, jump_size, points;

  double *A, *B, *C, *M1, *X;
  double one = 1.0;
  double *times;

  char transpose = 'N';
  std::ofstream ofile;

  bool ex_alg[n_algs];
  float margin = 2.0;
  unsigned long long flops1, flops2;

  string filename = "../timings/MC3/";

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

  initialise_BLAS();

  for (int alg = 0; alg < n_algs; alg++){
    for (d1 = jump_size; d1 <= max_size; d1 += jump_size){
      cout << "FILE alg" << alg << "/d1_" << d1 << endl;
      ofile.open(filename + "alg" + to_string(alg) + "/d1_" + to_string(d1) + ".csv", std::ios::out);
      if (ofile.fail()){
        cout << "HOLY SMOKES!" << endl;
        return -1;
      }

      ofile << "d1, d2, d3, d4, #Flops";
      for (int it = 0; it < iterations; it++){
        ofile << ", Sample_" << it;
      }
      ofile << '\n';

      for (d2 = jump_size; d2 <= max_size; d2 += jump_size){
        for (d3 = jump_size; d3 <= max_size; d3 += jump_size){
          cout << "\t{d2, d3}: {" << d2 << ", " << d3 << "}" << endl;
          for (d4 = jump_size; d4 <= max_size; d4 += jump_size){

            compare_algs (d1, d2, d3, d4, margin, ex_alg);
            ofile << d1 << ", " << d2 << ", " << d3 << ", " << d4 << ", ";
            if (ex_alg[alg]){
              A = (double*)malloc(d1 * d2 * sizeof(double));
              B = (double*)malloc(d2 * d3 * sizeof(double));
              C = (double*)malloc(d3 * d4 * sizeof(double));
              X = (double*)malloc(d1 * d4 * sizeof(double));

              for (int i = 0; i < d1 * d2; i++)
                A[i] = drand48();

              for (int i = 0; i < d2 * d3; i++)
                B[i] = drand48();

              for (int i = 0; i < d3 * d4; i++)
                C[i] = drand48();

              for (int i = 0; i < d1 * d4; i++)
                X[i] = drand48();

              if (alg == 0){
                M1 = (double*)malloc(d1 * d3 * sizeof(double));
                for (int i = 0; i < d1 * d3; i++)
                  M1[i] = drand48();

                flops1 = d1 * d3 * (d2 + d4);
                ofile << flops1;

                for (int it = 0; it < iterations; it++){
                  cache_flush();
                  auto time1 = std::chrono::high_resolution_clock::now();
                  dgemm_(&transpose, &transpose, &d1, &d3, &d2, &one, A, &d1, B, &d2, &one, M1, &d1);
                  dgemm_(&transpose, &transpose, &d1, &d4, &d3, &one, M1, &d1, C, &d3, &one, X, &d1);
                  auto time2 = std::chrono::high_resolution_clock::now();
                  times[it] = std::chrono::duration<double>(time2 - time1).count();
                }
              } else {
                M1 = (double*)malloc(d2 * d4 * sizeof(double));
                for (int i = 0; i < d2 * d4; i++)
                  M1[i] = drand48();

                flops2 = d2 * d4 * (d1 + d3);
                ofile << flops2;

                for (int it = 0; it < iterations; it++){
                  cache_flush();
                  auto time1 = std::chrono::high_resolution_clock::now();
                  dgemm_(&transpose, &transpose, &d2, &d4, &d3, &one, B, &d2, C, &d3, &one, M1, &d2);
                  dgemm_(&transpose, &transpose, &d1, &d4, &d2, &one, A, &d1, M1, &d2, &one, X, &d1);
                  auto time2 = std::chrono::high_resolution_clock::now();
                  times[it] = std::chrono::duration<double>(time2 - time1).count();
                }
              }
              for (int it = 0; it < iterations; it++){
                  ofile << ", " << std::fixed << std::setprecision(10) << times[it];
              }
              ofile << '\n';
              // ofile << std::fixed << std::setprecision(10) << to_string(min_array (times, iterations)) << ", ";
              // cout << std::fixed << std::setprecision(10) << to_string(min_array (times, iterations)) << endl;
              // ofile << std::fixed << std::setprecision(10) << to_string(mean_array (times, iterations)) << '\n';
              free(A);
              free(B);
              free(C);
              free(M1);
              free(X);
            }
            else {
              for (int it = 0; it < iterations; it++){
                ofile << ", NaN";
              }
              ofile << '\n';
            }
          }
        }
      }
      ofile.close();
    }
  }


  return 0;
}


void compare_algs (int d1, int d2, int d3, int d4, float margin, bool* ex_alg){
  unsigned long long flops1, flops2;

  flops1 = d1 * d3 * (d2 + d4);
  flops2 = d2 * d4 * (d1 + d3);

  if (flops1 > static_cast<unsigned long long>(flops2 * margin)){
    ex_alg[0] = false;
    ex_alg[1] = true;
  } else if (flops2 > static_cast<unsigned long long>(flops1 * margin)){
    ex_alg[0] = true;
    ex_alg[1] = false;
  } else {
    ex_alg[0] = true;
    ex_alg[1] = true;
  }
}







