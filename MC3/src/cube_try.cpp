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
// using namespace lamb;

int main (int argc, char **argv){

  string filename = "../timings/3D/GEMM_eff.csv";
  int overhead = 2;
  int limit = 51;
  double value = 0.0;

  auto timex = std::chrono::high_resolution_clock::now();
  lamb::GEMM_Cube kube (filename, overhead);
  auto timexx = std::chrono::high_resolution_clock::now();
  printf("Cube created in %.10fs\n", std::chrono::duration<double>(timexx - timex).count());


  for (int i = 10; i < limit; i += 10){
    for (int j = 10; j < limit; j += 10){
      for (int k = 10; k < limit; k += 10){
        auto time1 = std::chrono::high_resolution_clock::now();
        value = kube.get_value(i, j, k);
        auto time2 = std::chrono::high_resolution_clock::now();

        // printf ("{%d, %d, %d} --> %f in %.10f s\n", i, j, k, value, std::chrono::duration<double>(time2 - time1).count());

      }
    }
  }
  return 0;
}