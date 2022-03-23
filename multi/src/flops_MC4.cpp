#include <limits.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>


unsigned long long int compute_flops (int dims[], int parenth);
unsigned long long int flops_parenth_0 (int dims[]);
unsigned long long int flops_parenth_1 (int dims[]);
unsigned long long int flops_parenth_2 (int dims[]);
unsigned long long int flops_parenth_3 (int dims[]);
unsigned long long int flops_parenth_4 (int dims[]);
unsigned long long int flops_parenth_5 (int dims[]);

int main (int argc, char **argv){
  int ndim = 5;
  int dims[ndim];

  if (argc != 6){
    printf("Execution: %s d0 d1 d2 d3 d4\n", argv[0]);
    exit(-1);
  } else{
    for (int i = 0; i < ndim; i++)
      dims[i] = atoi (argv[i + 1]);
  }

  printf("max value for unsigned long long int: %llu\n", ULLONG_MAX);
  printf("flops parenth_0: %llu\n", flops_parenth_0(dims));
  printf("flops parenth_1: %llu\n", flops_parenth_1(dims));
  printf("flops parenth_2: %llu\n", flops_parenth_2(dims));
  printf("flops parenth_3: %llu\n", flops_parenth_3(dims));
  printf("flops parenth_4: %llu\n", flops_parenth_4(dims));
  printf("flops parenth_5: %llu\n", flops_parenth_5(dims));

  return 0;
}














  unsigned long long int flops_parenth_0 (int dims[]){
    unsigned long long int result;
    result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
    // printf("\tFirst GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]));
    result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[3]);
    // printf("\tSecond GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[3]));
    result += uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]);
    // printf("\tThird GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]));
    return result * 2;
  }

  unsigned long long int flops_parenth_1 (int dims[]){
    unsigned long long int result;
    result = uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]);
    // printf("\tFirst GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[2]));
    result += uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]);
    // printf("\tSecond GEMM: %lu\n", 2 * uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]));
    result += uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[4]);
    // printf("\tThird GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[2]) * uint64_t(dims[4]));
    return result * 2;
  }

  unsigned long long int flops_parenth_2 (int dims[]){
    unsigned long long int result;
    result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
    // printf("\tFirst GEMM: %lu\n", 2 * uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]));
    result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[3]);
    // printf("\tSecond GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[3]));
    result += uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]);
    // printf("\tThird GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[3]) * uint64_t(dims[4]));
    return result * 2;
  }

  unsigned long long int flops_parenth_3 (int dims[]){
    unsigned long long int result;
    result = uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]);
    // printf("\tFirst GEMM: %lu\n", 2 * uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[3]));
    result += uint64_t(dims[1]) * uint64_t(dims[3]) * uint64_t(dims[4]);
    // printf("\tSecond GEMM: %lu\n", 2 * uint64_t(dims[1]) * uint64_t(dims[3]) * uint64_t(dims[4]));
    result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]);
    // printf("\tThird GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]));
    return result * 2;
  }

  unsigned long long int flops_parenth_4 (int dims[]){
    return flops_parenth_1(dims);
  }

  unsigned long long int flops_parenth_5 (int dims[]){
    unsigned long long int result;
    result = uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]);
    // printf("\tFirst GEMM: %lu\n", 2 * uint64_t(dims[2]) * uint64_t(dims[3]) * uint64_t(dims[4]));
    result += uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[4]);
    // printf("\tSecond GEMM: %lu\n", 2 * uint64_t(dims[1]) * uint64_t(dims[2]) * uint64_t(dims[4]));
    result += uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]);
    // printf("\tThird GEMM: %lu\n", 2 * uint64_t(dims[0]) * uint64_t(dims[1]) * uint64_t(dims[4]));
    return result * 2;
  }