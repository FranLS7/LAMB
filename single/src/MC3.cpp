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

bool computation_decision (int *dims, int ndim, int threshold, double lo_margin,
  double up_margin, lamb::GEMM_Cube& cube);

inline double gemm_flops (int d0, int d1, int d2);
inline bool within_range (double x, double y, double lo_margin, double up_margin);
inline bool compare_flops (double flops_0, double flops_1, double margin_flops);

void parenth_0 (int *dims, double *times, const int iterations);
void parenth_1 (int *dims, double *times, const int iterations);

int main (int argc, char **argv){

  int *dims, ndim = 4;
  int max_size, jump_size, iterations;
  double *times;

  int threshold = 200;
  double lo_margin = 0.20;
  double up_margin = 0.40;

  int overhead = 2;
  string cube_filename = "../timings/3D/";
  string out_file0 = "../timings/MC3/";
  string out_file1 = "../timings/MC3/";

  std::ofstream ofile0, ofile1;

  if (argc != 6){
    cout << "Execution: " << argv[0] << " max_size jump_size iterations cube_file output_files" << endl;
    return (-1);
  } else {
    max_size = atoi(argv[1]);
    jump_size = atoi(argv[2]);
    iterations = atoi(argv[3]);
    cube_filename.append(argv[4]);
    out_file0.append(argv[5]); out_file0.append("0.csv");
    out_file1.append(argv[5]); out_file1.append("1.csv");
  }

  lamb::GEMM_Cube kubo (cube_filename, overhead);

  dims = (int*)malloc(ndim * sizeof(int));
  for (int i = 0; i < ndim; i++)
    dims[i] = jump_size;

  times = (double*)malloc(iterations * sizeof(double));


  // ==================================================================
  //   - - - - - - - - - - Opening output files - - - - - - - - - - -
  // ==================================================================
  ofile0.open (out_file0, std::ios::out);
  if (ofile0.fail()){
    printf("Error opening the file %s\n", out_file0.c_str());
    return(-1);
  }
  add_headers (ofile0, ndim, iterations);

  ofile1.open (out_file1, std::ios::out);
  if (ofile1.fail()){
    printf("Error opening the file %s\n", out_file1.c_str());
    return(-1);
  }
  add_headers (ofile1, ndim, iterations);

  bool foo;
  auto inicio = std::chrono::high_resolution_clock::now();
  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  initialise_BLAS();
  for (dims[0] = jump_size; dims[0] <= max_size; dims[0] += jump_size){
    for (dims[1] = jump_size; dims[1] <= max_size; dims[1] += jump_size){
      printf(">> {%d,%d} ", dims[0], dims[1]);
      auto timex = std::chrono::high_resolution_clock::now();
      for (dims[2] = jump_size; dims[2] <= max_size; dims[2] += jump_size){
        for (dims[3] = jump_size; dims[3] <= max_size; dims[3] += jump_size){
          // TODO: HERE IS WHERE THE INTELLIGENCE AND COMPUTATION TAKE PLACE
          // >> perhaps it's better to use while-loosps instead.
          // auto time1 = std::chrono::high_resolution_clock::now();
          foo = computation_decision (dims, ndim, threshold, lo_margin, up_margin, kubo);
          // auto time2 = std::chrono::high_resolution_clock::now();
          // printf("\t * Decision time: %.10f\n", std::chrono::duration<double>(time2 - time1).count());
          if(foo){
            // printf("\tWe are computing boyz!\n");
            parenth_0 (dims, times, iterations);
            add_line (ofile0, dims, ndim, times, iterations);
            parenth_1 (dims, times, iterations);
            add_line (ofile1, dims, ndim, times, iterations);
          }
          // else
            // printf("\tWe are AVOIDING computing boyz!\n");
        }
      }
      auto timexx = std::chrono::high_resolution_clock::now();
      printf("\t · Computing time: %5.10f\n", std::chrono::duration<double>(timexx - timex).count());
    }
  }

  auto fin = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(fin - inicio).count());

  free(dims);
  free(times);

  ofile0.close();
  ofile1.close();


  return 0;
}

// This function may eventually include all the computation
bool computation_decision (int *dims, int ndim, int threshold, double lo_margin,
  double up_margin, lamb::GEMM_Cube& cube){
  // First, check whether the values are within the compulsory range.
  bool compute = true;
  for (int i = 0; i < ndim; i++){
    if (dims[i] > threshold){
      compute = false;
      break;
    }
  }
  if (compute)
    return compute;
  else{
    for (int i = 0; i < ndim; i++){
      if (dims[i] < threshold)
        compute = true;
    }
    if (!compute)
      return compute;
    else{
      // Check the number of flops
      double flops_0 = gemm_flops(dims[0], dims[1], dims[2]) +
        gemm_flops(dims[0], dims[2], dims[3]);
      double flops_1 = gemm_flops(dims[1], dims[2], dims[3]) +
        gemm_flops(dims[0], dims[1], dims[3]);

      if (within_range (flops_0, flops_1, lo_margin, up_margin)){
        // If they are close (still) check the time predictions or v_FLOPs
        double e_time_0 = cube.get_value (dims[0], dims[1], dims[2]) +
          cube.get_value (dims[0], dims[2], dims[3]);

        double e_time_1 = cube.get_value (dims[1], dims[2], dims[3]) +
          cube.get_value (dims[0], dims[1], dims[3]);
        // double v_flops_0 = gemm_flops(dims[0], dims[1], dims[2]) /
        //   cube.get_value(dims[0], dims[1], dims[2]) +
        //   gemm_flops(dims[0], dims[2], dims[3]) /
        //   cube.get_value(dims[0], dims[2], dims[3]);

        // double v_flops_1 = gemm_flops(dims[1], dims[2], dims[3]) /
        //   cube.get_value(dims[1], dims[2], dims[3]) +
        //   gemm_flops(dims[0], dims[1], dims[3]) /
        //   cube.get_value(dims[0], dims[1], dims[3]);
        if (within_range (e_time_0, e_time_1, lo_margin, up_margin))
          compute = true;
        else
          compute = false;
      }
      else
        compute = false;
    }
  }
  return compute;
}

inline double gemm_flops (int d0, int d1, int d2){
  return double (2 * d0 * d1 * d2);
}

inline bool within_range (double x, double y, double lo_margin, double up_margin){
  double z = (abs(x - y) / max(x, y));
  return (lo_margin < z && z < up_margin);
}

inline bool compare_flops (double flops_0, double flops_1, double margin_flops){
  return ((flops_0 / flops_1) < margin_flops && (flops_1 / flops_0) < margin_flops);
}


void parenth_0 (int *dims, double *times, const int iterations){
  double *A, *B, *C, *M1, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  B = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  C = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  M1 = (double*)malloc(dims[0] * dims[2] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[3] * sizeof(double));

  // Initialise each value for each matrix
  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    C[i] = drand48();

  for (int i = 0; i < dims[0] * dims[2]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    X[i] = drand48();

  // Compute the operation as many times as iterations specifies
  for (int it = 0; it < iterations; it++){
    // Flush/scrub the cache so we are working with 'cold' data
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[0], &dims[2], &dims[1], &one, A,
      &dims[0], B, &dims[1], &one, M1, &dims[0]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[2], &one, M1,
      &dims[0], C, &dims[2], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();
    // Store the timings in one array so we can write them all at once
    // in the output file.
    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  // Free the previously allocated memory.
  free(A);
  free(B);
  free(C);
  free(M1);
  free(X);
}

void parenth_1 (int *dims, double *times, const int iterations){
  double *A, *B, *C, *M1, *X;
  double one = 1.0;
  char transpose = 'N';

  // Memory allocation for all the matrices
  A = (double*)malloc(dims[0] * dims[1] * sizeof(double));
  B = (double*)malloc(dims[1] * dims[2] * sizeof(double));
  C = (double*)malloc(dims[2] * dims[3] * sizeof(double));
  M1 = (double*)malloc(dims[1] * dims[3] * sizeof(double));
  X = (double*)malloc(dims[0] * dims[3] * sizeof(double));

  // Initialise each value for each matrix
  for (int i = 0; i < dims[0] * dims[1]; i++)
    A[i] = drand48();

  for (int i = 0; i < dims[1] * dims[2]; i++)
    B[i] = drand48();

  for (int i = 0; i < dims[2] * dims[3]; i++)
    C[i] = drand48();

  for (int i = 0; i < dims[1] * dims[3]; i++)
    M1[i] = drand48();

  for (int i = 0; i < dims[0] * dims[3]; i++)
    X[i] = drand48();

  // Compute the operation as many times as iterations specifies
  for (int it = 0; it < iterations; it++){
    // Flush/scrub the cache so we are working with 'cold' data
    cache_flush();
    auto time1 = std::chrono::high_resolution_clock::now();
    dgemm_(&transpose, &transpose, &dims[1], &dims[3], &dims[2], &one, B,
      &dims[1], C, &dims[2], &one, M1, &dims[1]);
    dgemm_(&transpose, &transpose, &dims[0], &dims[3], &dims[1], &one, A,
      &dims[0], M1, &dims[1], &one, X, &dims[0]);
    auto time2 = std::chrono::high_resolution_clock::now();
    // Store the timings in one array so we can write them all at once
    // in the output file.
    times[it] = std::chrono::duration<double>(time2 - time1).count();
  }

  // Free the previously allocated memory.
  free(A);
  free(B);
  free(C);
  free(M1);
  free(X);
}













