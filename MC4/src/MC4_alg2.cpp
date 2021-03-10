#include <iostream>
#include <fstream>
#include <float.h>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <string>

using namespace std;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

static double *cs = NULL;

void initialise_BLAS();
void cache_flush();

int main (int argc, char **argv){

  int d1, d2, d3, d4, d5;
  // int max_size, jump_size, iterations;

  double *A, *B, *C, *D, *M1, *M2, *X;
  double one = 1.0;

  // std::ofstream file;
  // string filename = "../timings/MC4/";

  char transpose = 'N';

  if (argc != 6){
    cout << "Execution: " << argv[0] << " d1 d2 d3 d4 d5" << endl;
    return(-1);
  } else {
    d1 = atoi(argv[1]);
    d2 = atoi(argv[2]);
    d3 = atoi(argv[3]);
    d4 = atoi(argv[4]);
    d5 = atoi(argv[5]);
  }

  A = (double*)malloc(d1 * d2 * sizeof(double));
  B = (double*)malloc(d2 * d3 * sizeof(double));
  C = (double*)malloc(d3 * d4 * sizeof(double));
  D = (double*)malloc(d4 * d5 * sizeof(double));
  X = (double*)malloc(d1 * d5 * sizeof(double));

  M1 = (double*)malloc(d1 * d3 * sizeof(double));
  M2 = (double*)malloc(d1 * d4 * sizeof(double));

  initialise_BLAS();

  for (int i = 0; i < d1 * d2; i++)
    A[i] = drand48();

  for (int i = 0; i < d2 * d3; i++)
    B[i] = drand48();

  for (int i = 0; i < d3 * d4; i++)
    C[i] = drand48();

  for (int i = 0; i < d4 * d5; i++)
    D[i] = drand48();

  for (int i = 0; i < d1 * d5; i++)
    X[i] = drand48();

  for (int i = 0; i < d1 * d3; i++)
    M1[i] = drand48();

  for (int i = 0; i < d1 * d4; i++)
    M2[i] = drand48();


  auto time1 = std::chrono::high_resolution_clock::now();
  cache_flush();
  dgemm_(&transpose, &transpose, &d1, &d3, &d2, &one, A, &d1, B, &d2, &one, M1, &d1);
  dgemm_(&transpose, &transpose, &d1, &d4, &d3, &one, M1, &d1, C, &d3, &one, M2, &d1);
  dgemm_(&transpose, &transpose, &d1, &d5, &d4, &one, M2, &d1, D, &d4, &one, X, &d1);

  auto time2 = std::chrono::high_resolution_clock::now();

  cout << argv[0] << ": " << std::fixed << std::setprecision(10) << std::chrono::duration<double>(time2 - time1).count() << endl;

  free(A);
  free(B);
  free(C);
  free(D);
  free(X);
  free(M1);
  free(M2);

  return (0);
}


void initialise_BLAS(){
  double *A, *B, *C;
  double one = 1.0;

  int m = 500, n = 500, k = 500;
  char transpose = 'N';

  A = (double*)malloc(m * k * sizeof(double));
  B = (double*)malloc(k * n * sizeof(double));
  C = (double*)malloc(m * n * sizeof(double));

  for (int i = 0; i < m * k; i++)
    A[i] = drand48();

  for (int i = 0; i < k * n; i++)
    B[i] = drand48();

  for (int i = 0; i < m * n; i++)
    C[i] = drand48();

  // auto time1 = std::chrono::high_resolution_clock::now();
  dgemm_(&transpose, &transpose, &m, &n, &k, &one, A, &m, B, &k, &one, C, &m);
  // auto time2 = std::chrono::high_resolution_clock::now();

  free(A);
  free(B);
  free(C);
}


void cache_flush(){
	if (cs == NULL){
		cs = (double*)malloc(GEMM_L3_CACHE_SIZE * sizeof(double));
		for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++)
			cs[i] = drand48();
	}
	for (int i = 0; i < GEMM_L3_CACHE_SIZE; i++)
		cs[i] += 1e-3;
}