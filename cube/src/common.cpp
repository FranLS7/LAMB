#include <stdlib.h>
#include <fstream>
#include <limits>
#include <iomanip>
#include "common.h"

using namespace std;

static double *cs = NULL;

extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);


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

  dgemm_(&transpose, &transpose, &m, &n, &k, &one, A, &m, B, &k, &one, C, &m);

  free(A);
  free(B);
  free(C);
}

void add_headers (std::ofstream &ofile, int ndim, int nsamples){
  ofile << "# ";
  for (int i = 0; i < ndim; i++)
    ofile << 'd' << i << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << "Sample_" << i;

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
}

void add_headers_cube (std::ofstream &ofile, int ndim, int *min_size, int *max_size,
  int *npoints, int nsamples, int nthreads, int **points){

    ofile << "# " << nthreads << endl;

  for (int i = 0; i < ndim; i++){
    // printf("I am printing stuff onto the output files!\n");
    ofile << "# " << min_size[i] << "," << max_size[i] << "," << npoints[i] << endl;
    // ofile << "# " << max_size[i] << endl;
    // ofile << "# " << npoints[i] << endl;
    ofile << "# ";

    for (int ii = 0; ii < npoints[i]; ii++){
      ofile << points[i][ii];
      if (ii == npoints[i] - 1)
        ofile << '\n';
      else
        ofile << ',';
    }
  }

  add_headers (ofile, ndim, nsamples);
}



void add_line (std::ofstream &ofile, int *dims, int ndim, double *times, int nsamples){
  for (int i = 0; i < ndim; i++)
    ofile << dims[i] << ',';

  for (int i = 0; i < nsamples; i++){
    ofile << std::fixed << std::setprecision(10) << times[i];

    if (i == nsamples - 1)
      ofile << '\n';
    else
      ofile << ',';
  }
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


double min_array (double* a, int size){
  double x = numeric_limits<double>::max();

  for (int i = 0; i < size; i++){
    if (a[i] < x)
      x = a[i];
  }

  return x;
}


double mean_array (double* a, int size){
  double x = 0.0;

  for (int i = 0; i < size; i++){
      x += a[i];
  }

  return x/size;
}

