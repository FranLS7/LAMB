#ifndef GC_LIB
#define GC_LIB

extern "C" void CubeGEMM (int *min_size, int *max_size, int *npoints, int nsamples,
  int nthreads, char *output_filex);

void compute_points (int min_size, int max_size, int npoints, int *points);

#endif