#ifndef COMMON_FUNC
#define COMMON_FUNC

void initialise_BLAS();

void add_headers (std::ofstream & ofile, int ndim, int nsamples);

void add_line (std::ofstream &ofile, int *dims, int ndim, double *times, int nsamples);

void cache_flush();

double min_array (double* a, int size);

double mean_array (double* a, int size);

#endif