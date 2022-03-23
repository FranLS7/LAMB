#ifndef GC_LIB
#define GC_LIB

extern "C" void CubeGEMM (int *min_size, int *max_size, int *npoints, int nsamples,
  int nthreads, char *output_filex);

void compute_points (int min_size, int max_size, int npoints, int *points);

/**
 * Adds the headers to an output file for a generated cube. Format:
 * # nthreads
 * # d0:    min_size, max_size, npoints
 * # ...
 * # di-1:  min_size, max_size, npoints
 * # d0:    points
 * # ...
 * # di-1:  points
 * # | ndim dims || nsamples samples |
 *
 * @param ofile     Output file manager which has been previously opened.
 * @param ndim      The number of dimensions in the cube (usually 3).
 * @param min_size  Allocated memory with the min sizes for each dimension.
 * @param max_size  Allocated memory with the max sizes for each dimension.
 * @param npoints   Allocated memory with the number of points for each
 *      dimension (might be different depending on the dimension).
 * @param nsamples  The number of samples to store in the output file.
 * @param nthreads  The number of threads to use in the computation - is stored.
 * @param points    Allocated memory with the points for each dimension (might
 *      be different depending on the dimension).
 */
void GEMM_Cube::print_header_cube (std::ofstream &ofile, const int ndim, const int* min_size,
    const int* max_size, const int* npoints, const int nsamples, const int nthreads,
    const int **points);

#endif