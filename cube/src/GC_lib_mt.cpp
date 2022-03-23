#include <float.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <string>
#include <math.h>
#include "mkl.h"

#include "common.h"
#include "GC_lib.h"

extern "C"{
  void CubeGEMM_mt (int *min_size, int *max_size, int *npoints, int nsamples,
    int n_threads, char *output_filex){

    mkl_set_num_threads(n_threads);
    CubeGEMM (min_size, max_size, npoints, nsamples, n_threads, output_filex);
    // printf("JAJAJAJAJJAJAJJAJA\n");

  }
}
