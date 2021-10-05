#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <string>
#include <random>
#include <algorithm>
#include "mkl.h"

#include <cube.h>
#include <common.h>
#include <MC4.h>

using namespace std;

// ----------------- CONSTANTS ----------------- //
const int NPAR = 6;
const int NDIM = 5;

// ----------------- FUNCTIONS ----------------- //
extern "C" int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/**
 * Checks whether there is at least one dimension within the threshold.
 * This way we increase the possibilities of finding anomalies.
 *
 * @param dims      The array with the matrices dimensions.
 * @param ndim      The number of total dimensions in the problem.
 * @param threshold The max size for at least one dimension.
 * @return          True if at least one dimension is smaller than threshold.
 * False, otherwise.
 */
bool computation_decision (int dims[], const int ndim, const int threshold);

/**
 * Checks whether there is an anomaly in the computed results.
 *
 * @param dims        The array with the matrices dimensions.
 * @param ndim        The number of dimensions in the problem.
 * @param times       The array containing all the execution times.
 * @param nparenth    The number of parenthesisations that have been computed.
 * @param iterations  The number of times each parenthesisation has been computed.
 * @param lo_margin   Minimum difference between median values to consider a
 * point as an anomaly.
 * @ratio             The percentage of total executions for which the
 * parenthesisation with more FLOPs must be faster.
 * @param ofile_an    The output file manager for anomalies.
 * @return            Whether there is an anomaly in the computed results.
 */
bool check_anomaly(int dims[], int ndim, double* times, const int nparenth,
  const int iterations, const double lo_margin, const double ratio, const int n_threads,
  std::ofstream &ofile_an);


int main (int argc, char **argv) {
  int dims[NDIM];
  int min_size, max_size, iterations, n_anomalies, n_threads, threshold;
  double lo_margin, ratio;

  string base_filename;
  std::ofstream ofiles[NPAR];
  std::ofstream ofile_an;

  if (argc != 10){
    cout << "Execution: " << argv[0] << " min_size max_size iterations n_anomalies n_threads"
      " threshold lo_margin ratio base_filename" << endl;
    return (-1);
  } else {
    min_size    = atoi (argv[1]); // min size a dimension can take.
    max_size    = atoi (argv[2]); // max size a dimension can take.
    iterations  = atoi (argv[3]); // number of times each parenth is computed.
    n_anomalies = atoi (argv[4]); // number of anomalies to be found.
    n_threads   = atoi (argv[5]); // number of threads to use during computation.
    threshold   = atoi (argv[6]); // at least one dim must be less than this.
    lo_margin   = atof (argv[7]); // min score to be considered as anomaly.
    ratio       = atof (argv[8]); // porcentual number of times a parenth
      // must be faster than the other one.
    base_filename.append (argv[9]); // base filename to create output files.
  }

  // double* times = (double*)malloc(nparenth * iterations * sizeof(double));
  double times [NPAR * iterations]; // array where execution times are stored
  // for all the parenthesisations.

  // ==================================================================
  //   - - - - - - - - - - Opening output files - - - - - - - - - - -
  // ==================================================================
  for (int i = 0; i < NPAR; i++){
    ofiles[i].open (base_filename + string("parenth_") + to_string(i) + string(".csv"));
    if (ofiles[i].fail()) {
      cout << "Error opening the file for parenthesisation " << i << endl;
      exit(-1);
    }
    add_headers (ofiles[i], NDIM, iterations);
  }

  ofile_an.open (base_filename + string("summary.csv"));
  if (ofile_an.fail()) {
    cout << "Error opening the file for anomalies summary" << endl;
    exit(-1);
  } else {
    add_headers_anomalies (ofile_an, NDIM);
  }

  // ==================================================================
  //   - - - - - - - - - - - Proper computation  - - - - - - - - - - -
  // ==================================================================
  mkl_set_dynamic (0);
  mkl_set_num_threads (n_threads);
  initialise_BLAS ();

  auto start = std::chrono::high_resolution_clock::now();
  unsigned long long int computed_points = 0;
  bool compute;

  for (int i = 0; i < n_anomalies; ){
    // Fix possible values at zero and below min_size
    for (int dim = 0; dim < NDIM; dim++){
      dims[dim] = int(drand48() * (max_size - min_size)) + min_size;
    }

    compute = computation_decision (dims, NDIM, threshold);

    if (compute){
      computed_points++;
      printf(">> %llu points, %d anomalies found -- {", computed_points, i);
      for (int i = 0; i < NDIM; i++)
        printf("%d, ", dims[i]);
      printf("}\n");
      MC4_parenth_0 (dims, times, iterations, 1, n_threads);
      MC4_parenth_1 (dims, &times[1 * iterations], iterations, 1, n_threads);
      MC4_parenth_2 (dims, &times[2 * iterations], iterations, 1, n_threads);
      MC4_parenth_3 (dims, &times[3 * iterations], iterations, 1, n_threads);
      MC4_parenth_4 (dims, &times[4 * iterations], iterations, 1, n_threads);
      MC4_parenth_5 (dims, &times[5 * iterations], iterations, 1, n_threads);

      if (check_anomaly(dims, NDIM, times, NPAR, iterations, lo_margin,
        ratio, n_threads, ofile_an)){
        i++;
        printf("Anomaly found! So far we have found %d\n", i);
        for (int i = 0; i < NPAR; i++)
          add_line_ (ofiles[i], dims, NDIM, &times[i * iterations], iterations);
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  printf("TOTAL Computing time: %f\n", std::chrono::duration<double>(end - start).count());
  printf("TOTAL computed points: %llu\n", computed_points);

  // ==================================================================
  //   - - - - - - - - - - Closing output files - - - - - - - - - - -
  // ==================================================================
  for (int i = 0; i < NPAR; i++){
    ofiles[i].close();
  }
  ofile_an.close();

  return 0;
}


/**
 * Checks whether there is at least one dimension within the threshold.
 * This way we increase the possibilities of finding anomalies.
 *
 * @param dims      The array with the matrices dimensions.
 * @param ndim      The number of total dimensions in the problem.
 * @param threshold The max size for at least one dimension.
 * @return          True if at least one dimension is smaller than threshold.
 * False, otherwise.
 */
bool computation_decision (int dims[], const int ndim, const int threshold){
  bool compute = false;

  for (int i = 0; i < ndim; i++){
    if (dims[i] <= threshold)
      compute = true;
  }
  return compute;
}

/**
 * Checks whether there is an anomaly in the computed results.
 *
 * @param dims        The array with the matrices dimensions.
 * @param times       The array containing all the execution times.
 * @param nparenth    The number of parenthesisations that have been computed.
 * @param iterations  The number of times each parenthesisation has been computed.
 * @param lo_margin   Minimum difference between median values to consider a
 * point as an anomaly.
 * @ratio             The percentage of total executions for which the
 * parenthesisation with more FLOPs must be faster.
 * @return            Whether there is an anomaly in the computed results.
 */
bool check_anomaly(int dims[], int ndim, double* times, const int nparenth,
  const int iterations, const double lo_margin, const double ratio, const int n_threads,
  std::ofstream &ofile_an){

  bool anomaly = false;

  unsigned long long int flops[nparenth];
  double median[nparenth];
  double time_score = 0.0;
  double flops_score = 0.0;

  unsigned seed;

  for (int p = 0; p < nparenth; p++){
    flops[p] = MC4_flops (dims, p);
    median[p] = median_array (&times[p * iterations], iterations);
  }

  // Shuffle the timings belonging to each parenthesisation
  for (int p = 0; p < nparenth; p++){
    seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    shuffle (&times[p * iterations], &times[p * iterations] + iterations,
      default_random_engine (seed));
  }

  for (int i = 0; i < nparenth; i++){
    for (int j = i + 1; j < nparenth; j++){
      time_score = score (median[i], median[j]);
      flops_score = score (flops[i], flops[j]);

      if ((time_score >= lo_margin) &&
        is_anomaly (flops[i], flops[j], &times[i * iterations],
                    &times[j * iterations], iterations, ratio)){
          anomaly = true;
          add_line_anomalies (ofile_an, dims, ndim, n_threads, i, j, flops_score, time_score);
          break;
      }
    }
  }
  return anomaly;
}
