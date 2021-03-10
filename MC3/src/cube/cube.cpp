#include <float.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <math.h>

#include "cube.h"

using namespace std;
using namespace lamb;


namespace {
  // Function that returns the index of a value in a certain array.
  // The returned index will be within the range {l, r}, unless the value
  // we are looking for is not present in the array. In such case, the
  // returned value will be -1.
  int binarySearch (int* arr, int l, int r, int value){
    if (r >= l){
      int mid = l + int((r - l) / 2);

      if (arr[mid] == value)
        return mid;

      if (arr[mid] > value)
        return binarySearch (arr, l, mid - 1, value);

      return binarySearch (arr, mid + 1, r, value);
    }

    return -1;
  }

  // Function that returns the index of the greatest number that is less than
  // the passed value
  int searchLower (int* arr, int length, int value){
    int idx = 0;
    for (idx = 0; idx < length - 1; idx++){
      if (arr[idx + 1] > value) break;
    }
    return idx;
  }
}


namespace lamb{
  GEMM_Cube::GEMM_Cube ()
    : min_size(0), max_size(0), jump_size(0), npoints(0) {}


  GEMM_Cube::GEMM_Cube (std::string filename, const int overhead){
    if (!load_cube(filename, overhead))
      cout << "Error loading the cube" << endl;
  }


  bool GEMM_Cube::load_cube (std::string filename, const int overhead){
    std::ifstream ifile;
    ifile.open (filename, std::ios::in);

    if (ifile.fail()){
      cout << "Error opening the gemm_cube file" << endl;
      return false;
    }

    string s;

    // Load headers - cube information
    getline (ifile, s);
    min_size = stoi (s.substr (overhead, s.length() - overhead));

    getline (ifile, s);
    max_size = stoi (s.substr (overhead, s.length() - overhead));

    getline (ifile, s);
    jump_size = stoi (s.substr (overhead, s.length() - overhead));

    // Initialise the linearised cube
    npoints = (max_size - min_size) / jump_size + 1;
    data = (double*)malloc(pow(npoints, 3.0) * sizeof(double));

    create_points ();

    for (int i = 0; i < pow(npoints, 3.0); i++){
      getline (ifile, s);
      data[i] = stod (s);
    }

    ifile.close();
    cout << "DUDE, THE CUBE HAS BEEN CREATED PROPERLY!" << endl;
    return true;
  }


  // Creating this list of points would be useful when the sampling is not uniform
  void GEMM_Cube::create_points (){
    points = (int*)malloc(npoints * sizeof(int));
    for (int i = 0; i < npoints; i++){
      points[i] = min_size + i * jump_size;
    }
  }


  bool GEMM_Cube::is_in_cube (const int* dims, int* indices) const{
    bool is = true;

    for (int i = 0; i < 3; i++){
      indices[i] = binarySearch (points, 0, npoints - 1, dims[i]);
      if (indices[i] == -1){
        is = false;
      }
    }

    return is;
  }

  inline double GEMM_Cube::access_cube (const int *indices) const {
    return data[indices[0] * npoints * npoints + indices[1] * npoints + indices[2]];
  }

  void GEMM_Cube::get_ranges (const int *dims, const int *indices, int *ranges) const{
    for (int i = 0; i < 3; i++){
      if (indices[i] != -1){
        ranges[2 * i] = indices[i];
        ranges[2 * i + 1] = indices[i];
      }
      else{
        ranges[2 * i] = searchLower(points, npoints, dims[i]);
        ranges[2 * i + 1] = ranges[2 * i] + 1;
      }
    }
  }


  // Trilinear interpolation with uniform sampling
  double GEMM_Cube::trilinear_inter (const int* dims_o, const int* ranges) const{
    float r_distances[3];
    double values_lattice[8];
    for (int i = 0; i < 3; i++){
      if (ranges[2 * i] == ranges[2 * i + 1])
        r_distances[i] = 0.0f;
      else
        r_distances[i] = double(dims_o[i] - points[ranges[2 * i]]) /
          double(points[ranges[2 * i + 1]] - points[ranges[2 * i]]);
    }

    for (int i = 0; i < 2; i++){
      for (int j = 0; j < 2; j++){
        for (int k = 0; k < 2; k++){
          int indices[3] = {ranges[i], ranges[2 + j], ranges[4 + k]};
          values_lattice[i * 4 + j * 2 + k] = access_cube(indices);
        }
      }
    }

    double c00 = values_lattice[0] * (1.0f - r_distances[0]) + values_lattice[4] * r_distances[0];
    double c01 = values_lattice[1] * (1.0f - r_distances[0]) + values_lattice[5] * r_distances[0];
    double c10 = values_lattice[2] * (1.0f - r_distances[0]) + values_lattice[6] * r_distances[0];
    double c11 = values_lattice[3] * (1.0f - r_distances[0]) + values_lattice[7] * r_distances[0];

    double c0 = c00 * (1.0f - r_distances[1]) + c10 * r_distances[1];
    double c1 = c01 * (1.0f - r_distances[1]) + c11 * r_distances[1];

    return (c0 * (1.0f - r_distances[2]) + c1 * r_distances[2]);
  }


  // Function that returns a certain value from the eff cube
  double GEMM_Cube::get_value (const int d1, const int d2, const int d3) const{
    int dims[3] = {d1, d2, d3};
    int indices[3] = {-1, -1, -1};
    int ranges[6];

    // Check whether values are out of range; this truncation might be revised
    // in the future.
    for (int i = 0; i < 3; i++){
      if (dims[i] > max_size) dims[i] = max_size;
      else if (dims[i] < min_size) dims[i] = min_size;
      // printf ("\t >> dims[%d] == %d\n", i, dims[i]);
    }

    if (is_in_cube (dims, indices)){
      return access_cube(indices);
    }

    // Check how many dimensions we have to interpolate AND EXTRACT THE RANGES
    get_ranges (dims, indices, ranges);

    return trilinear_inter(dims, ranges);
  }
}


