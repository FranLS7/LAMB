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

  // Compute points in the form of a parabola: y = ax2 + bx + c
  // This parabola has its vertex in x=0 --> 'b' = 0.
  // For the same reason --> 'c' = min_size.
  // 'a' is left to compute, then.
  void compute_points (int min_size, int max_size, int npoints, int *points){
    double a = (max_size - min_size) / pow(npoints - 1, 2.0);

    for (int i = 0; i < npoints; i++){
      points[i] = int(a * pow(i, 2.0)) + min_size;
      if (i > 0 && points[i] <= points[i - 1])
        points[i] = points[i-1] + 1;
      // printf("points[%d] : %d\n", i, points[i]);
    }
  }
}


namespace lamb{
  GEMM_Cube::GEMM_Cube () {}

  GEMM_Cube::GEMM_Cube (std::string filename, const int overhead){
    axes[0] = &d0;
    axes[1] = &d1;
    axes[2] = &d2;

    if (!load_cube(filename, overhead))
      cout << "Error loading the cube" << endl;
  }


  bool GEMM_Cube::load_cube (std::string filename, const int overhead){
    std::ifstream ifile;
    ifile.open (filename, std::ios::in);
    total_points=1;

    if (ifile.fail()){
      cout << "Error opening the gemm_cube file" << endl;
      return false;
    }

    string s;

    // Load headers - cube information
    for (int i = 0; i < 3; i++){
      getline (ifile, s);
      axes[i]->min_size = stoi (s.substr (overhead, s.length() - overhead));
      getline (ifile, s);
      axes[i]->max_size = stoi (s.substr (overhead, s.length() - overhead));
      getline (ifile, s);
      axes[i]->npoints = stoi (s.substr (overhead, s.length() - overhead));
      axes[i]->points = (int*)malloc(axes[i]->npoints * sizeof(int));

      compute_points (axes[i]->min_size, axes[i]->max_size, axes[i]->npoints, axes[i]->points);
      // for (int kk = 0; kk < axes[i]->npoints; kk++){
      //   printf("Value [%d, %d]: %d\n", i, kk, axes[i]->points[kk]);
      // }
      total_points *= axes[i]->npoints;
    }


    // getline (ifile, s);
    // d1.min_size = stoi (s.substr (overhead, s.length() - overhead));
    // getline (ifile, s);
    // d1.max_size = stoi (s.substr (overhead, s.length() - overhead));
    // getline (ifile, s);
    // d1.npoints = stoi (s.substr (overhead, s.length() - overhead));
    //
    // getline (ifile, s);
    // d2.min_size = stoi (s.substr (overhead, s.length() - overhead));
    // getline (ifile, s);
    // d2.max_size = stoi (s.substr (overhead, s.length() - overhead));
    // getline (ifile, s);
    // d2.npoints = stoi (s.substr (overhead, s.length() - overhead));

    // getline (ifile, s);
    // jump_size = stoi (s.substr (overhead, s.length() - overhead));

    // Initialise the linearised cube
    data = (double*)malloc(total_points * sizeof(double));

    for (int i = 0; i < total_points; i++){
      getline (ifile, s);
      data[i] = stod (s);
    }

    ifile.close();
    cout << "DUDE, THE CUBE HAS BEEN CREATED PROPERLY!" << endl;
    return true;
  }

  void GEMM_Cube::print_info (){
    printf("· Total number of points: %d\n", total_points);
    for (int i = 0; i < 3; i++){
      printf(">> Axis %d:\n", i + 1);
      printf("\tmin_size: %d\n", axes[i]->min_size);
      printf("\tmax_size: %d\n", axes[i]->max_size);
      printf("\tnpoints: %d\n", axes[i]->npoints);
    }
  }



  // Creating this list of points would be useful when the sampling is not uniform
  // void GEMM_Cube::create_points (){
  //   points = (int*)malloc(npoints * sizeof(int));
  //   for (int i = 0; i < npoints; i++){
  //     points[i] = min_size + i * jump_size;
  //   }
  // }


  bool GEMM_Cube::is_in_cube (const int* dims_o, int* indices) const{
    bool is = true;

    for (int i = 0; i < 3; i++){
      indices[i] = binarySearch (axes[i]->points, 0, axes[i]->npoints, dims_o[i]);
      if (indices[i] == -1){
        is = false;
      }
    }

    return is;
  }


  inline double GEMM_Cube::access_cube (const int *indices) const {
    return data[indices[0] * axes[1]->npoints * axes[2]->npoints +
      indices[1] * axes[2]->npoints + indices[2]];
  }


  void GEMM_Cube::get_ranges (const int *dims_o, const int *indices, int *ranges) const{
    for (int i = 0; i < 3; i++){
      if (indices[i] != -1){
        ranges[2 * i] = indices[i];
        ranges[2 * i + 1] = indices[i];
      }
      else{
        ranges[2 * i] = searchLower(axes[i]->points, axes[i]->npoints, dims_o[i]);
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
        r_distances[i] = double(dims_o[i] - axes[i]->points[ranges[2 * i]]) /
          double(axes[i]->points[ranges[2 * i + 1]] - axes[i]->points[ranges[2 * i]]);
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
    int dims_o[3] = {d1, d2, d3};
    int indices[3] = {-1, -1, -1};
    int ranges[6];

    // Check whether values are out of range; this truncation might be revised
    // in the future.
    for (int i = 0; i < 3; i++){
      if (dims_o[i] > axes[i]->max_size) dims_o[i] = axes[i]->max_size;
      else if (dims_o[i] < axes[i]->min_size) dims_o[i] = axes[i]->min_size;
      printf ("\t >> dims[%d] == %d\n", i, dims_o[i]);
    }
    if (is_in_cube (dims_o, indices)){
      return access_cube(indices);
    }

    // Check how many dimensions we have to interpolate AND EXTRACT THE RANGES
    get_ranges (dims_o, indices, ranges);

    return trilinear_inter(dims_o, ranges);
  }

  // double GEMM_Cube::get_data (const int i) const{
  //   return data[i];
  // }

  // int GEMM_Cube::get_axis_value (const int ax, const int i) const{
  //   return axes[ax]->points[i];
  // }
}



