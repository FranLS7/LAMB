#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cube_old.h"

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

int searchLower (std::vector<int> v, int value) {
  int idx = 0;
  for (idx = 0; idx < static_cast<int>(v.size()) - 1; idx++)
    if (v[idx + 1] > value) break;
  return idx;
}

// Compute points in the form of a parabola: y = ax2 + bx + c
// This parabola has its vertex in x=0 --> 'b' = 0.
// For the same reason --> 'c' = min_size.
// 'a' is left to compute, then.
void compute_points (cube::Axis axis) {
  double a = (axis.max_size - axis.min_size) / pow(axis.npoints - 1, 2.0);

  for (int i = 0; i < axis.npoints; i++){
    axis.points.push_back (int(a * pow(i, 2.0)) + axis.min_size);
    if (i > 0 && axis.points[i] <= axis.points[i - 1])
      axis.points[i] = axis.points[i - 1] + 1;
  }
}

} // namespace helper

namespace cube{
GEMM_Cube::GEMM_Cube () {}

GEMM_Cube::GEMM_Cube (std::string filename, const int overhead){
  for (int i = 0; i < 3; i++)
    axes.push_back(Axis());
  if (!load_cube(filename, overhead)) {
    std::cout << "Error loading the cube" << '\n';
    exit(-1);
  }
}


bool GEMM_Cube::load_cube (std::string filename, const int overhead){
  std::ifstream ifile;
  ifile.open (filename, std::ifstream::in);
  total_points = 1;

  if (ifile.fail()){
    std::cout << "Error opening the gemm_cube file" << '\n';
    return false;
  }

  std::string s;

  // Load headers - cube information
  for (int i = 0; i < 3; i++){
    getline (ifile, s);
    axes[i].min_size =  std::stoi (s.substr (overhead, s.length() - overhead));
    getline (ifile, s);
    axes[i].max_size = std::stoi (s.substr (overhead, s.length() - overhead));
    getline (ifile, s);
    axes[i].npoints = std::stoi (s.substr (overhead, s.length() - overhead));
    // axes[i].points = (int*)malloc(axes[i]->npoints * sizeof(int));

    compute_points (axes[i]);
    // for (int kk = 0; kk < axes[i]->npoints; kk++){
    //   printf("Value [%d, %d]: %d\n", i, kk, axes[i]->points[kk]);
    // }
    total_points *= axes[i].npoints;
  }

  // Initialise the linearised cube
  data.resize(total_points);

  for (int i = 0; i < total_points; i++){
    getline (ifile, s);
    data[i] = std::stod(s);
  }

  ifile.close();
  std::cout << "DUDE, THE CUBE HAS BEEN CREATED PROPERLY!" << '\n';
  return true;
}

void GEMM_Cube::print_info (){
  std::cout << "· Total number of points: " << data.size() << '\n';
  for (auto& ax : axes) {
    std::cout << "\tmin_size: " << ax.min_size << '\n';
    std::cout << "\tmax_size: " << ax.max_size << '\n';
    std::cout << "\tnpoints: " << ax.npoints << '\n' << '\n';
  }
}


bool GEMM_Cube::find (std::vector<int> dims_o, std::vector<int> indices) const {
  bool is = true;

  for (unsigned i = 0; i < dims_o.size(); i++) {
    auto it = std::find (axes[i].points.begin(), axes[i].points.end(), dims_o[i]);
    indices[i] = (it != axes[i].points.end()) ? (it - axes[i].points.begin()) : -1;
  }
  return is;
}


inline double GEMM_Cube::access_cube (std::vector<int> indices) const {
  return data[indices[0] * axes[1].npoints * axes[2].npoints +
    indices[1] * axes[2].npoints + indices[2]];
}

std::vector<std::vector<int>> GEMM_Cube::get_ranges (std::vector<int> dims_o, 
    std::vector<int> indices) const {
  std::vector<std::vector<int>> ranges;
  ranges.resize (indices.size());

  for (unsigned i = 0; i < indices.size(); i++) {
    if (indices[i] != -1) {
      ranges[i].push_back(indices[i]);
      ranges[i].push_back(indices[i]);
    }
    else {
      ranges[i].push_back(searchLower (axes[i].points, dims_o[i]));
      ranges[i].push_back(ranges[i][0] + 1);
    }
  }
  return ranges;
}


// Trilinear interpolation with uniform sampling
double GEMM_Cube::trilinear_inter (std::vector<int> dims_o, 
    std::vector<std::vector<int>> ranges) const {
  std::vector<double> r_distances;
  std::vector<double> values_lattice;
  
  for (unsigned i = 0; i < ranges.size(); i++){
    if (ranges[i].front() == ranges[i].back())
      r_distances.push_back(0.0f);
    else
      r_distances.push_back (double(dims_o[i] - axes[i].points[ranges[i].front()]) / 
          double(axes[i].points[ranges[i].back()] - axes[i].points[ranges[i].front()]));
  }

  std::vector<int> indices;
  indices.resize (3);
  int aux = 0;

  for (auto &x : ranges[0]){
    indices[0] = x;
    for (auto &y : ranges[1]){
      indices[1] = y;
      for (auto &z : ranges[2]){
        indices[2] = z;
        values_lattice[aux] = access_cube(indices);
        aux++;
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
  std::vector<int> dims_o {d1, d2, d3};
  std::vector<int> indices {-1, -1, -1};
  std::vector<std::vector<int>> ranges;

  // Check whether values are out of range; this truncation might be revised
  // in the future.
  for (unsigned i = 0; i < dims_o.size(); i++) {
    if (dims_o[i] > axes[i].max_size)
      dims_o[i] = axes[i].max_size;
    else if (dims_o[i] < axes[i].min_size)
      dims_o[i] = axes[i].min_size;
    std::cout << "\t >> dims[" << i << "] == " << dims_o[i] << '\n';
  }

  if (GEMM_Cube::find (dims_o, indices)) return access_cube (indices);

  // Check how many dimensions we have to interpolate AND EXTRACT THE RANGES
  ranges = get_ranges (dims_o, indices);

  return trilinear_inter(dims_o, ranges);
}

} // namespace cube
