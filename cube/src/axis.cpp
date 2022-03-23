#include "axis.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "common.h"


Axis::Axis(const iVector1D& points) {
  if (std::is_sorted(points.begin(), points.end())) {
    this->points = points;
    this->min = points[0];
    this->max = points.back();
  }
  else 
    std::cerr << "Input vector is not sorted\n";
}

int Axis::getMin() const {
  return min;
}

int Axis::getMax() const {
  return max;
}

int Axis::numPoints() const {
  return this->points.size();
}

Axis& Axis::operator=(const iVector1D& points) {
  if (this->points == points)
    return *this;

  if (std::is_sorted(points.begin(), points.end())) {
    this->points = points;
    this->min = points[0];
    this->max = points.back();

    return *this;
  }
  else
    std::cerr << "Input vector is not sorted\n";
}

Axis& Axis::operator=(const Axis& rhs) {
  if (this == &rhs)
    return *this;
  
  points = rhs.points;
  min = rhs.min;
  max = rhs.max;

  return *this;
}