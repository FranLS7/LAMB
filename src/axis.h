#ifndef AXIS_CLASS
#define AXIS_CLASS

#include <vector>

#include "common.h"

class Axis{

iVector1D points;
int min;
int max;

public:

  Axis() = default;

  ~Axis() = default;

  Axis(const iVector1D& points);

  int getMin() const;

  int getMax() const;

  int numPoints() const;

  Axis& operator=(const iVector1D& points);

  Axis& operator=(const Axis& rhs);
};


#endif