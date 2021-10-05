#include <iostream>
#include <vector>

#include <common.h>

int main () {
  std::vector<double> v {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  double median = median_vector (v);

  double avg = avg_vector (v);

  std::cout << "the avg is: " << avg << std::endl;

  return 0;
}