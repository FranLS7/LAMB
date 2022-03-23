#include "model.h"

#include <algorithm>
#include <string>
#include <vector>

#include "axis.h"
#include "common.h"

namespace lamb{
Model::~Model() {
  if (data)
    delete[] data;
}

Model::Model(const iVector1D& axis) {
  axes.push_back(axis);
  n_elements = static_cast<int>(axis.size());
}

Model::Model(const iVector2D& axes) {
  this->axes = axes;
  
  n_elements = 1;
  for (const auto& axis : axes) {
    n_elements *= static_cast<int>(axis.size());
  }
}

double Model::operator()(const int d0) const {
  if (!data) 
    return -1.0;

  if (d0 < axes[0][0])
    return data[axes[0][0]];
  else if (d0 > axes[0].back())
    return data[axes[0].back()];
  
  auto it = std::find(axes[0].begin(), axes[0].end(), d0);

  if (it != axes[0].end()) {
    int index = std::distance(axes[0].begin(), it);
    return data[index];
  }
  else {
    int index = findGreater(d0, axes[0]);
    iVector1D coords {axes[0][index - 1], axes[0][index]};
    dVector1D values {data[index - 1], data[index]};
    return linInterpolation(d0, coords, values);
  }
}


double Model::operator()(const int d0, const int d1) const {
  if (!data)
    return -1.0;

  iVector1D dims {d0, d1};

  for (unsigned i = 0; i < axes.size(); ++i) {
    if (dims[i] < axes[i][0])
      dims[i] = axes[i][0];
    else if (dims[i] > axes[i].back())
      dims[i] = axes[i].back();
  }

  iVector1D index;

  for (unsigned i = 0; i < axes.size(); ++i) {
    auto it = std::find(axes[i].begin(), axes[i].end(), dims[i]);
    if (it != axes[i].end())
      index.push_back(std::distance(axes[i].begin(), it));
    else
      index.push_back(-1);
  }

  bool isInModel = true;
  for (const auto& idx : index) {
    if (idx == -1)
      isInModel = false;
  }

  if (isInModel)
    return data[index[0] + index[1] * axes[0].size()];
  else {
    iVector2D coords (2);
    dVector2D z (2);
    for (unsigned i = 0; i < index.size(); ++i) {
      index[i] = findGreater(dims[i], axes[i]);
      coords[i].push_back(axes[i][index[i] - 1]);
      coords[i].push_back(axes[i][index[i]]);
    }

    // @TODO: COMPLETAR LA FUNCION COGIENDO LOS VALORES CORRECTORS DEL MODELO.
    // TIENE QUE SER DATA[ALGO_CON_LOS_INDICES]. TENER EN CUENTA LOS SALTOS EN
    // LAS COORDENADAS.
    return biInterpolation(dims, coords, z);
  }
}

double Model::operator()(const int d0, const int d1, const int d2) const {
    if (!data)
    return -1.0;

  iVector1D dims {d0, d1, d2};

  for (unsigned i = 0; i < axes.size(); ++i) {
    if (dims[i] < axes[i][0])
      dims[i] = axes[i][0];
    else if (dims[i] > axes[i].back())
      dims[i] = axes[i].back();
  }


}

double Model::operator()(const std::vector<int>& dims) const {
  if (dims.size() == 1U)
    return this->operator()(dims[0]);

  else if (dims.size() == 2U)
    return this->operator()(dims[0], dims[1]);
  
  else if (dims.size() == 3U)
    return this->operator()(dims[0], dims[1], dims[2]);
  
  else return -1.0;
}


double Model::linInterpolation(const int dim, const iVector1D& x, const dVector1D& y) const {
  return y[0] + (y[1] - y[0]) * (dim - x[0]) / (x[1] - x[0]);
}

double Model::biInterpolation(const iVector1D& dims, const iVector2D& coords,
    const dVector2D& z) const {
  dVector1D temporary;
  temporary.push_back(linInterpolation(dims[0], coords[0], z[0]));
  temporary.push_back(linInterpolation(dims[0], coords[0], z[1]));

  return linInterpolation(dims[1], coords[1], temporary);
}

double Model::triInterpolation(const iVector1D& dims, const iVector2D& coords,
    const dVector3D w) const {
  dVector1D temporary;
  temporary.push_back(biInterpolation(dims, coords, w[0]));
  temporary.push_back(biInterpolation(dims, coords, w[1]));

  return linInterpolation(dims[2], coords[2], temporary);
}

int Model::findGreater(const int value, const iVector1D& axis) const {
  unsigned index;
  for (unsigned i = 0; i < axis.size(); ++i) {
    if (axis[i] >= value) {
      index = i;
      break;
    }
  }
  return static_cast<int>(index);
}

// bool Model::find(const int value, const iVector1D& axis) const {
//   auto it = std::find(axis.begin(), axis.end(), value);
//   if (it != axis.end())
//     return true;
//   else
//     return false; 
// }

} // end namespace lamb