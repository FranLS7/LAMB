#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "common.h"
#include "operation.h"

#include "mkl.h"

void printMatrix (const Matrix& matrix);

int main() {
  Matrix A;
  A.name = "A";
  A.rows = 3;
  A.columns = 4;
  A.data = new double[A.rows * A.columns];

  for (int i = 0; i < A.rows * A.columns; ++i) A.data[i] = static_cast<double>(i);
  printMatrix(A);

  Matrix B;
  B.name = "B";
  B.rows = A.columns;
  B.columns = 2;
  B.data = new double[B.rows * B.columns];

  for (int i = 0; i < B.rows * B.columns; ++i) B.data[i] = static_cast<double>(i * i - 4 * i);
  printMatrix(B);

  Matrix X;
  X.name = "X";
  X.rows = A.rows;
  X.columns = B.columns;
  X.data = new double[X.rows * X.columns];
  for (int i = 0; i < X.rows * X.columns; ++i) X.data[i] = 0.0;
  printMatrix(X);

  
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows, B.columns, A.columns, 1.0,
      A.data, A.rows, B.data, B.rows, 0.0, X.data, X.rows);

  printMatrix(X);





  delete[] A.data;
  delete[] B.data;
  delete[] X.data;

  return 0;
}


void printMatrix (const Matrix& matrix) {
  std::cout << "================= " << matrix.name << " =================\n";
  for (int i = 0; i < matrix.rows; ++i) {
    for (int j = 0; j < matrix.columns; ++j) {
      std::cout << std::setprecision(10) << matrix.data[j * matrix.rows + i];
      if (j == matrix.columns - 1) std::cout << '\n';
      else std::cout << '\t';
    }
  }
  std::cout << "=====================================\n\n";
}