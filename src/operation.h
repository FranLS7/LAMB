/**
 * @file operation.h
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief header file with algorithms to solve A*trans(A)*B
 * @version 0.1
 * @date 2022-01-27
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __OPERATION_FUNC__
#define __OPERATION_FUNC__

#include "common.h"


dVector1D syrkAndSymm(const Matrix& A, const Matrix& B, Matrix& X);

dVector1D syrkAndGemm(const Matrix& A, const Matrix& B, Matrix& X);

dVector1D gemmAndSymm(const Matrix& A, const Matrix& B, Matrix& X);

dVector1D lgemmAndGemm(const Matrix& A, const Matrix& B, Matrix& X);

dVector1D rgemmAndGemm(const Matrix& A, const Matrix& B, Matrix& X);

unsigned long flopsSyrkAndSymm(const Matrix& A, const Matrix& B);
unsigned long flopsSyrkAndGEMM(const Matrix& A, const Matrix& B);
unsigned long flopsGemmAndSymm(const Matrix& A, const Matrix& B);
unsigned long flopsLGemmAndGemm(const Matrix& A, const Matrix& B);
unsigned long flopsRGemmAndGemm(const Matrix& A, const Matrix& B);

dVector2D executeAll(const int m, const int k, const int n);

dVector2D executeAll(const iVector1D& dims);

std::vector<unsigned long> flopsAll(const int m, const int k, const int n);

std::vector<unsigned long> flopsAll(const iVector1D& dims);

#endif