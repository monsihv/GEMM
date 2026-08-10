#pragma once

#include "base.h"
#include "arena.h"
#include "matrix.h"
#include "matmul.h"

void evaluate(Arena *a, u32 *size, u32 index, 
              Matrix **result, const Matrix *m1, const Matrix *m2, MatMul func,
              struct timespec *thingy, double *results);

void printToCSV(const char *fileName, u32 arrayLength, u32 *sizes, 
                double *naive, double *cache, double *blocking, double *SIMD);
