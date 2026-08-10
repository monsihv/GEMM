#pragma once

#include "base.h"
#include "matrix.h"

typedef Matrix *(*MatMul)(Arena*, const Matrix*, const Matrix*);

Matrix *matMulNaive(Arena *a, const Matrix *m1, const Matrix *m2);
Matrix *matMulCache(Arena *a, const Matrix *m1, const Matrix *m2);
Matrix *matMulBlocking(Arena *a, const Matrix *m1, const Matrix *m2);
Matrix *matMulRegisterBlocking(Arena *a, const Matrix *m1, const Matrix *m2);

void matMulRegisterBlockingHelper(Matrix *output, const Matrix *m1, const Matrix *m2, 
                               u32 iOffset, u32 jOffset, u32 kOffset, 
                               u32 iTile, u32 jTile, u32 kTile);
