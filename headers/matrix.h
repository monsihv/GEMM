#pragma once

#include "arena.h"

typedef struct {
    float *data;
    u32 rows, cols;

} Matrix;

#define idx(m, i, j) ((i) * (m)->cols + (j))

Matrix *matCreate(Arena *a, u32 rows, u32 cols);
void randomElements(Matrix *m);
bool checkMatricesEqual(const Matrix *m1, const Matrix *m2, u32 *failedIndex);
