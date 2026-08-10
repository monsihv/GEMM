#include "../headers/matrix.h"
#include "../headers/matmul.h"

Matrix *matMulNaive(Arena *a, const Matrix *m1, const Matrix *m2) {
    Matrix *output = matCreate(a, m1->rows, m2->cols);

    for (u32 i = 0; i < m1->rows; ++i) {
        for (u32 j = 0; j < m2->cols; ++j) {
            for (u32 k = 0; k < m1->cols; ++k) {
                output->data[idx(output, i, j)] += m1->data[idx(m1, i, k)] * m2->data[idx(m2, k, j)];
            }
        }
    }

    return output;
}

Matrix *matMulCache(Arena *a, const Matrix *m1, const Matrix *m2) {
    Matrix *output = matCreate(a, m1->rows, m2->cols);

    for (u32 i = 0; i < m1->rows; ++i) {
        for (u32 k = 0; k < m1->cols; ++k) {
            for (u32 j = 0; j < m2->cols; ++j) {
                output->data[idx(output, i, j)] += m1->data[idx(m1, i, k)] * m2->data[idx(m2, k, j)];
            }
        }
    }

    return output;
}

Matrix *matMulBlocking(Arena *a, const Matrix *m1, const Matrix *m2) {
    Matrix *output = matCreate(a, m1->rows, m2->cols);
    u32 N = m1->rows;
    u32 iTileSize = 64;
    u32 tileSize = 1024;

    if (tileSize > N) {
        tileSize = N;
    }

    u32 iTile = iTileSize;
    u32 kTile = tileSize;
    u32 jTile = tileSize;
    for (u32 iOffset = 0; iOffset < m1->rows; iOffset += iTileSize) {
        for (u32 kOffset = 0; kOffset < m2->rows; kOffset += tileSize) {
            for (u32 jOffset = 0; jOffset < m2->cols; jOffset += tileSize) {
                iTile = iTileSize;
                if (iTileSize > N - iOffset) {
                    iTile = N - iOffset;
                }
                jTile = tileSize;
                if (tileSize > N - jOffset) {
                    jTile = N - jOffset;
                }
                kTile = tileSize;
                if (tileSize > N - kOffset) {
                    kTile = N - kOffset;
                }
                matMulRegisterBlockingHelper(output, m1, m2, 
                                       iOffset, jOffset, kOffset,
                                       iTile, jTile, kTile);
            }
        }
    }

    return output;
}

Matrix *matMulRegisterBlocking(Arena *a, const Matrix *m1, const Matrix *m2) {
    Matrix *output = matCreate(a, m1->rows, m2->cols);
    for (u32 i = 0; i < m1->rows; i += 4) {
        for (u32 j = 0; j < m2->cols; j += 4) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            float sum8 = 0;
            float sum9 = 0;
            float sum10 = 0;
            float sum11 = 0;
            float sum12 = 0;
            float sum13 = 0;
            float sum14 = 0;
            float sum15 = 0;
            float constant0;
            float constant1;
            float constant2;
            float constant3;
            for (u32 k = 0; k < m2->rows; ++k) {
                constant0 = m1->data[idx(m1, i, k)];
                constant1 = m1->data[idx(m1, i + 1, k)];
                constant2 = m1->data[idx(m1, i + 2, k)];
                constant3 = m1->data[idx(m1, i + 3, k)];
                sum0 += constant0 * m2->data[idx(m2, k, j)];
                sum1 += constant0 * m2->data[idx(m2, k, j + 1)];
                sum2 += constant0 * m2->data[idx(m2, k, j + 2)];
                sum3 += constant0 * m2->data[idx(m2, k, j + 3)];
                sum4 += constant1 * m2->data[idx(m2, k, j)];
                sum5 += constant1 * m2->data[idx(m2, k, j + 1)];
                sum6 += constant1 * m2->data[idx(m2, k, j + 2)];
                sum7 += constant1 * m2->data[idx(m2, k, j + 3)];
                sum8 += constant2 * m2->data[idx(m2, k, j)];
                sum9 += constant2 * m2->data[idx(m2, k, j + 1)];
                sum10 += constant2 * m2->data[idx(m2, k, j + 2)];
                sum11 += constant2 * m2->data[idx(m2, k, j + 3)];
                sum12 += constant3 * m2->data[idx(m2, k, j)];
                sum13 += constant3 * m2->data[idx(m2, k, j + 1)];
                sum14 += constant3 * m2->data[idx(m2, k, j + 2)];
                sum15 += constant3 * m2->data[idx(m2, k, j + 3)];
            }
            output->data[idx(output, i, j)] = sum0;
            output->data[idx(output, i, j + 1)] = sum1;
            output->data[idx(output, i, j + 2)] = sum2;
            output->data[idx(output, i, j + 3)] = sum3;
            output->data[idx(output, i + 1, j)] = sum4;
            output->data[idx(output, i + 1, j + 1)] = sum5;
            output->data[idx(output, i + 1, j + 2)] = sum6;
            output->data[idx(output, i + 1, j + 3)] = sum7;
            output->data[idx(output, i + 2, j)] = sum8;
            output->data[idx(output, i + 2, j + 1)] = sum9;
            output->data[idx(output, i + 2, j + 2)] = sum10;
            output->data[idx(output, i + 2, j + 3)] = sum11;
            output->data[idx(output, i + 3, j)] = sum12;
            output->data[idx(output, i + 3, j + 1)] = sum13;
            output->data[idx(output, i + 3, j + 2)] = sum14;
            output->data[idx(output, i + 3, j + 3)] = sum15;
        }
    }

    return output;
}

void matMulRegisterBlockingHelper(Matrix *output, const Matrix *m1, const Matrix *m2, 
                               u32 iOffset, u32 jOffset, u32 kOffset, 
                               u32 iTile, u32 jTile, u32 kTile) {

    for (u32 i = iOffset; i < iOffset + iTile; i += 4) {
        for (u32 j = jOffset; j < jOffset + jTile; j += 4) {
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            float sum4 = 0;
            float sum5 = 0;
            float sum6 = 0;
            float sum7 = 0;
            float sum8 = 0;
            float sum9 = 0;
            float sum10 = 0;
            float sum11 = 0;
            float sum12 = 0;
            float sum13 = 0;
            float sum14 = 0;
            float sum15 = 0;
            float constant0;
            float constant1;
            float constant2;
            float constant3;
            for (u32 k = kOffset; k < kOffset + kTile; ++k) {
                constant0 = m1->data[idx(m1, i, k)];
                constant1 = m1->data[idx(m1, i + 1, k)];
                constant2 = m1->data[idx(m1, i + 2, k)];
                constant3 = m1->data[idx(m1, i + 3, k)];
                sum0 += constant0 * m2->data[idx(m2, k, j)];
                sum1 += constant0 * m2->data[idx(m2, k, j + 1)];
                sum2 += constant0 * m2->data[idx(m2, k, j + 2)];
                sum3 += constant0 * m2->data[idx(m2, k, j + 3)];
                sum4 += constant1 * m2->data[idx(m2, k, j)];
                sum5 += constant1 * m2->data[idx(m2, k, j + 1)];
                sum6 += constant1 * m2->data[idx(m2, k, j + 2)];
                sum7 += constant1 * m2->data[idx(m2, k, j + 3)];
                sum8 += constant2 * m2->data[idx(m2, k, j)];
                sum9 += constant2 * m2->data[idx(m2, k, j + 1)];
                sum10 += constant2 * m2->data[idx(m2, k, j + 2)];
                sum11 += constant2 * m2->data[idx(m2, k, j + 3)];
                sum12 += constant3 * m2->data[idx(m2, k, j)];
                sum13 += constant3 * m2->data[idx(m2, k, j + 1)];
                sum14 += constant3 * m2->data[idx(m2, k, j + 2)];
                sum15 += constant3 * m2->data[idx(m2, k, j + 3)];
            }
            output->data[idx(output, i, j)] += sum0;
            output->data[idx(output, i, j + 1)] += sum1;
            output->data[idx(output, i, j + 2)] += sum2;
            output->data[idx(output, i, j + 3)] += sum3;
            output->data[idx(output, i + 1, j)] += sum4;
            output->data[idx(output, i + 1, j + 1)] += sum5;
            output->data[idx(output, i + 1, j + 2)] += sum6;
            output->data[idx(output, i + 1, j + 3)] += sum7;
            output->data[idx(output, i + 2, j)] += sum8;
            output->data[idx(output, i + 2, j + 1)] += sum9;
            output->data[idx(output, i + 2, j + 2)] += sum10;
            output->data[idx(output, i + 2, j + 3)] += sum11;
            output->data[idx(output, i + 3, j)] += sum12;
            output->data[idx(output, i + 3, j + 1)] += sum13;
            output->data[idx(output, i + 3, j + 2)] += sum14;
            output->data[idx(output, i + 3, j + 3)] += sum15;
        }
    }
}
