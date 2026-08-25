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

Matrix *matMulBlockingSIMD(Arena *a, const Matrix *m1, const Matrix *m2) {
    Matrix *output = matCreate(a, m1->rows, m2->cols);
    u32 N = m1->rows;
    u32 iTileSize = 64;
    u32 kTileSize = 128;
    u32 jTileSize = 128;

    if (iTileSize > N) {
        iTileSize = N;
    }
    if (kTileSize > N) {
        kTileSize = N;
    }
    if (jTileSize > N) {
        jTileSize = N;
    }

    u32 iTile = iTileSize;
    u32 kTile = kTileSize;
    u32 jTile = jTileSize;
    for (u32 jOffset = 0; jOffset < m2->cols; jOffset += jTileSize) {
        for (u32 kOffset = 0; kOffset < m2->rows; kOffset += kTileSize) {
            for (u32 iOffset = 0; iOffset < m1->rows; iOffset += iTileSize) {
                iTile = iTileSize;
                if (iTileSize > N - iOffset) {
                    iTile = N - iOffset;
                }
                jTile = jTileSize;
                if (jTileSize > N - jOffset) {
                    jTile = N - jOffset;
                }
                kTile = kTileSize;
                if (kTileSize > N - kOffset) {
                    kTile = N - kOffset;
                }
                matMulVecHelper(output, m1, m2, 
                                       iOffset, jOffset, kOffset,
                                       iTile, jTile, kTile);
            }
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

void matMulVecHelper(Matrix *output, const Matrix *m1, const Matrix *m2, 
                               u32 iOffset, u32 jOffset, u32 kOffset, 
                               u32 iTile, u32 jTile, u32 kTile) {

    for (u32 i = iOffset; i < iOffset + iTile; i += 4) {
        for (u32 j = jOffset; j < jOffset + jTile; j += 16) {
            float constant0;
            float constant1;
            float constant2;
            float constant3;

            float *c0 = &output->data[idx(output, i, j)];
            float *c1 = &output->data[idx(output, i + 1, j)];
            float *c2 = &output->data[idx(output, i + 2, j)];
            float *c3 = &output->data[idx(output, i + 3, j)];
            float *c4 = &output->data[idx(output, i, j + 4)];
            float *c5 = &output->data[idx(output, i + 1, j + 4)];
            float *c6 = &output->data[idx(output, i + 2, j + 4)];
            float *c7 = &output->data[idx(output, i + 3, j + 4)];
            float *c8 = &output->data[idx(output, i, j + 8)];
            float *c9 = &output->data[idx(output, i + 1, j + 8)];
            float *c10 = &output->data[idx(output, i + 2, j + 8)];
            float *c11 = &output->data[idx(output, i + 3, j + 8)];
            float *c12 = &output->data[idx(output, i, j + 12)];
            float *c13 = &output->data[idx(output, i + 1, j + 12)];
            float *c14 = &output->data[idx(output, i + 2, j + 12)];
            float *c15 = &output->data[idx(output, i + 3, j + 12)];
            float32x4_t cv0 = vld1q_f32(c0);
            float32x4_t cv1 = vld1q_f32(c1);
            float32x4_t cv2 = vld1q_f32(c2);
            float32x4_t cv3 = vld1q_f32(c3);
            float32x4_t cv4 = vld1q_f32(c4);
            float32x4_t cv5 = vld1q_f32(c5);
            float32x4_t cv6 = vld1q_f32(c6);
            float32x4_t cv7 = vld1q_f32(c7);
            float32x4_t cv8 = vld1q_f32(c8);
            float32x4_t cv9 = vld1q_f32(c9);
            float32x4_t cv10 = vld1q_f32(c10);
            float32x4_t cv11 = vld1q_f32(c11);
            float32x4_t cv12 = vld1q_f32(c12);
            float32x4_t cv13 = vld1q_f32(c13);
            float32x4_t cv14 = vld1q_f32(c14);
            float32x4_t cv15 = vld1q_f32(c15);
            for (u32 k = kOffset; k < kOffset + kTile; ++k) {
                constant0 = m1->data[idx(m1, i, k)];
                constant1 = m1->data[idx(m1, i + 1, k)];
                constant2 = m1->data[idx(m1, i + 2, k)];
                constant3 = m1->data[idx(m1, i + 3, k)];

                float *b = &m2->data[idx(m2, k, j)];
                float32x4_t bv0 = vld1q_f32(b);
                float32x4_t bv1 = vld1q_f32(b + 4);
                float32x4_t bv2 = vld1q_f32(b + 8);
                float32x4_t bv3 = vld1q_f32(b + 12);

                cv0 = vfmaq_n_f32(cv0, bv0, constant0);
                cv1 = vfmaq_n_f32(cv1, bv0, constant1);
                cv2 = vfmaq_n_f32(cv2, bv0, constant2);
                cv3 = vfmaq_n_f32(cv3, bv0, constant3);
                cv4 = vfmaq_n_f32(cv4, bv1, constant0);
                cv5 = vfmaq_n_f32(cv5, bv1, constant1);
                cv6 = vfmaq_n_f32(cv6, bv1, constant2);
                cv7 = vfmaq_n_f32(cv7, bv1, constant3);
                cv8 = vfmaq_n_f32(cv8, bv2, constant0);
                cv9 = vfmaq_n_f32(cv9, bv2, constant1);
                cv10 = vfmaq_n_f32(cv10, bv2, constant2);
                cv11 = vfmaq_n_f32(cv11, bv2, constant3);
                cv12 = vfmaq_n_f32(cv12, bv3, constant0);
                cv13 = vfmaq_n_f32(cv13, bv3, constant1);
                cv14 = vfmaq_n_f32(cv14, bv3, constant2);
                cv15 = vfmaq_n_f32(cv15, bv3, constant3);
            }

            vst1q_f32(c0, cv0);
            vst1q_f32(c1, cv1);
            vst1q_f32(c2, cv2);
            vst1q_f32(c3, cv3);
            vst1q_f32(c4, cv4);
            vst1q_f32(c5, cv5);
            vst1q_f32(c6, cv6);
            vst1q_f32(c7, cv7);
            vst1q_f32(c8, cv8);
            vst1q_f32(c9, cv9);
            vst1q_f32(c10, cv10);
            vst1q_f32(c11, cv11);
            vst1q_f32(c12, cv12);
            vst1q_f32(c13, cv13);
            vst1q_f32(c14, cv14);
            vst1q_f32(c15, cv15);
        }
    }
}
