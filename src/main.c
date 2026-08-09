//I am on apple silicon. I used references from CSAPP and google for optimizing GEMM
//
//Their material is on intel processor but it was more so used for conceptual understanding.
//
//The main purpose of this project is to optimize Matrix Matrix multiplication so my
//deep learning library from scratch can train and inference faster for both MLPs and CNNs.
//
//This project will be done in C, tho if I ever were to rewrite my work in C++, the ideas should cleanly
//transfer.

#include "../headers/base.h"
#include "../headers/arena.h"
#include "../headers/helpers.h"

#include "arena.c"
#include "helpers.c"

// Stick with classic matrices for this project
typedef struct {
    float *data;
    u32 rows, cols;

} Matrix;

#define idx(m, i, j) ((i) * (m)->cols + (j))

Matrix *matCreate(Arena *a, u32 rows, u32 cols) {
    Matrix *output = pushStruct(a, Matrix);

    output->rows = rows;
    output->cols = cols;
    output->data = pushArray(a, float, rows * cols);

    memset(output->data, 0, sizeof(float) * rows * cols);

    return output;
}

void randomElements(Matrix *m) {
    for (u32 i = 0; i < m->rows * m->cols; ++i) {
        m->data[i] = (float)rand() / (RAND_MAX + 1.0);
    }
}

bool checkMatricesEqual(const Matrix *m1, const Matrix *m2, u32 *failedIndex) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        fprintf(stderr, "Matrices incorrect dimensions for checking equality\n");
        exit(1);
    }

    float epsilon = 0.01f;
    bool equal = true;
    for (u32 i = 0; i < m1->rows * m2->cols; ++i) {
        bool epsilonCheck1 = (m1->data[i] < m2->data[i] + epsilon) && (m1->data[i] > m2->data[i] - epsilon);
        bool epsilonCheck2 = (m2->data[i] < m1->data[i] + epsilon) && (m2->data[i] > m1->data[i] - epsilon);
        if (!(epsilonCheck1 || epsilonCheck2)) {
            equal = false;
            *failedIndex = i;
        }
    }

    return equal;
}

Matrix *matMulNaive(Arena *a, Matrix *m1, Matrix *m2) {
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

Matrix *matMulCache(Arena *a, Matrix *m1, Matrix *m2) {
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

Matrix *matMulBlocking(Arena *a, Matrix *m1, Matrix *m2) {
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

int main() {
    Arena *a = arenaAlloc(GIGABYTE(1));
    Arena *b = arenaAlloc(GIGABYTE(1));

    u32 size[] = {64, 128, 192, 256,
                  320, 384, 448, 512,
                  576, 640, 704, 768,
                  832, 896, 960, 1024,
                  2048, 3072, 4096, 5120};

    u32 length = sizeof(size) / sizeof(u32);

    double naive[length];
    double cache[length];
    double blocking[length];
    double before, after, timeElapsed;
    struct timespec thingy;
    double opPerFma = 2.0;

    u32 pos = a->pos;
    u32 posB = b->pos;

    for (u32 i = 0; i < length; ++i) {
        Matrix *m1 = matCreate(b, size[i], size[i]);
        randomElements(m1);
        Matrix *m2 = matCreate(b, size[i], size[i]);
        randomElements(m2);

        //naive
        printf("Starting Naive\n");
        clock_gettime(CLOCK_MONOTONIC, &thingy);
        before = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        Matrix *result = matMulNaive(a, m1, m2);

        clock_gettime(CLOCK_MONOTONIC, &thingy);
        after = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        timeElapsed = after - before;

        double elements = size[i] * size[i];
        double fmaCount = size[i];

        naive[i] = (elements * fmaCount * opPerFma) / (timeElapsed * 1000000000); 

        arenaPopTo(a, pos);

        //cache
        printf("Starting Cache\n");
        clock_gettime(CLOCK_MONOTONIC, &thingy);
        before = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        Matrix *resultsCache = matMulCache(a, m1, m2);

        clock_gettime(CLOCK_MONOTONIC, &thingy);
        after = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        timeElapsed = after - before;

        cache[i] = (elements * fmaCount * opPerFma) / (timeElapsed * 1000000000); 

        //blocking
        printf("Starting Blocking\n");
        clock_gettime(CLOCK_MONOTONIC, &thingy);
        before = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        Matrix *resultsBlocking = matMulBlocking(a, m1, m2);

        clock_gettime(CLOCK_MONOTONIC, &thingy);
        after = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        timeElapsed = after - before;

        blocking[i] = (elements * fmaCount * opPerFma) / (timeElapsed * 1000000000); 

        //SIMD

        //clear matrix A & B
        arenaPopTo(b, pos);

        //matrix checks
        u32 trueCount = 0;
        if (checkMatricesEqual(resultsCache, resultsBlocking, &trueCount)) {
            printf("size: %d, %d\n", size[i], 1);
        }
        else {
            printf("size: %d, %d\n", size[i], 0);
        }

        u32 lastIndex = size[i] * size[i] - 1;
        printf("cache: %f, blocking: %f\n", resultsCache->data[0], resultsBlocking->data[0]);
        printf("cache: %f, blocking: %f\n", resultsCache->data[lastIndex], resultsBlocking->data[lastIndex]);

        //clear results
        arenaPopTo(a, pos);
    }

    clearArena(a);
    clearArena(b);
    arenaFree(a);
    arenaFree(b);

    printToCSV("results.csv", length, size, naive, cache, blocking, NULL);

    return 0;
}
