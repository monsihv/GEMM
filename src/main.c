//I am on apple silicon. I used references from https://github.com/flame/how-to-optimize-gemm/wiki
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

    bool equal = true;
    for (u32 i = 0; i < m1->rows * m2->cols; ++i) {
        if (m1->data[i] != m2->data[i]) {
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

Matrix *matMulBlocking(Arena *a, Matrix *m1, Matrix *m2) {
    Matrix *output = matCreate(a, m1->rows, m2->cols);
    u32 tileSize = 192;
    u32 N = m1->rows;

    if (tileSize > N) {
        tileSize = N;
    }

    u32 iTile = tileSize;
    u32 kTile = tileSize;
    u32 jTile = tileSize;
    for (u32 iOffset = 0; iOffset < m1->rows; iOffset += tileSize) {
        for (u32 kOffset = 0; kOffset < m2->rows; kOffset += tileSize) {
            for (u32 jOffset = 0; jOffset < m2->cols; jOffset += tileSize) {
                iTile = tileSize;
                if (tileSize > N - iOffset) {
                    iTile = N - iOffset;
                }
                for (u32 i = iOffset; i < iOffset + iTile; ++i) {
                    kTile = tileSize;
                    if (tileSize > N - kOffset) {
                        kTile = N - kOffset;
                    }
                    for (u32 k = kOffset; k < kOffset + kTile; ++k) {
                        jTile = tileSize;
                        if (tileSize > N - jOffset) {
                            jTile = N - jOffset;
                        }
                        for (u32 j = jOffset; j < jOffset + jTile; ++j) {
                            output->data[idx(output, i, j)] += 
                            m1->data[idx(m1, i, k)] * m2->data[idx(m2, k, j)];
                        }
                    }
                }
            }
        }
    }

    return output;
}

int main() {
    Arena *a = arenaAlloc(GIGABYTE(1));

    u32 size[] = {64, 128, 192, 256,
                  320, 384, 448, 512,
                  576, 640, 704, 768,
                  832, 896, 960, 1024,
                  2048};

    u32 length = sizeof(size) / sizeof(u32);
    Matrix *m[length];
    Matrix *m1[length];
    Matrix **resultsCache = pushArray(a, Matrix*, length);
    Matrix **resultsBlocking = pushArray(a, Matrix*, length);

    for (u32 i = 0; i < length; ++i) {
        m[i] = matCreate(a, size[i], size[i]);
        randomElements(m[i]);
        m1[i] = matCreate(a, size[i], size[i]);
        randomElements(m1[i]);
    }

    double naive[length];
    double cache[length];
    double blocking[length];
    double before, after, timeElapsed;
    struct timespec thingy;
    double opPerFma = 2.0;

    u32 pos = a->pos;

    for (u32 i = 0; i < length; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &thingy);
        before = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        Matrix *result = matMulNaive(a, m[i], m1[i]);

        clock_gettime(CLOCK_MONOTONIC, &thingy);
        after = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        timeElapsed = after - before;

        double elements = size[i] * size[i];
        double fmaCount = size[i];

        naive[i] = (elements * fmaCount * opPerFma) / (timeElapsed * 1000000000); 
    }

    arenaPopTo(a, pos);

    for (u32 i = 0; i < length; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &thingy);
        before = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        resultsCache[i] = matMulCache(a, m[i], m1[i]);

        clock_gettime(CLOCK_MONOTONIC, &thingy);
        after = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        timeElapsed = after - before;

        double elements = size[i] * size[i];
        double fmaCount = size[i];

        cache[i] = (elements * fmaCount * opPerFma) / (timeElapsed * 1000000000); 
    }

    for (u32 i = 0; i < length; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &thingy);
        before = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        resultsBlocking[i] = matMulBlocking(a, m[i], m1[i]);

        clock_gettime(CLOCK_MONOTONIC, &thingy);
        after = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        timeElapsed = after - before;

        double elements = size[i] * size[i];
        double fmaCount = size[i];

        blocking[i] = (elements * fmaCount * opPerFma) / (timeElapsed * 1000000000); 
    }

    u32 trueCount = 0;
    u32 failedIndex = 0;
    for (u32 i = 0; i < length; ++i) {
        if (checkMatricesEqual(resultsCache[i], resultsBlocking[i], &failedIndex)) {
            ++trueCount;
            printf("size: %d, %d\n", size[i], 1);
        }
        else {
            printf("size: %d, %d\n", size[i], 0);
        }
    }

    if (trueCount != length) {
        fprintf(stderr, "Matrix dont match, trueCount: %d, length: %d\n", trueCount, length);
    }

    clearArena(a);
    arenaFree(a);

    printToCSV("results.csv", length, size, naive, cache, blocking, NULL);

    return 0;
}
