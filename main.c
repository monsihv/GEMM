//I am on apple silicon. I used references from https://github.com/flame/how-to-optimize-gemm/wiki
//
//Their material is on intel processor but it was more so used for conceptual understanding.
//
//The main purpose of this project is to optimize Matrix Matrix multiplication so my
//deep learning library from scratch can train and inference faster for both MLPs and CNNs.
//
//This project will be done in C, tho if I ever were to rewrite my work in C++, the ideas should cleanly
//transfer.

#include "base.h"
#include "arena.h"

#include "arena.c"

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

int main() {
    Arena *a = arenaAlloc(GIGABYTE(1));

    u32 size[] = {64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024};
    u32 length = sizeof(size) / sizeof(u32);
    Matrix *m[length];
    Matrix *m1[length];

    for (u32 i = 0; i < length; ++i) {
        m[i] = matCreate(a, size[i], size[i]);
        randomElements(m[i]);
        m1[i] = matCreate(a, size[i], size[i]);
        randomElements(m1[i]);
    }

    double naive[length];
    double cache[length];
    double before, after, timeElapsed;
    struct timespec thingy;

    double opPerFma = 2.0;
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

    for (u32 i = 0; i < length; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &thingy);
        before = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        Matrix *result = matMulCache(a, m[i], m1[i]);

        clock_gettime(CLOCK_MONOTONIC, &thingy);
        after = (double)thingy.tv_nsec / 1000000000 + (double)thingy.tv_sec;

        timeElapsed = after - before;

        double elements = size[i] * size[i];
        double fmaCount = size[i];

        cache[i] = (elements * fmaCount * opPerFma) / (timeElapsed * 1000000000); 
    }

    printf("N, naive, cache\n");
    for (u32 i = 0; i < length; ++i) {
        printf("%d, %f, %f\n", size[i], naive[i], cache[i]);
    }

    return 0;
}
