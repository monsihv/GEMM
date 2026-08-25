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
#include "../headers/matrix.h"
#include "../headers/matmul.h"
#include "../headers/helpers.h"

#include "arena.c"
#include "matrix.c"
#include "matmul.c"
#include "helpers.c"

int main() {
    Arena *a = arenaAlloc(GIGABYTE(1));
    Arena *b = arenaAlloc(GIGABYTE(1));

    u32 size[] = {64, 128, 192, 256,
                  320, 384, 448, 512,
                  576, 640, 704, 768,
                  832, 896, 960, 1024,
                  2048, 3072, 4096, 5120};

    u32 length = sizeof(size) / sizeof(u32);

    double *naive = pushArray(a, double, length);
    double *cache =  pushArray(a, double, length);
    double *blocking = pushArray(a, double, length);
    double *SIMD = pushArray(a, double, length);
    struct timespec thingy;

    u32 pos = a->pos;
    u32 posB = b->pos;

    bool correctCheck = true;
    for (u32 i = 0; i < length; ++i) {
        Matrix *m1 = matCreate(b, size[i], size[i]);
        randomElements(m1);
        Matrix *m2 = matCreate(b, size[i], size[i]);
        randomElements(m2);

        //naive
        printf("Starting Naive\n");
        Matrix *resultNaive;
        evaluate(a, size, i, &resultNaive, m1, m2, &matMulNaive, &thingy, naive);
        arenaPopTo(a, pos);

        //blocking
        printf("Starting Blocking\n");
        Matrix *resultBlocking;
        evaluate(a, size, i, &resultBlocking, m1, m2, &matMulBlocking, &thingy, blocking);
        arenaPopTo(a, pos);

        //cache
        printf("Starting Cache\n");
        Matrix *resultCache;
        evaluate(a, size, i, &resultCache, m1, m2, &matMulCache, &thingy, cache);

        if (!correctCheck) {
            arenaPopTo(a, pos);
        }

        //SIMD
        printf("Starting SIMD\n");
        Matrix *resultSIMD;
        evaluate(a, size, i, &resultSIMD, m1, m2, &matMulBlockingSIMD, &thingy, SIMD);

        if (!correctCheck) {
            arenaPopTo(a, pos);
        }

        //clear matrix A & B
        arenaPopTo(b, posB);

        //matrix checks
        if (correctCheck) {
            u32 trueCount = 0;
            if (checkMatricesEqual(resultCache, resultSIMD, &trueCount)) {
                printf("size: %d, %d\n", size[i], 1);
            }
            else {
                printf("size: %d, %d\n", size[i], 0);
            }

            u32 lastIndex = size[i] * size[i] - 1;
            printf("cache: %f, blocking: %f\n", resultCache->data[0], resultSIMD->data[0]);
            printf("cache: %f, blocking: %f\n", resultCache->data[lastIndex], resultSIMD->data[lastIndex]);
        }

        //clear results
        arenaPopTo(a, pos);
    }

    printToCSV("results.csv", length, size, naive, cache, blocking, SIMD);

    clearArena(a);
    clearArena(b);
    arenaFree(a);
    arenaFree(b);

    return 0;
}
