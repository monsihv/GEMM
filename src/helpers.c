#include "../headers/helpers.h"

void evaluate(Arena *a, u32 *size, u32 index, 
              Matrix **result, const Matrix *m1, const Matrix *m2, MatMul func,
              struct timespec *thingy, double *results) {

    clock_gettime(CLOCK_MONOTONIC, thingy);
    double before = (double)thingy->tv_nsec / 1000000000 + (double)thingy->tv_sec;

    *result = func(a, m1, m2);

    clock_gettime(CLOCK_MONOTONIC, thingy);
    double after = (double)thingy->tv_nsec / 1000000000 + (double)thingy->tv_sec;

    double timeElapsed = after - before;

    double elements = size[index] * size[index];
    double fmaCount = size[index];

    results[index] = (elements * fmaCount * 2) / (timeElapsed * 1000000000);
}

void printToCSV(const char *fileName, u32 arrayLength, u32 *sizes, 
                double *naive, double *cache, double *blocking, double *SIMD) {

    FILE *csv = fopen(fileName, "w");
    if (!csv) {
        fprintf(stderr, "Couldnt open csv file\n");
        exit(1);
    }

    fprintf(csv, "N, Naive, Cache, Blocking, SIMD\n");
    for (u32 i = 0; i < arrayLength; ++i) {
        fprintf(csv, "%d, %f, %f, ", sizes[i], naive[i], cache[i]);
        if (blocking) {
            fprintf(csv, "%f, ", blocking[i]);
        }
        else {
            fprintf(csv, "%f, ", 0.0);
        }
        if (SIMD) {
            fprintf(csv, "%f", SIMD[i]);
        }
        else {
            fprintf(csv, "%f, ", 0.0);
        }

        fprintf(csv, "\n");
    }

    fclose(csv);
}
