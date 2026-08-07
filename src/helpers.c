#include "../headers/helpers.h"

void printToCSV(const char *fileName, u32 arrayLength, u32 *sizes, 
                double *naive, double *cache, double *blocking, double *SIMD) {

    FILE *csv = fopen(fileName, "w");
    if (!csv) {
        fprintf(stderr, "Couldnt open csv file\n");
        exit(1);
    }

    fprintf(csv, "#MATRIX MULTIPLICATION PERFORMANCE OPTIMIZATION (GFLOPS)\n");
    fprintf(csv, "\n");
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
