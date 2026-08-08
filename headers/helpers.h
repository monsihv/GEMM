#pragma once

#include "base.h"

void printToCSV(const char *fileName, u32 arrayLength, u32 *sizes, 
                double *naive, double *cache, double *blocking, double *SIMD);
