#include <iostream>
#include <cstdio>
#include <vector>

typedef uint32_t u32;

struct Matrix {
    std::vector<float> data;
    u32 rows, cols;

    Matrix(u32 rows, u32 cols) : rows{rows}, cols{cols} {
        this->data.resize(rows * cols);
    }
};

int main() {

    return 0;
}
