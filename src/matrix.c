#include "../headers/matrix.h"

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
