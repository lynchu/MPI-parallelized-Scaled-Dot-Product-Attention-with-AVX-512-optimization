#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// NOTE: feel free to include any header you need, but we will not
// link libraries other than C's math library for you.

// NOTE: feel free to add new macros

// NOTE: feel free to add new functions

/*
 * Q: m by dk
 * K: n by dk
 * V: n by dv
 * result: m by dv, containing the attention result
 */
void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv) {

    double scale = 1.0 / sqrt((double)dk);

    // Temporary buffer for attention scores per row (length n)
    double* scores = (double*) malloc(sizeof(double) * n);

    for (int i = 0; i < m; i++) {

        /* -----------------------------
         * 1. Compute scores[i][j] = dot(Q[i], K[j]) * scale
         * ----------------------------- */
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            int qi_base = i * dk;
            int kj_base = j * dk;

            for (int t = 0; t < dk; t++) {
                dot += Q[qi_base + t] * K[kj_base + t];
            }
            scores[j] = dot * scale;
        }

        /* -----------------------------
         * 2. Softmax(scores)
         * ----------------------------- */
        double max_val = scores[0];
        for (int j = 1; j < n; j++) {
            if (scores[j] > max_val) max_val = scores[j];
        }

        double sum_exp = 0.0;
        for (int j = 0; j < n; j++) {
            scores[j] = exp(scores[j] - max_val);
            sum_exp += scores[j];
        }
        for (int j = 0; j < n; j++) {
            scores[j] /= sum_exp;
        }

        /* -----------------------------
         * 3. result[i] = scores * V
         * result[i][d] = Σ_j scores[j] * V[j][d]
         * ----------------------------- */
        for (int d = 0; d < dv; d++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += scores[j] * V[j * dv + d];
            }
            result[i * dv + d] = sum;
        }
    }

    free(scores);
}


// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 115 <template code>) <(tail -n 115 <your code>)`

// ----------------------------- You shall not pass! ----------------------------- //

void read_matrix(double** M, size_t len, FILE* file) {
    *M = (double*) malloc(len * sizeof(double));
    if (fread(*M, sizeof(double), len, file) != len) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }
}

/*
 * Reads Q, K, and V matrices from the testing data file
 * File format:
 *   1. 4 integers: m, n, dk, dv
 *   2. m*dk doubles -> Q
 *   3. n*dk doubles -> K
 *   4. n*dv doubles -> V
 */
void read_matrices(const char* file_path, double** Q, double** K, double** V,
                  int *m, int *n, int *dk, int *dv) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", file_path);
        exit(1);
    }

    if (fread(m, sizeof(int), 1, file) != 1 ||
        fread(n, sizeof(int), 1, file) != 1 ||
        fread(dk, sizeof(int), 1, file) != 1 ||
        fread(dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    read_matrix(Q, (*m) * (*dk), file);
    read_matrix(K, (*n) * (*dk), file);
    read_matrix(V, (*n) * (*dv), file);

    fclose(file);
}

bool verify(const char* file_path, const double* result) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open answer file: %s\n", file_path);
        return false;
    }

    int m, n, dk, dv;
    if (fread(&m, sizeof(int), 1, file) != 1 ||
        fread(&n, sizeof(int), 1, file) != 1 ||
        fread(&dk, sizeof(int), 1, file) != 1 ||
        fread(&dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    int offset = sizeof(int) * 4 + sizeof(double) * (m * dk + n * dk + n * dv);
    fseek(file, offset, SEEK_SET);

    bool res = true;
    double threshold = 0.02;
    double* row = (double*) malloc(sizeof(double) * dv);

    for (int i = 0; i < m; i++) {
        int base = i * dv;
        fread(row, sizeof(double), dv, file);
        for (int j = 0; j < dv; j++) {
            if (isnan(result[base + 1]) || fabs(result[base + j] - row[j]) > threshold) {
                printf("Expect result[%d][%d] to be %lf, but it is %lf\n", i, j, row[j], result[base + j]);
                res = false;
                goto end;
            }
        }
    }

end:
    free(row);
    fclose(file);
    return res;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <testing data>\n", argv[0]);
        return 1;
    }

    double* Q = NULL;
    double* K = NULL;
    double* V = NULL;
    double* result = NULL;
    int m, n, dk, dv;

    read_matrices(argv[1], &Q, &K, &V, &m, &n, &dk, &dv);
    result = malloc(sizeof(double) * m * dv);

    struct timespec beg, end;
    clock_gettime(CLOCK_MONOTONIC, &beg);
    attention(Q, K, V, result, m, n, dk, dv);
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (verify(argv[1], result)) {
        double elapsed_time = (end.tv_sec - beg.tv_sec) * 1e6 + (end.tv_nsec - beg.tv_nsec) / 1e3;
        printf("Correct!\nElapsed time: %.2lf us\n", elapsed_time);
    } else {
        puts("Wrong!");
    }

    free(Q);
    free(K);
    free(V);
    free(result);
    return 0;
}
