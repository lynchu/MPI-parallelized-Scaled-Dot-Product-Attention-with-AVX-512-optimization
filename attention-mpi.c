#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <string.h>

__attribute__((constructor))
static void force_ucx_env() {
    setenv("UCX_LOG_LEVEL", "error", 1);
    setenv("OMPI_MCA_btl", "^openib,tcp", 1);
    setenv("OMPI_MCA_orte_base_help_aggregate", "1", 1);
    if (!getenv("UCX_TLS"))         setenv("UCX_TLS", "rc,sm,self", 1);
    if (!getenv("UCX_NET_DEVICES")) setenv("UCX_NET_DEVICES", "mlx5_0:1", 1);
}

static inline int owner_count(int n, int size, int rank) {
    int base = n / size, rem = n % size;
    return base + (rank < rem ? 1 : 0);
}

static inline int owner_disp(int n, int size, int rank) {
    int base = n / size, rem = n % size;
    return rank * base + (rank < rem ? rank : rem);
}

// AVX512 vectorized double to float conversion
// Converts 8 doubles to 8 floats per iteration
static inline void cvt_d2f_avx512(float* __restrict dst, const double* __restrict src, size_t n) {
    size_t i = 0;
    
    // Process 32 doubles (4 x 8) at a time for better throughput
    for (; i + 31 < n; i += 32) {
        // Load 8 doubles, convert to 8 floats, repeat 4 times
        __m512d d0 = _mm512_loadu_pd(src + i);
        __m512d d1 = _mm512_loadu_pd(src + i + 8);
        __m512d d2 = _mm512_loadu_pd(src + i + 16);
        __m512d d3 = _mm512_loadu_pd(src + i + 24);
        
        __m256 f0 = _mm512_cvtpd_ps(d0);
        __m256 f1 = _mm512_cvtpd_ps(d1);
        __m256 f2 = _mm512_cvtpd_ps(d2);
        __m256 f3 = _mm512_cvtpd_ps(d3);
        
        _mm256_storeu_ps(dst + i, f0);
        _mm256_storeu_ps(dst + i + 8, f1);
        _mm256_storeu_ps(dst + i + 16, f2);
        _mm256_storeu_ps(dst + i + 24, f3);
    }
    
    // Process remaining 8 doubles at a time
    for (; i + 7 < n; i += 8) {
        __m512d d = _mm512_loadu_pd(src + i);
        __m256 f = _mm512_cvtpd_ps(d);
        _mm256_storeu_ps(dst + i, f);
    }
    
    // Handle remainder with scalar code
    for (; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

// AVX512 vectorized float to double conversion
// Converts 8 floats to 8 doubles per iteration
static inline void cvt_f2d_avx512(double* __restrict dst, const float* __restrict src, size_t n) {
    size_t i = 0;
    
    // Process 32 floats (4 x 8) at a time for better throughput
    for (; i + 31 < n; i += 32) {
        // Load 8 floats, convert to 8 doubles, repeat 4 times
        __m256 f0 = _mm256_loadu_ps(src + i);
        __m256 f1 = _mm256_loadu_ps(src + i + 8);
        __m256 f2 = _mm256_loadu_ps(src + i + 16);
        __m256 f3 = _mm256_loadu_ps(src + i + 24);
        
        __m512d d0 = _mm512_cvtps_pd(f0);
        __m512d d1 = _mm512_cvtps_pd(f1);
        __m512d d2 = _mm512_cvtps_pd(f2);
        __m512d d3 = _mm512_cvtps_pd(f3);
        
        _mm512_storeu_pd(dst + i, d0);
        _mm512_storeu_pd(dst + i + 8, d1);
        _mm512_storeu_pd(dst + i + 16, d2);
        _mm512_storeu_pd(dst + i + 24, d3);
    }
    
    // Process remaining 8 floats at a time
    for (; i + 7 < n; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m512d d = _mm512_cvtps_pd(f);
        _mm512_storeu_pd(dst + i, d);
    }
    
    // Handle remainder with scalar code
    for (; i < n; i++) {
        dst[i] = (double)src[i];
    }
}

static inline float dot_avx512(const float* __restrict a, const float* __restrict b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 63 < n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i),    acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
    }
    for (; i + 15 < n; i += 16) 
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), acc0);
    int rem = n - i;
    if (rem > 0) {
        __mmask16 k = (1u << rem) - 1u;
        acc0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(k,a+i), _mm512_maskz_loadu_ps(k,b+i), acc0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0,acc1), _mm512_add_ps(acc2,acc3)));
}

static inline void axpy_avx512(float* __restrict y, const float* __restrict x, float alpha, int n) {
    __m512 aval = _mm512_set1_ps(alpha);
    int i = 0;
    for (; i + 63 < n; i += 64) {
        _mm512_storeu_ps(y+i,    _mm512_fmadd_ps(aval, _mm512_loadu_ps(x+i),    _mm512_loadu_ps(y+i)));
        _mm512_storeu_ps(y+i+16, _mm512_fmadd_ps(aval, _mm512_loadu_ps(x+i+16), _mm512_loadu_ps(y+i+16)));
        _mm512_storeu_ps(y+i+32, _mm512_fmadd_ps(aval, _mm512_loadu_ps(x+i+32), _mm512_loadu_ps(y+i+32)));
        _mm512_storeu_ps(y+i+48, _mm512_fmadd_ps(aval, _mm512_loadu_ps(x+i+48), _mm512_loadu_ps(y+i+48)));
    }
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(y+i, _mm512_fmadd_ps(aval, _mm512_loadu_ps(x+i), _mm512_loadu_ps(y+i)));
    int rem = n - i;
    if (rem > 0) {
        __mmask16 k = (1u << rem) - 1u;
        _mm512_mask_storeu_ps(y+i, k, _mm512_fmadd_ps(aval, _mm512_maskz_loadu_ps(k,x+i), 
                                                       _mm512_mask_loadu_ps(_mm512_setzero_ps(),k,y+i)));
    }
}

static inline void memset_zero_scale(float* y, float scale, int n, bool zero_first) {
    __m512 val = zero_first ? _mm512_setzero_ps() : _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 63 < n; i += 64) {
        if (zero_first) {
            _mm512_storeu_ps(y+i, val); _mm512_storeu_ps(y+i+16, val);
            _mm512_storeu_ps(y+i+32, val); _mm512_storeu_ps(y+i+48, val);
        } else {
            _mm512_storeu_ps(y+i,    _mm512_mul_ps(_mm512_loadu_ps(y+i),    val));
            _mm512_storeu_ps(y+i+16, _mm512_mul_ps(_mm512_loadu_ps(y+i+16), val));
            _mm512_storeu_ps(y+i+32, _mm512_mul_ps(_mm512_loadu_ps(y+i+32), val));
            _mm512_storeu_ps(y+i+48, _mm512_mul_ps(_mm512_loadu_ps(y+i+48), val));
        }
    }
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(y+i, zero_first ? val : _mm512_mul_ps(_mm512_loadu_ps(y+i), val));
    int rem = n - i;
    if (rem > 0) {
        __mmask16 k = (1u << rem) - 1u;
        if (zero_first)
            _mm512_mask_storeu_ps(y+i, k, val);
        else
            _mm512_mask_storeu_ps(y+i, k, _mm512_mul_ps(_mm512_maskz_loadu_ps(k,y+i), val));
    }
}

static inline void online_softmax_attention(float* __restrict contrib, float* lmax, float* lsum,
    const float* __restrict q, const float* __restrict K_local, const float* __restrict V_local,
    int n_local, int dk, int dv, float scale)
{
    float rmax = -INFINITY, rsum = 0.0f;
    memset_zero_scale(contrib, 0.0f, dv, true);
    
    for (int j = 0; j < n_local; ++j) {
        float s = dot_avx512(q, K_local + (size_t)j*dk, dk) * scale;
        float old = rmax;
        rmax = (s > rmax) ? s : rmax;
        float corr = expf(old - rmax);
        rsum = rsum * corr + expf(s - rmax);
        if (j > 0) memset_zero_scale(contrib, corr, dv, false);
        axpy_avx512(contrib, V_local + (size_t)j*dv, expf(s - rmax), dv);
        if (j+1 < n_local) {
            _mm_prefetch((const char*)(K_local + (size_t)(j+1)*dk), _MM_HINT_T0);
            _mm_prefetch((const char*)(V_local + (size_t)(j+1)*dv), _MM_HINT_T0);
        }
    }
    *lmax = rmax; *lsum = rsum;
}

void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv, int mpi_rank, int mpi_size) {
    const int root = 0;
    int dims[4];
    if (mpi_rank == root) { dims[0]=m; dims[1]=n; dims[2]=dk; dims[3]=dv; }
    MPI_Bcast(dims, 4, MPI_INT, root, MPI_COMM_WORLD);
    m=dims[0]; n=dims[1]; dk=dims[2]; dv=dims[3];

    int n_local = owner_count(n, mpi_size, mpi_rank);
    int B = 512;  // Default: full batch for small m
    
    // Alloc shared buffers
    float *K_local = malloc(sizeof(float)*n_local*dk);
    float *V_local = malloc(sizeof(float)*n_local*dv);
    float *lmax = malloc(sizeof(float)*B), *lsum = malloc(sizeof(float)*B);
    float *gmax = malloc(sizeof(float)*B), *gsum = malloc(sizeof(float)*B);
    
    float scale = 1.0f / sqrtf((float)dk);
    
    // Adaptive strategy: choose between Bcast and Scatterv based on data size
    // Threshold: if total data < 64 MB, use Bcast (faster for small data)
    // Otherwise use Scatterv (more memory efficient for large data)
    size_t total_kv_size = (size_t)n * (dk + dv) * sizeof(float);
    const size_t BCAST_THRESHOLD = 64 * 1024 * 1024;  // 64 MB
    bool use_bcast = (total_kv_size < BCAST_THRESHOLD);
    
    if (use_bcast) {
        // Small data: Use Bcast (avoids Scatterv initialization overhead)
        float *Kf = NULL, *Vf = NULL;
        
        if (mpi_rank == root) {
            Kf = malloc(sizeof(float)*n*dk);
            Vf = malloc(sizeof(float)*n*dv);
            cvt_d2f_avx512(Kf, K, (size_t)n*dk);
            cvt_d2f_avx512(Vf, V, (size_t)n*dv);
        } else {
            Kf = malloc(sizeof(float)*n*dk);
            Vf = malloc(sizeof(float)*n*dv);
        }
        
        // Broadcast K and V to all ranks
        MPI_Bcast(Kf, n*dk, MPI_FLOAT, root, MPI_COMM_WORLD);
        MPI_Bcast(Vf, n*dv, MPI_FLOAT, root, MPI_COMM_WORLD);
        
        // Each rank extracts its own partition
        int my_start = owner_disp(n, mpi_size, mpi_rank);
        memcpy(K_local, Kf + (size_t)my_start * dk, sizeof(float) * n_local * dk);
        memcpy(V_local, Vf + (size_t)my_start * dv, sizeof(float) * n_local * dv);
        
        free(Kf);
        free(Vf);
    } else {
        // Large data: Use Scatterv (more memory efficient)
        if (mpi_rank == root) {
            float *Kf = malloc(sizeof(float)*n*dk);
            float *Vf = malloc(sizeof(float)*n*dv);
            
            cvt_d2f_avx512(Kf, K, (size_t)n*dk);
            cvt_d2f_avx512(Vf, V, (size_t)n*dv);
            
            int *ck = malloc(sizeof(int)*mpi_size), *dk_d = malloc(sizeof(int)*mpi_size);
            int *cv = malloc(sizeof(int)*mpi_size), *dv_d = malloc(sizeof(int)*mpi_size);
            for (int r=0; r<mpi_size; ++r) {
                int cnt = owner_count(n,mpi_size,r), d = owner_disp(n,mpi_size,r);
                ck[r]=cnt*dk; dk_d[r]=d*dk; cv[r]=cnt*dv; dv_d[r]=d*dv;
            }
            
            MPI_Scatterv(Kf, ck, dk_d, MPI_FLOAT, K_local, n_local*dk, MPI_FLOAT, root, MPI_COMM_WORLD);
            MPI_Scatterv(Vf, cv, dv_d, MPI_FLOAT, V_local, n_local*dv, MPI_FLOAT, root, MPI_COMM_WORLD);
            
            free(ck); free(dk_d); free(cv); free(dv_d); free(Kf); free(Vf);
        } else {
            MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, K_local, n_local*dk, MPI_FLOAT, root, MPI_COMM_WORLD);
            MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, V_local, n_local*dv, MPI_FLOAT, root, MPI_COMM_WORLD);
        }
    }

    // Ping-pong for Q batches (2 buffers for prefetch)
    const int num_q_buf = 2;
    float *q_bufs = malloc(num_q_buf * B * dk * sizeof(float));
    float **q_batch_ptrs = malloc(num_q_buf * sizeof(float *));
    for (int k = 0; k < num_q_buf; ++k) {
        q_batch_ptrs[k] = q_bufs + (size_t)k * B * dk;
    }
    MPI_Request q_pending_req = MPI_REQUEST_NULL;  // For prefetch I bcast
    int next_batch_id = 0;  // Ping-pong index

    // Ping-pong buffers for contrib
    const int num_buf = 2;
    float *cbs = malloc(num_buf * B * dv * sizeof(float));
    float **cb_bufs = malloc(num_buf * sizeof(float *));
    for (int k = 0; k < num_buf; ++k) {
        cb_bufs[k] = cbs + (size_t)k * B * dv;
    }
    float *rbs = NULL;
    float **rb_bufs = NULL;
    if (mpi_rank == root) {
        rbs = malloc(num_buf * B * dv * sizeof(float));
        rb_bufs = malloc(num_buf * sizeof(float *));
        for (int k = 0; k < num_buf; ++k) {
            rb_bufs[k] = rbs + (size_t)k * B * dv;
        }
    }

    // Pipeline state
    MPI_Request pending_req = MPI_REQUEST_NULL;
    int prev_i = 0, prev_bs = 0;
    int num_iter = (m + B - 1) / B;

    // Prefetch first Q batch
    int first_bs = (B <= m) ? B : m;
    if (mpi_rank == root) {
        cvt_d2f_avx512(q_batch_ptrs[0], Q, (size_t)first_bs*dk);
    }
    MPI_Ibcast(q_batch_ptrs[0], first_bs * dk, MPI_FLOAT, root, MPI_COMM_WORLD, &q_pending_req);

    for (int ii = 0; ii < num_iter; ++ii) {
        int i = ii * B;
        int bs = (i + B <= m) ? B : m - i;
        int buf_id = ii % num_buf;
        float *curr_cb = cb_bufs[buf_id];
        float *curr_q = q_batch_ptrs[next_batch_id];  // Current Q batch

        // Wait for current Q batch (overlap with prev iter)
        if (ii > 0 || (ii == 0 && q_pending_req != MPI_REQUEST_NULL)) {
            MPI_Wait(&q_pending_req, MPI_STATUS_IGNORE);
        }

        // Prefetch NEXT Q batch (non-blocking)
        next_batch_id = (next_batch_id + 1) % num_q_buf;
        int next_i = (ii + 1) * B;
        int next_bs = (next_i + B <= m) ? B : m - next_i;
        if (next_i < m) {
            if (mpi_rank == root) {
                cvt_d2f_avx512(q_batch_ptrs[next_batch_id], Q + (size_t)next_i * dk, (size_t)next_bs*dk);
            }
            MPI_Ibcast(q_batch_ptrs[next_batch_id], next_bs * dk, MPI_FLOAT, root, MPI_COMM_WORLD, &q_pending_req);
        } else {
            q_pending_req = MPI_REQUEST_NULL;  // No more
        }

        // 1. Local batched softmax (use curr_q)
        for (int b = 0; b < bs; ++b) {
            float *curr_contrib = curr_cb + (size_t)b * dv;
            online_softmax_attention(curr_contrib, lmax + b, lsum + b,
                                     curr_q + (size_t)b * dk, K_local, V_local,
                                     n_local, dk, dv, scale);
        }

        // 2. Global max reduction
        MPI_Request temp_req;
        MPI_Iallreduce(lmax, gmax, bs, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD, &temp_req);
        MPI_Wait(&temp_req, MPI_STATUS_IGNORE);

        // 3. Correct local sum & contrib
        for (int b = 0; b < bs; ++b) {
            float corr = expf(lmax[b] - gmax[b]);
            lsum[b] *= corr;
            float *curr_contrib = curr_cb + (size_t)b * dv;
            memset_zero_scale(curr_contrib, corr, dv, false);
        }

        // 4. Global sum reduction
        MPI_Iallreduce(lsum, gsum, bs, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &temp_req);
        MPI_Wait(&temp_req, MPI_STATUS_IGNORE);

        // 5. Normalize contrib
        for (int b = 0; b < bs; ++b) {
            float inv = (gsum[b] == 0.0f) ? 0.0f : 1.0f / gsum[b];
            float *curr_contrib = curr_cb + (size_t)b * dv;
            memset_zero_scale(curr_contrib, inv, dv, false);
        }

        // 6. Wait prev Reduce & copy result
        if (ii > 0) {
            MPI_Wait(&pending_req, MPI_STATUS_IGNORE);
            if (mpi_rank == root) {
                int prev_buf_id = (ii - 1) % num_buf;
                float *prev_rb = rb_bufs[prev_buf_id];
                for (int b = 0; b < prev_bs; ++b) {
                    double *ri = result + (size_t)(prev_i + b) * dv;
                    float *rbp = prev_rb + (size_t)b * dv;
                    cvt_f2d_avx512(ri, rbp, dv);
                }
            }
        }

        // 7. Issue non-blocking Reduce
        float *recvbuf = (mpi_rank == root ? rb_bufs[buf_id] : NULL);
        MPI_Ireduce(curr_cb, recvbuf, bs * dv, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD, &pending_req);

        prev_i = i;
        prev_bs = bs;
    }

    // Final wait & copy (with AVX512 conversion)
    if (pending_req != MPI_REQUEST_NULL) {
        MPI_Wait(&pending_req, MPI_STATUS_IGNORE);
        if (mpi_rank == root) {
            int last_buf_id = (num_iter - 1) % num_buf;
            int last_start = (num_iter - 1) * B;
            float *last_rb = rb_bufs[last_buf_id];
            for (int b = 0; b < prev_bs; ++b) {
                double *ri = result + (size_t)(last_start + b) * dv;
                float *rbp = last_rb + (size_t)b * dv;
                cvt_f2d_avx512(ri, rbp, dv);
            }
        }
    }

    // Cleanup
    free(q_bufs); free(q_batch_ptrs);
    free(cbs); free(cb_bufs);
    if (mpi_rank == root) { free(rbs); free(rb_bufs); }
    free(lmax); free(lsum); free(gmax); free(gsum);
    free(K_local); free(V_local);
}



// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 127 <template code>) <(tail -n 127 <your code>)`

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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* Q = NULL;
    double* K = NULL;
    double* V = NULL;
    double* result = NULL;
    int m, n, dk, dv;

    if (rank == 0) {
        read_matrices(argv[1], &Q, &K, &V, &m, &n, &dk, &dv);
        result = malloc(sizeof(double) * m * dv);
    }

    double beg, duration, duration_max;
    beg = MPI_Wtime();
    attention(Q, K, V, result, m, n, dk, dv, rank, size);
    duration = MPI_Wtime() - beg;

    MPI_Reduce(&duration, &duration_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (verify(argv[1], result)) {
            printf("Correct!\nElapsed time: %.2lf us\n", duration_max * 1e6);
        } else {
            puts("Wrong!");
        }
    }

    MPI_Finalize();

    free(Q);
    free(K);
    free(V);
    free(result);
    return 0;
}