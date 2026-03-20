// GPU Merge-Path Project - Strict CUDA C Single-File Program

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <time.h>


// CPU timing helper

double cpu_seconds()
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}


// qsort comparison function
int cmpfunc(const void *a, const void *b)
{
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}


// CPU merge (baseline, strict C)
void merge_cpu(const int *A, int m,
               const int *B, int n,
               int *C)
{
    int i = 0, j = 0, k = 0;

    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}


// The Device helper: causing merge-path partition (binary search on diagonal)

__device__ int find_partition(const int *A, int m,
                               const int *B, int n,
                               int k)
{
    int low = (k > n) ? k - n : 0;
    int high = (k < m) ? k : m;

    while (low < high) {
        int i = (low + high + 1) >> 1;
        int j = k - i;

        int A_val = (i == 0) ? INT_MIN : A[i-1];
        int B_val = (j == n) ? INT_MAX : B[j];

        if (A_val <= B_val) {
            low = i;
        } else {
            high = i - 1;
        }
    }

    return low;
}


// The Kernel: merge a slice using merge path
__global__ void merge_path_kernel(const int *A, int m,
                                  const int *B, int n,
                                  int *C)
{
    int b  = blockIdx.x;
    int Bn = gridDim.x;
    /*
     * Perform arithmetic in 64-bit to avoid overflow when m+n is large
     * and gridDim.x (Bn) is non-trivial.  The product (b+1)*L can exceed
     * INT_MAX for big arrays (e.g. 256 * 10e6 > 2^31), which might cause
     * incorrect results and zero timing for largest test case.
     */
    long long L = (long long)m + (long long)n;

    long long k_start_ll = ((long long)b * L) / Bn;
    long long k_end_ll   = ((long long)(b + 1) * L) / Bn;

    int k_start = (int)k_start_ll;
    int k_end   = (int)k_end_ll;

    int i_start = find_partition(A, m, B, n, k_start);
    int j_start = k_start - i_start;
    int i_end   = find_partition(A, m, B, n, k_end);
    int j_end   = k_end - i_end;

    int i = i_start;
    int j = j_start;
    int k = k_start;

    while (i < i_end && j < j_end) {
        if (A[i] <= B[j]) {
            C[k] = A[i];
            i++;
        } else {
            C[k] = B[j];
            j++;
        }
        k++;
    }

    while (i < i_end) {
        C[k] = A[i];
        i++;
        k++;
    }

    while (j < j_end) {
        C[k] = B[j];
        j++;
        k++;
    }
}


// The Host wrapper, To merge two sorted arrays using GPU merge-path
void merge_gpu_path(const int *A, int m,
                    const int *B, int n,
                    int *C,
                    int numBlocks)
{
    int *d_A, *d_B, *d_C;
    int sizeA = m * sizeof(int);
    int sizeB = n * sizeof(int);
    int sizeC = (m + n) * sizeof(int);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    merge_path_kernel<<<numBlocks, 1>>>(d_A, m, d_B, n, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


// The GPU timing wrapper, To measure kernel-only time using CUDA events

float gpu_merge_time_ms(const int *A, int m,
                        const int *B, int n,
                        int *C,
                        int numBlocks)
{
    int *d_A, *d_B, *d_C;
    int sizeA = m * sizeof(int);
    int sizeB = n * sizeof(int);
    int sizeC = (m + n) * sizeof(int);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    merge_path_kernel<<<numBlocks, 1>>>(d_A, m, d_B, n, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}


// The Main to test + large-N benchmarking

int main()
{
    cudaSetDevice(0);

    
    // For Small correctness test
    {
        int m = 8;
        int n = 8;

        int A[8] = {1, 4, 6, 7, 10, 13, 15, 20};
        int B[8] = {2, 3, 5, 8, 9, 11, 12, 14};

        int *C_cpu = (int*)malloc((m + n) * sizeof(int));
        int *C_gpu = (int*)malloc((m + n) * sizeof(int));

        merge_cpu(A, m, B, n, C_cpu);
        merge_gpu_path(A, m, B, n, C_gpu, 256);

        int ok = 1;
        for (int i = 0; i < m + n; i++) {
            if (C_cpu[i] != C_gpu[i]) {
                ok = 0;
                break;
            }
        }

        if (ok) printf("Small test: GPU merge matches CPU merge.\n");
        else    printf("Small test: Mismatch detected.\n");

        free(C_cpu);
        free(C_gpu);
    }

    // Benchmarking for big N's
    FILE *fp_clear = fopen("results.txt", "w");
    if (fp_clear) fclose(fp_clear);
    
    srand(time(NULL));

    int Ns[] = {100000, 500000, 1000000, 5000000, 10000000};
    int numSizes = sizeof(Ns) / sizeof(Ns[0]);

    for (int s = 0; s < numSizes; s++) {
        int N = Ns[s];
        int m = N / 2;
        int n = N - m;

        int *A = (int*)malloc(m * sizeof(int));
        int *B = (int*)malloc(n * sizeof(int));
        int *C_cpu = (int*)malloc(N * sizeof(int));
        int *C_gpu = (int*)malloc(N * sizeof(int));

        // Generate random data
        for (int i = 0; i < m; i++) A[i] = rand();
        for (int i = 0; i < n; i++) B[i] = rand();

        // Sort halves on CPU
        qsort(A, m, sizeof(int), cmpfunc);
        qsort(B, n, sizeof(int), cmpfunc);

        // CPU timing
        double t0 = cpu_seconds();
        merge_cpu(A, m, B, n, C_cpu);
        double t1 = cpu_seconds();
        double cpu_time = t1 - t0;

        // GPU timing (kernel only)
        int numBlocks = 256;
        float gpu_ms = gpu_merge_time_ms(A, m, B, n, C_gpu, numBlocks);
        double gpu_time = gpu_ms / 1000.0;

        // Correctness check
        int ok = 1;
        for (int i = 0; i < N; i++) {
            if (C_cpu[i] != C_gpu[i]) {
                ok = 0;
                break;
            }
        }

        double speedup = cpu_time / gpu_time;

        printf("\nN = %d\n", N);
        printf("CPU merge time : %.6f s\n", cpu_time);
        printf("GPU merge time : %.6f s (kernel only)\n", gpu_time);
        printf("Speedup (CPU/GPU): %.2f x\n", speedup);
        printf("Correctness: %s\n", ok ? "PASS" : "FAIL");
        
        
        // This will append results to results.txt automatically
       
        FILE *fp = fopen("results.txt", "a");
        if (fp) {
            fprintf(fp, "N = %d\n", N);
            fprintf(fp, "CPU merge time : %.6f s\n", cpu_time);
            fprintf(fp, "GPU merge time : %.6f s (kernel only)\n", gpu_time);
            fprintf(fp, "Speedup (CPU/GPU): %.2f x\n", speedup);
            fprintf(fp, "Correctness: %s\n\n", ok ? "PASS" : "FAIL");
            fclose(fp);
        } 

        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
    }

    return 0;
}