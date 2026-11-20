#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void vectorMul(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

int main()
{
    int n;
    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "ERROR: invalid n\n");
        return 0;
    }

    float *h_a = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    float *h_c = (float*)malloc(n * sizeof(float));

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "ERROR: host memory allocation failed\n");
        free(h_a); free(h_b); free(h_c);
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        if (scanf("%f", &h_a[i]) != 1) {
            fprintf(stderr, "ERROR: failed to read vector A\n");
            free(h_a); free(h_b); free(h_c);
            return 0;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (scanf("%f", &h_b[i]) != 1) {
            fprintf(stderr, "ERROR: failed to read vector B\n");
            free(h_a); free(h_b); free(h_c);
            return 0;
        }
    }

    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    cudaError_t err;

    err = cudaMalloc((void**)&d_a, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_a failed: %s\n", cudaGetErrorString(err));
        free(h_a); free(h_b); free(h_c);
        return 0;
    }
    err = cudaMalloc((void**)&d_b, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }
    err = cudaMalloc((void**)&d_c, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_c failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }

    err = cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy H2D a failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }
    err = cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy H2D b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }

    err = cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        if (i > 0) printf(" ");
        printf("%.10e", h_c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
