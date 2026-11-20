#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define ERROR(msg) do { fprintf(stderr, "ERROR: %s\n", msg); exit(0); } while(0)
#define ZERO 1e-7

__global__ void extract_abs_column(const double *A, double *abs_col, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int ii = i; ii < n - k; ii += stride) {
        abs_col[ii] = fabs(A[(k + ii) * n + k]);
    }
}

__global__ void swap_rows_kernel(double *A, double *b, int n, int r1, int r2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int jj = j; jj < n; jj += stride) {
        double tmp = A[r1 * (long long)n + jj];
        A[r1 * (long long)n + jj] = A[r2 * (long long)n + jj];
        A[r2 * (long long)n + jj] = tmp;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double tmp = b[r1];
        b[r1] = b[r2];
        b[r2] = tmp;
    }
}

__global__ void eliminate_kernel(double *A, double *b, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_y = gridDim.y * blockDim.y;
    int stride_x = gridDim.x * blockDim.x;

    for (int ii = i + k + 1; ii < n; ii += stride_y) {
        double factor = A[ii * (long long)n + k] / A[k * (long long)n + k];
        for (int jj = j + k; jj < n; jj += stride_x) {
            A[ii * (long long)n + jj] -= factor * A[k * (long long)n + jj];
        }
        // Update b only by threads that cover the pivot column position
        if (j == 0) {
            b[ii] -= factor * b[k];
        }
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1 || n <= 0 || n > 10000) {
        ERROR("invalid matrix size");
    }

    std::vector<double> h_A((size_t)n * n);
    std::vector<double> h_b(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (scanf("%lf", &h_A[(size_t)i * n + j]) != 1) {
                ERROR("failed to read matrix element");
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (scanf("%lf", &h_b[i]) != 1) {
            ERROR("failed to read b element");
        }
    }

    double *d_A = nullptr, *d_b = nullptr;
    cudaError_t err = cudaMalloc(&d_A, (size_t)n * n * sizeof(double));
    if (err != cudaSuccess) ERROR("failed to allocate device matrix");
    err = cudaMalloc(&d_b, n * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_A);
        ERROR("failed to allocate device vector");
    }

    err = cudaMemcpy(d_A, h_A.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_b);
        ERROR("failed to copy matrix to device");
    }
    err = cudaMemcpy(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A); cudaFree(d_b);
        ERROR("failed to copy vector to device");
    }

    // Fixed number of threads
    const int BLOCK_X = 16;
    const int BLOCK_Y = 16;
    const int GRID_X = 32;
    const int GRID_Y = 32;
    dim3 block2d(BLOCK_X, BLOCK_Y);
    dim3 grid2d(GRID_X, GRID_Y);
    dim3 block1d(256);
    dim3 grid1d(1024);

    thrust::device_vector<double> abs_col(n);

    for (int k = 0; k < n; ++k) {
        // Extract absolute values of column k (from k to n-1) using kernel + Thrust max
        extract_abs_column<<<grid1d, block1d>>>(d_A, thrust::raw_pointer_cast(abs_col.data()), n, k);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_b);
            ERROR("extract column failed");
        }

        auto max_it = thrust::max_element(abs_col.begin(), abs_col.begin() + (n - k));
        int piv_offset = max_it - abs_col.begin();
        int piv_row = k + piv_offset;

        double piv_val;
        cudaMemcpy(&piv_val, d_A + (size_t)piv_row * n + k, sizeof(double), cudaMemcpyDeviceToHost);
        if (fabs(piv_val) < ZERO) {
            cudaFree(d_A); cudaFree(d_b);
            ERROR("matrix is singular or nearly singular");
        }

        if (piv_row != k) {
            swap_rows_kernel<<<grid1d, block1d>>>(d_A, d_b, n, k, piv_row);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                cudaFree(d_A); cudaFree(d_b);
                ERROR("row swap failed");
            }
        }

        // Elimination
        eliminate_kernel<<<grid2d, block2d>>>(d_A, d_b, n, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_b);
            ERROR("elimination kernel launch failed");
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_b);
            ERROR("elimination kernel failed");
        }
    }

    // Copy back for back-substitution on host
    err = cudaMemcpy(h_A.data(), d_A, (size_t)n * n * sizeof(double), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(h_b.data(), d_b, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_b);
    if (err != cudaSuccess) {
        ERROR("failed to copy result from device");
    }

    // Back substitution
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = h_b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= h_A[(size_t)i * n + j] * x[j];
        }
        double diag = h_A[(size_t)i * n + i];
        if (fabs(diag) < ZERO) {
            ERROR("singular matrix during back substitution");
        }
        x[i] = sum / diag;
    }

    // Output with relative precision 1e-10
    for (int i = 0; i < n; ++i) {
        if (i) printf(" ");
        printf("%.10e", x[i]);
    }
    printf("\n");

    return 0;
}
