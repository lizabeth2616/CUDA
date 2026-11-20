#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define ERROR(msg) do { fprintf(stderr, "ERROR: %s\n", msg); exit(0); } while(0)

__constant__ float d_avg[32][3];

__global__ void min_distance_kernel(uchar4 *img, int total, int nc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < total; i += stride) {
        uchar4 p = img[i];
        float pr = (float)p.x;
        float pg = (float)p.y;
        float pb = (float)p.z;

        float min_d2 = 1e30f;
        int best = 0;

        for (int j = 0; j < nc; ++j) {
            float dr = pr - d_avg[j][0];
            float dg = pg - d_avg[j][1];
            float db = pb - d_avg[j][2];
            float d2 = dr * dr + dg * dg + db * db;
            if (d2 < min_d2) {
                min_d2 = d2;
                best = j;
            }
        }
        p.w = (unsigned char)best;
        img[i] = p;
    }
}

int main() {
    char in_path[1024], out_path[1024];
    if (fgets(in_path, sizeof(in_path), stdin) == NULL) {
        ERROR("failed to read input path");
    }
    if (fgets(out_path, sizeof(out_path), stdin) == NULL) {
        ERROR("failed to read output path");
    }
    in_path[strcspn(in_path, "\r\n")] = 0;
    out_path[strcspn(out_path, "\r\n")] = 0;

    FILE *fin = fopen(in_path, "rb");
    if (!fin) {
        ERROR("cannot open input file");
    }

    int w, h;
    if (fread(&w, sizeof(int), 1, fin) != 1 || fread(&h, sizeof(int), 1, fin) != 1) {
        fclose(fin);
        ERROR("failed to read image dimensions");
    }
    if (w <= 0 || h <= 0 || (long long)w * h > 400000000LL) {
        fclose(fin);
        ERROR("invalid image dimensions");
    }

    size_t pixels = (size_t)w * h;
    uchar4 *host_img = (uchar4*)malloc(pixels * sizeof(uchar4));
    if (!host_img) {
        fclose(fin);
        ERROR("host memory allocation failed");
    }
    if (fread(host_img, sizeof(uchar4), pixels, fin) != pixels) {
        free(host_img);
        fclose(fin);
        ERROR("failed to read image data");
    }
    fclose(fin);

    int nc;
    if (scanf("%d", &nc) != 1 || nc <= 0 || nc > 32) {
        free(host_img);
        ERROR("invalid number of classes");
    }

    float host_avg[32][3] = {0};
    for (int j = 0; j < nc; ++j) {
        int npj;
        if (scanf("%d", &npj) != 1 || npj <= 0 || npj > (1 << 19)) {
            free(host_img);
            ERROR("invalid number of sample pixels");
        }
        double sum_r = 0, sum_g = 0, sum_b = 0;
        for (int k = 0; k < npj; ++k) {
            int x, y;
            if (scanf("%d %d", &x, &y) != 2) {
                free(host_img);
                ERROR("failed to read sample coordinates");
            }
            if (x < 0 || x >= w || y < 0 || y >= h) {
                free(host_img);
                ERROR("sample coordinates out of bounds");
            }
            uchar4 pix = host_img[(size_t)y * w + x];
            sum_r += pix.x;
            sum_g += pix.y;
            sum_b += pix.z;
        }
        host_avg[j][0] = (float)(sum_r / npj);
        host_avg[j][1] = (float)(sum_g / npj);
        host_avg[j][2] = (float)(sum_b / npj);
    }

    cudaError_t err = cudaMemcpyToSymbol(d_avg, host_avg, sizeof(host_avg));
    if (err != cudaSuccess) {
        free(host_img);
        ERROR("cudaMemcpyToSymbol failed");
    }

    uchar4 *dev_img;
    err = cudaMalloc(&dev_img, pixels * sizeof(uchar4));
    if (err != cudaSuccess) {
        free(host_img);
        ERROR("cudaMalloc failed");
    }
    err = cudaMemcpy(dev_img, host_img, pixels * sizeof(uchar4), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dev_img);
        free(host_img);
        ERROR("cudaMemcpy to device failed");
    }

    // Fixed number of threads
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = 1024;
    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    min_distance_kernel<<<grid, block>>>(dev_img, (int)pixels, nc);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dev_img);
        free(host_img);
        ERROR("kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(dev_img);
        free(host_img);
        ERROR("kernel execution failed");
    }

    err = cudaMemcpy(host_img, dev_img, pixels * sizeof(uchar4), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(dev_img);
        free(host_img);
        ERROR("cudaMemcpy from device failed");
    }
    cudaFree(dev_img);

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        free(host_img);
        ERROR("cannot open output file");
    }
    if (fwrite(&w, sizeof(int), 1, fout) != 1 ||
        fwrite(&h, sizeof(int), 1, fout) != 1 ||
        fwrite(host_img, sizeof(uchar4), pixels, fout) != pixels) {
        fclose(fout);
        free(host_img);
        ERROR("failed to write output image");
    }
    fclose(fout);
    free(host_img);

    return 0;
}
