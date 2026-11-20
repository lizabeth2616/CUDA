#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define ERROR(msg) do { fprintf(stderr, "ERROR: %s\n", msg); exit(0); } while(0)

__global__ void prewitt_kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int yy = y; yy < h; yy += stride_y) {
        for (int xx = x; xx < w; xx += stride_x) {
            float gx = 0.0f;
            float gy = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uchar4 p = tex2D<uchar4>(tex, (xx + dx) + 0.5f, (yy + dy) + 0.5f);
                    float val = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
                    gx += dx * val;
                    gy += dy * val;
                }
            }
            float mag = sqrtf(gx * gx + gy * gy);
            int im = (int)(mag + 0.5f);
            if (im > 255) im = 255;
            if (im < 0) im = 0;
            out[yy * w + xx] = make_uchar4(im, im, im, 0);
        }
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
    // strip newlines
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
    if (w <= 0 || h <= 0 || (long long)w * h > 100000000LL) {
        fclose(fin);
        ERROR("invalid image dimensions");
    }

    size_t pixels = (size_t)w * h;
    uchar4 *host_in = (uchar4*)malloc(pixels * sizeof(uchar4));
    if (!host_in) {
        fclose(fin);
        ERROR("host memory allocation failed");
    }
    if (fread(host_in, sizeof(uchar4), pixels, fin) != pixels) {
        free(host_in);
        fclose(fin);
        ERROR("failed to read image data");
    }
    fclose(fin);

    // Device allocations and texture
    cudaArray *cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaError_t err = cudaMallocArray(&cuArray, &channelDesc, w, h);
    if (err != cudaSuccess) {
        free(host_in);
        ERROR("cudaMallocArray failed");
    }

    err = cudaMemcpy2DToArray(cuArray, 0, 0, host_in, w * sizeof(uchar4),
                              w * sizeof(uchar4), h, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("cudaMemcpy2DToArray failed");
    }

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("cudaCreateTextureObject failed");
    }

    uchar4 *dev_out;
    err = cudaMalloc(&dev_out, pixels * sizeof(uchar4));
    if (err != cudaSuccess) {
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("cudaMalloc for output failed");
    }

    // Fixed number of threads
    const int BLOCK_X = 16;
    const int BLOCK_Y = 16;
    const int GRID_X = 32;
    const int GRID_Y = 32;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(GRID_X, GRID_Y);

    prewitt_kernel<<<grid, block>>>(texObj, dev_out, w, h);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dev_out);
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("kernel launch failed");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(dev_out);
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("kernel execution failed");
    }

    uchar4 *host_out = (uchar4*)malloc(pixels * sizeof(uchar4));
    if (!host_out) {
        cudaFree(dev_out);
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("host output allocation failed");
    }
    err = cudaMemcpy(host_out, dev_out, pixels * sizeof(uchar4), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(host_out);
        cudaFree(dev_out);
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("cudaMemcpy from device failed");
    }

    FILE *fout = fopen(out_path, "wb");
    if (!fout) {
        free(host_out);
        cudaFree(dev_out);
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("cannot open output file");
    }
    if (fwrite(&w, sizeof(int), 1, fout) != 1 || fwrite(&h, sizeof(int), 1, fout) != 1 ||
        fwrite(host_out, sizeof(uchar4), pixels, fout) != pixels) {
        fclose(fout);
        free(host_out);
        cudaFree(dev_out);
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        free(host_in);
        ERROR("failed to write output image");
    }
    fclose(fout);

    // Cleanup
    free(host_out);
    cudaFree(dev_out);
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    free(host_in);

    return 0;
}
