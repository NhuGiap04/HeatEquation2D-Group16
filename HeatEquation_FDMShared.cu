#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 100
#define MAX_STEPS 10000
#define C 0.1f
#define DELTA_S (1.0f / (N + 1))
#define DELTA_T ((DELTA_S * DELTA_S) / (4.0f * C))

__device__ float alpha0(float y) { return 0.0f; }
__device__ float alpha1(float y) { return sqrtf(y / 4.0f); }
__device__ float beta0(float x)  { return 100.0f * (0.7f + 0.3f * sinf(5.0f * 3.14f * x / 4.0f)); }
__device__ float beta1(float x)  { return 100.0f * cbrtf(x / 4.0f); }

__global__ void JacobiSharedKernel(float* u_new, float* u_old) {
    __shared__ float tile[16 + 2][16 + 2]; // Assuming 16x16 block + halo
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx;
    int i = blockIdx.y * blockDim.y + ty;

    int tile_x = tx + 1;
    int tile_y = ty + 1;

    if (i < N && j < N) {
        tile[tile_y][tile_x] = u_old[i * N + j];

        if (tx == 0 && j > 0)
            tile[tile_y][tile_x - 1] = u_old[i * N + (j - 1)];
        if (tx == blockDim.x - 1 && j < N - 1)
            tile[tile_y][tile_x + 1] = u_old[i * N + (j + 1)];
        if (ty == 0 && i > 0)
            tile[tile_y - 1][tile_x] = u_old[(i - 1) * N + j];
        if (ty == blockDim.y - 1 && i < N - 1)
            tile[tile_y + 1][tile_x] = u_old[(i + 1) * N + j];
    }

    __syncthreads();

    if (i < N && j < N) {
        float y = (j + 1) * DELTA_S;
        float x = (i + 1) * DELTA_S;
        float up    = (i == 0)     ? alpha0(y) : tile[tile_y - 1][tile_x];
        float down  = (i == N - 1) ? alpha1(y) : tile[tile_y + 1][tile_x];
        float left  = (j == 0)     ? beta0(x)  : tile[tile_y][tile_x - 1];
        float right = (j == N - 1) ? beta1(x)  : tile[tile_y][tile_x + 1];
        u_new[i * N + j] = 0.25f * (up + down + left + right);
    }
}

void InputData(float* u) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            u[i * N + j] = 0.0f;
}

int main() {
    float* u_old_h = (float*)malloc(N * N * sizeof(float));
    float* u_new_h = (float*)malloc(N * N * sizeof(float));
    float *u_old_d, *u_new_d;

    cudaMalloc((void**)&u_old_d, N * N * sizeof(float));
    cudaMalloc((void**)&u_new_d, N * N * sizeof(float));

    InputData(u_old_h);
    cudaMemcpy(u_old_d, u_old_h, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    FILE* fp = fopen("heat_output.txt", "w");

    for (int step = 0; step < MAX_STEPS; step++) {
        JacobiSharedKernel<<<numBlocks, threadsPerBlock>>>(u_new_d, u_old_d);
        cudaDeviceSynchronize();

        // Swap pointers
        float* temp = u_old_d;
        u_old_d = u_new_d;
        u_new_d = temp;

        // Write output to file
        cudaMemcpy(u_new_h, u_old_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < N * N; i++) {
        //     fprintf(fp, "%.6f ", u_new_h[i]);
        // }
        // fprintf(fp, "\n");
    }

    fclose(fp);
    cudaFree(u_old_d);
    cudaFree(u_new_d);
    free(u_old_h);
    free(u_new_h);

    return 0;
}
