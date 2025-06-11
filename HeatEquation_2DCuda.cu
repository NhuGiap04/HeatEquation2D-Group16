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

__global__ void JacobiKernel(float* u_new, float* u_old) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float uij = u_old[i * N + j];
        float y = (j + 1) * DELTA_S;
        float x = (i + 1) * DELTA_S;

        float down = (i == N - 1) ? alpha1(y) : u_old[(i + 1) * N + j];
        float up   = (i == 0)     ? alpha0(y) : u_old[(i - 1) * N + j];
        float left  = (j == 0)     ? beta0(x) : u_old[i * N + (j - 1)];
        float right = (j == N - 1) ? beta1(x) : u_old[i * N + (j + 1)];

        u_new[i * N + j] = 0.25f * (down + up + left + right);
    }
}

void InputData(float* u) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            u[i * N + j] = 0.0f;
}

int main() {
    float *u_old_h, *u_new_h;
    float *u_old_d, *u_new_d;
    FILE* fp = fopen("heat_output.txt", "w");

    u_old_h = (float*)malloc(N * N * sizeof(float));
    u_new_h = (float*)malloc(N * N * sizeof(float));

    InputData(u_old_h);

    cudaMalloc((void**)&u_old_d, N * N * sizeof(float));
    cudaMalloc((void**)&u_new_d, N * N * sizeof(float));

    cudaMemcpy(u_old_d, u_old_h, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    for (int step = 0; step < MAX_STEPS; step++) {
        JacobiKernel<<<numBlocks, threadsPerBlock>>>(u_new_d, u_old_d);
        cudaDeviceSynchronize();

        // Swap
        float* temp = u_old_d;
        u_old_d = u_new_d;
        u_new_d = temp;

        // Copy to host and write to file
        cudaMemcpy(u_new_h, u_old_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N * N; i++) {
            fprintf(fp, "%.6f ", u_new_h[i]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    cudaFree(u_old_d);
    cudaFree(u_new_d);
    free(u_old_h);
    free(u_new_h);

    return 0;
}
