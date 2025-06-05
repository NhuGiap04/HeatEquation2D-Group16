#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cuda.h>
#include <float.h>
#define  N       100  /* mesh size = M = n  */
#define  c       0.1 /* diffusion coefficient */
#define  max_steps   10000  /* number of time iterations */
/* choose Δt ≤ (Δs)^2/(2c) for stability; here a bit more conservative */
#define  delta_s      1.0 / (N + 1)
#define  delta_t      (delta_s * delta_s) / (4.0 * c)
#define BlockSizeX 4
#define BlockSizeY 4
#define GridSizeX  (N - 1)/BlockSizeX + 1
#define GridSizeY  (N - 1)/BlockSizeY + 1
// #define ThreadSizeX (N - 1)/(GridSizeX*BlockSizeX) + 1
// #define ThreadSizeY (N - 1)/(GridSizeY*BlockSizeY) + 1

/* boundary temperatures */
__device__ float alpha0(float y) { return 10.0; }  /* u(t,0,y) */
__device__ float alpha1(float y) { return 40.0; }  /* u(t,1,y) */
__device__ float beta0(float x)  { return 30.0; }  /* u(t,x,0) */
__device__ float beta1(float x)  { return 50.0; }  /* u(t,x,1) */

/* initial interior temperature */
float Initial(float x, float y)
{
    return 0.0;
}

void InputData(float *U)
{
    int i, j;
    for (i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
            *(U+i*N+j) = Initial(i*delta_s, j*delta_s);
        }
    }
}

/* initial interior temperature */
__global__ void Jacobi_Iterator(float *U, float *U_new){
    // T = array N x N
    // dT = array N x N
    int i, j;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<N) && (j < N))
    {
        float down = (i==N-1) ? alpha1((j+1) * delta_s) : *(U + (i+1)*N+ j);
        float up = (i==0) ? alpha0((j+1) * delta_s) : *(U + (i-1)*N + j);
        float left = (j==0) ? beta0((i+1) * delta_s) : *(U + i*N + (j-1));
        float right = (j==N-1) ? beta1((i+1) * delta_s) : *(U + i*N + (j+1));

        *(U_new+i*N+j) = 0.25 * (down + up + left + right);
    }
}

int main() {
    int i, j, t;
    float *Uold_cpu, *Unew_cpu;
    Uold_cpu  = (float *)malloc((N * N)*sizeof(float));
    Unew_cpu = (float *)malloc((N * N)*sizeof(float));
    InputData(Uold_cpu);

    // CUDA code
    //1. Delare and Allocate Mem on GPU
    float *Uold_gpu,*Unew_gpu;
    cudaMalloc((void**)&Uold_gpu ,(N * N)*sizeof(float));
    cudaMalloc((void**)&Unew_gpu,(N * N)*sizeof(float));

    //2. Copy Input from CPU to GPU
    cudaMemcpy(Uold_gpu, Uold_cpu,(N * N)*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(Unew_gpu, Uold_gpu,(N * N)*sizeof(float),cudaMemcpyDeviceToDevice);

    //3. Define Block and Thread Structure
    dim3 dimGrid(GridSizeX,GridSizeY);
    dim3 dimBlock(BlockSizeX,BlockSizeY);

    for (t=0; t<max_steps; t=t+2) {
        Jacobi_Iterator<<<dimGrid,dimBlock>>>(Unew_gpu,Uold_gpu);
        cudaDeviceSynchronize();
        Jacobi_Iterator<<<dimGrid,dimBlock>>>(Uold_gpu,Unew_gpu);
        cudaDeviceSynchronize();
    }

    //5. Copy Output from GPU to CPU
    cudaMemcpy(Unew_cpu, Unew_gpu, (N * N)*sizeof(float),cudaMemcpyDeviceToHost);
    for (i = 0; i < N; i++ )
        for (j = 0; j < N; j++) printf("%f \n",*(Unew_cpu+i*N+j));
    //6. Free Mem on CPU and GPU
    free(Uold_cpu); free(Unew_cpu);
    cudaFree(Uold_gpu);cudaFree(Unew_gpu);
}