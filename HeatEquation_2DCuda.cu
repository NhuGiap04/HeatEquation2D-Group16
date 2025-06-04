#include <stdio.h>
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

//=========================
__global__ void Derivative(float *U, float *dU){
    // T = array N x N
    // dT = array N x N
    int i, j;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N))
    {
        float uij = *(U+i*N+j);
        float down = (i==N-1) ? alpha1((j+1) * delta_s) : *(U + (i+1)*N+ j);
        float up = (i==0) ? alpha0((j+1) * delta_s) : *(U + (i-1)*N + j);
        float left = (j==0) ? beta0((i+1) * delta_s) : *(U + i*N + (j-1));
        float right = (j==N-1) ? beta1((i+1) * delta_s) : *(U + i*N + (j+1));
        float laplace = right + left + up + down - 4.0 * uij;

        *(dU+i*N+j) = laplace; 
    }
}


__global__ void SolvingODE(float *U,float *dU) 
{

    int i, j;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N))
        *(U+i*N+j) = *(U+i*N+j) + c * delta_t / (delta_s * delta_s) * *(dU+i*N+j);
}


int main() {
    int i, j, t;
    float *Ucpu, *dUcpu;
    Ucpu  = (float *)malloc((N * N)*sizeof(float));
    dUcpu = (float *)malloc((N * N)*sizeof(float));
    InputData(Ucpu);
    // CUDA code
    //1. Delare and Allocate Mem on GPU
    float *Ugpu,*dUgpu;
    cudaMalloc((void**)&Ugpu ,(N * N)*sizeof(float));
    cudaMalloc((void**)&dUgpu,(N * N)*sizeof(float));

    //2. Copy Input from CPU to GPU
    cudaMemcpy(Ugpu, Ucpu,(N * N)*sizeof(float),cudaMemcpyHostToDevice);

    //3. Define Block and Thread Structure
    dim3 dimGrid(GridSizeX,GridSizeY);
    dim3 dimBlock(BlockSizeX,BlockSizeY);

    for (t=0; t<max_steps; t++) {
        Derivative<<<dimGrid,dimBlock>>>(Ugpu,dUgpu);
        SolvingODE<<<dimGrid,dimBlock>>>(Ugpu,dUgpu);
        cudaDeviceSynchronize();
    }

    //5. Copy Output from GPU to CPU
    cudaMemcpy(Ucpu, Ugpu, (N * N)*sizeof(float),cudaMemcpyDeviceToHost);
    for (i = 0; i < N; i++ )
        for (j = 0; j < N; j++) printf("%f \n",*(Ucpu+i*N+j));
    //6. Free Mem on CPU and GPU
    free(Ucpu); free(dUcpu);
    cudaFree(Ugpu);cudaFree(dUgpu);
}