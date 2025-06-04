//#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <time.h>
#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
#include <float.h>
int N = 100;  /* mesh size = M = n  */
#define c 0.1 /* diffusion coefficient */
#define  max_steps   10000  /* number of time iterations */
/* choose Δt ≤ (Δs)^2/(2c) for stability; here a bit more conservative */
// #define  delta_s      1.0 / (N + 1)
// #define  delta_t      (delta_s * delta_s) / (4.0 * c)
#define BlockSizeX 4
#define BlockSizeY 4
// #define GridSizeX  (N - 1)/BlockSizeX + 1
// #define GridSizeY  (N - 1)/BlockSizeY + 1
// #define ThreadSizeX (N - 1)/(GridSizeX*BlockSizeX) + 1
// #define ThreadSizeY (N - 1)/(GridSizeY*BlockSizeY) + 1

/* boundary temperatures */
__device__ double alpha0(double y) { return 0.0; }  /* u(t,0,y) */
__device__ double alpha1(double y) {  return sqrt(y / 4); }  /* u(t,1,y) */
__device__ double beta0(double x)  { return 100 * (0.7 + 0.3 * sin(5 * 3.14 * x / 4)); }  /* u(t,x,0) */
__device__ double beta1(double x)  { return 100 * cbrt(x / 4); }  /* u(t,x,1) */

/* initial interior temperature */
double Initial(double x, double y)
{
    return 0.0;
}

void InputData(double *U, double delta_s)
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
__global__ void Derivative(double *U, double *dU, double delta_s, int N){
    // T = array N x N
    // dT = array N x N
    int i, j;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N))
    {
        double uij = *(U+i*N+j);
        double down = (i==N-1) ? alpha1((j+1) * delta_s) : *(U + (i+1)*N+ j);
        double up = (i==0) ? alpha0((j+1) * delta_s) : *(U + (i-1)*N + j);
        double left = (j==0) ? beta0((i+1) * delta_s) : *(U + i*N + (j-1));
        double right = (j==N-1) ? beta1((i+1) * delta_s) : *(U + i*N + (j+1));
        double laplace = right + left + up + down - 4.0 * uij;

        *(dU+i*N+j) = laplace; 
    }
}


__global__ void SolvingODE(double *U,double *dU, double delta_s, double delta_t, int N) 
{

    int i, j;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N))
        *(U+i*N+j) = *(U+i*N+j) + c * delta_t / (delta_s * delta_s) * *(dU+i*N+j);
}


int main(int argc, char *argv[]) {
    // Parse command line arguments
    int opt;
    static struct option long_options[] = {
        {"number-interior-points", required_argument, 0, 'n'},
        {"diffusion-coefficient", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "n:c:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [--number-interior-points N] [--diffusion-coefficient c]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    double delta_s = 1.0 / (N + 1);
    double delta_t = 0.5 * (delta_s * delta_s) / (4.0 * c);
    int GridSizeX = (N - 1) / BlockSizeX + 1;
    int GridSizeY = (N - 1) / BlockSizeY + 1;

    /* Start timing */
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    int i, j, t;
    double *Ucpu, *dUcpu;
    Ucpu  = (double *)malloc((N * N)*sizeof(double));
    dUcpu = (double *)malloc((N * N)*sizeof(double));
    InputData(Ucpu, delta_s);
    // CUDA code
    //1. Delare and Allocate Mem on GPU
    int *Ngpu;
    double *Ugpu,*dUgpu,*delta_sgpu, *delta_tgpu;
    cudaMalloc((void**)&Ugpu ,(N * N)*sizeof(double));
    cudaMalloc((void**)&dUgpu,(N * N)*sizeof(double));
    cudaMalloc((void**)&Ngpu, sizeof(int));
    cudaMalloc((void**)&delta_sgpu, sizeof(double));
    cudaMalloc((void**)&delta_tgpu, sizeof(double));

    //2. Copy Input from CPU to GPU
    cudaMemcpy(Ugpu, Ucpu,(N * N)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(Ngpu, &N, sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(delta_sgpu, &delta_s, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_tgpu, &delta_t, sizeof(double), cudaMemcpyHostToDevice);

    //3. Define Block and Thread Structure
    dim3 dimGrid(GridSizeX,GridSizeY);
    dim3 dimBlock(BlockSizeX,BlockSizeY);

    for (t=0; t<max_steps; t++) {
        Derivative<<<dimGrid,dimBlock>>>(Ugpu, dUgpu, (double)delta_s, N);
        SolvingODE<<<dimGrid,dimBlock>>>(Ugpu, dUgpu, (double)delta_s, (double)delta_t, N);
        cudaDeviceSynchronize();
    }

    //5. Copy Output from GPU to CPU
    cudaMemcpy(Ucpu, Ugpu, (N * N)*sizeof(double),cudaMemcpyDeviceToHost);
    for (i = 0; i < N; i++ )
        for (j = 0; j < N; j++) printf("%f \n",*(Ucpu+i*N+j));

    /* End timing */
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    /* Display configuration and timing */
    printf("Simulation completed.\n");
    printf("Number of interior points (N): %d\n", N);
    printf("Diffusion coefficient (c): %.6f\n", c);
    printf("Total execution time: %.6f seconds\n", elapsed_time);
    printf("Number of iterations: %d\n", max_steps);
    
    //6. Free Mem on CPU and GPU
    free(Ucpu); free(dUcpu);
    cudaFree(Ugpu);cudaFree(dUgpu);
}