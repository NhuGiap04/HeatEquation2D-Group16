#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* problem parameters */
#define c 0.1          /* diffusion coefficient */
#define N 100       /* number of interior points per dimension */
#define max_steps 10000 /* number of time iterations */
#define delta_s 1.0 / (N + 1)
#define delta_t (delta_s * delta_s) / (4.0 * c)

/* boundary temperatures */
float alpha0(float y) { return 10.0; }  /* u(t,0,y) */
float alpha1(float y) { return 40.0; }  /* u(t,1,y) */
float beta0(float x)  { return 30.0; }  /* u(t,x,0) */
float beta1(float x)  { return 50.0; }  /* u(t,x,1) */

void DisplayArray(float *array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", array[i * cols + j]);
        }
        printf("\n");
    }
}

/* initial interior temperature */
float initial(float x, float y) {
    /* here: start from zero everywhere inside */
    return 0.0;
}

void LocalDerivative(float *u_new, float *u, int Nc, float *u_down, float *u_up, int Rank){
    for (int i = 0; i < Nc; i++){
        for (int j = 0; j < N; j++){
            float uij = u[i*N + j];
            float down = (i==Nc-1) ? u_down[j] : u[(i+1)*N + j];
            float up = (i==0) ? u_up[j] : u[(i-1)*N + j];
            float left = (j==0) ? beta0((i+1+Rank*Nc) * delta_s) : u[i*N + (j-1)];
            float right = (j==N-1) ? beta1((i+1+Rank*Nc) * delta_s) : u[i*N + (j+1)];
            float laplace = right + left + up + down - 4.0 * uij;
            u_new[i*N + j] = uij + c * delta_t / (delta_s * delta_s) * laplace;
        }
    }
}

void InputData(float *u){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            u[i*N + j] = initial((i+1) * delta_s, (j+1) * delta_s);
        }
    }
}

int main(int argc, char** argv) {

    int NP, Rank;
    MPI_Init(&argc, &argv);
    MPI_Status Stat;
    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

    /* allocate flat arrays of size N×N */
    float *u_old = malloc(N * N * sizeof(float));
    float *u_new = malloc(N * N * sizeof(float));

    if (Rank == 0) InputData(u_old);

    // domain decomposition
    int Nc;
    Nc = N / NP;
    float *u_old_c = malloc(N * Nc * sizeof(float));
    float *u_new_c = malloc(N * Nc * sizeof(float));

    MPI_Scatter(u_old, N * Nc, MPI_FLOAT, u_old_c, N * Nc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float *u_down = malloc(N * sizeof(float));
    float *u_up = malloc(N * sizeof(float));

    int step = 0;

    /* time-stepping loop */
    while (step < max_steps) {
        // thread-safe step increment
        if (Rank == 0) step++;
        MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // communicate u_up
        if (Rank == 0){
            for (int i = 0; i < N; i++){
                u_up[i] = alpha0((i+1) * delta_s);
            }
            MPI_Send(u_old_c + (Nc-1)*N, N, MPI_FLOAT, Rank + 1, Rank + 1000, MPI_COMM_WORLD);
        }
        else if (Rank == NP - 1){
            MPI_Recv(u_up, N, MPI_FLOAT, Rank - 1, Rank - 1 + 1000, MPI_COMM_WORLD, &Stat);
        }
        else{
            MPI_Sendrecv(u_old_c + (Nc-1)*N, N, MPI_FLOAT, Rank + 1, Rank + 1000, u_up, N, MPI_FLOAT, Rank - 1, Rank - 1 + 1000, MPI_COMM_WORLD, &Stat);
        }

        // communicate u_down
        if (Rank == NP - 1){
            for (int i = 0; i < N; i++){
                u_down[i] = alpha1((i+1) * delta_s);
            }
            MPI_Send(u_old_c, N, MPI_FLOAT, Rank - 1, Rank + 100, MPI_COMM_WORLD);
        }
        else if (Rank == 0){
            MPI_Recv(u_down, N, MPI_FLOAT, Rank + 1, Rank + 1 + 100, MPI_COMM_WORLD, &Stat);
        }
        else{
            MPI_Sendrecv(u_old_c, N, MPI_FLOAT, Rank - 1, Rank + 100, u_down, N, MPI_FLOAT, Rank + 1, Rank + 1 + 100, MPI_COMM_WORLD, &Stat);
        }

        LocalDerivative(u_new_c, u_old_c, Nc, u_down, u_up, Rank);

        for (int i = 0; i < Nc; i++){
            for (int j = 0; j < N; j++){
                u_old_c[i*N + j] = u_new_c[i*N + j];
            }
        }
    }

    MPI_Gather(u_new_c, N * Nc, MPI_FLOAT, u_new, N * Nc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* output final temperature field to stdout */
    if (Rank == 0){
        printf("Number of iterations: %d\n", max_steps);
        DisplayArray(u_new, N, N);
    }
    MPI_Finalize();
    return 0;
}
