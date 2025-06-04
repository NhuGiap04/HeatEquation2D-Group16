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

void LocalDerivativeCols(float *u_new, float *u, int Nc, float *u_left, float *u_right, int Rank){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < Nc; j++){
            float uij = u[i*Nc + j];
            float down = (i==N-1) ? alpha1((j+1+Rank*Nc) * delta_s) : u[(i+1)*Nc + j];
            float up = (i==0) ? alpha0((j+1+Rank*Nc) * delta_s) : u[(i-1)*Nc + j];
            float left = (j==0) ? u_left[i]: u[i*Nc + (j-1)];
            float right = (j==Nc-1) ? u_right[i] : u[i*Nc + (j+1)];
            float laplace = right + left + up + down - 4.0 * uij;
            u_new[i*Nc + j] = uij + c * delta_t / (delta_s * delta_s) * laplace;
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

    MPI_Datatype column_block, column_block_resized;
    MPI_Type_vector(   /* count = # of blocks (rows)     */ N,
                    /* blocklength = # of floats per block */ Nc,
                    /* stride = # of floats between starts */ N,
                    MPI_FLOAT,
                    &column_block);
    MPI_Type_commit(&column_block);

    MPI_Type_create_resized(
        column_block,
        /* lower bound */ 0,
        /* new extent  */ Nc * sizeof(float),
        &column_block_resized
    );
    MPI_Type_commit(&column_block_resized);

    // now each rank gets *one* of these column‐blocks
    MPI_Scatter(
        u_old,       /* sendbuf at root */
        1, column_block_resized,
        u_old_c,     /* local recvbuf is already contiguous */
        N*Nc, MPI_FLOAT,
        0, MPI_COMM_WORLD);

    MPI_Datatype column_type;
    MPI_Type_vector(N, 1, Nc, MPI_FLOAT, &column_type);
    MPI_Type_commit(&column_type);

    float *u_left  = malloc(N * sizeof(float));
    float *u_right = malloc(N * sizeof(float));

    int step = 0;

    /* time-stepping loop */
    while (step < max_steps) {
        // thread-safe step increment
        if (Rank == 0) step++;
        MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // communicate u_left
        if (Rank == 0){
            for (int i = 0; i < N; i++){
                u_left[i] = beta0((i+1) * delta_s);
            }
            MPI_Send(u_old_c + Nc - 1, 1, column_type, Rank + 1, Rank + 1000, MPI_COMM_WORLD);
        }
        else if (Rank == NP - 1){
            MPI_Recv(u_left, N, MPI_FLOAT, Rank - 1, Rank - 1 + 1000, MPI_COMM_WORLD, &Stat);
        }
        else{
            MPI_Sendrecv(u_old_c + Nc - 1, 1, column_type, Rank + 1, Rank + 1000, u_left, N, MPI_FLOAT, Rank - 1, Rank - 1 + 1000, MPI_COMM_WORLD, &Stat);
        }

        // communicate u_right
        if (Rank == NP - 1){
            for (int i = 0; i < N; i++){
                u_right[i] = beta1((i+1) * delta_s);
            }
            MPI_Send(u_old_c, 1, column_type, Rank - 1, Rank + 100, MPI_COMM_WORLD);
        }
        else if (Rank == 0){
            MPI_Recv(u_right, N, MPI_FLOAT, Rank + 1, Rank + 1 + 100, MPI_COMM_WORLD, &Stat);
        }
        else{
            MPI_Sendrecv(u_old_c, 1, column_type, Rank - 1, Rank + 100, u_right, N, MPI_FLOAT, Rank + 1, Rank + 1 + 100, MPI_COMM_WORLD, &Stat);
        }

        LocalDerivativeCols(u_new_c, u_old_c, Nc, u_left, u_right, Rank);

        for (int i = 0; i < N; i++){
            for (int j = 0; j < Nc; j++){
                u_old_c[i*Nc + j] = u_new_c[i*Nc + j];
            }
        }
    }

    MPI_Gather(u_new_c, N*Nc, MPI_FLOAT, u_new, 1, column_block_resized, 0, MPI_COMM_WORLD);

    MPI_Type_free(&column_block);
    MPI_Type_free(&column_block_resized);
    MPI_Type_free(&column_type);

    /* output final temperature field to stdout */
    if (Rank == 0){
        printf("Number of iterations: %d\n", max_steps);
        DisplayArray(u_new, N, N);
    }
    MPI_Finalize();
    return 0;
}
