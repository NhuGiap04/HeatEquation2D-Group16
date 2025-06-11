#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <unistd.h>
#include <mpi.h>
#include <time.h>

/* problem parameters */
float c = 0.1f;          /* diffusion coefficient */
int N = 100;       /* number of interior points per dimension */
#define max_steps 10000 /* number of time iterations */

/* boundary temperatures */
float alpha0(float y) { return 10.0; }  /* u(t,0,y) */
float alpha1(float y) { return 40.0; }  /* u(t,1,y) */
float beta0(float x)  { return 30.0; }  /* u(t,x,0) */
float beta1(float x)  { return 50.0; }  /* u(t,x,1) */

void DisplayArray(float* array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", array[i * cols + j]);
        }
        printf("\n");
    }
}

/* initial interior temperature */
float initial(float x, float y) {
    return 0.0;
}

void LocalJacobi(float* u_new, float* u, int Nc, float* u_down, float* u_up, int Rank, float delta_s, float delta_t) {
    for (int i = 0; i < Nc; i++) {
        for (int j = 0; j < N; j++) {
            float down = (i == Nc - 1) ? u_down[j] : u[(i + 1) * N + j];
            float up = (i == 0) ? u_up[j] : u[(i - 1) * N + j];
            float left = (j == 0) ? beta0((i + 1 + Rank * Nc) * delta_s) : u[i * N + (j - 1)];
            float right = (j == N - 1) ? beta1((i + 1 + Rank * Nc) * delta_s) : u[i * N + (j + 1)];
            u_new[i * N + j] = 0.25 * (left + right + up + down);
        }
    }
}

void InputData(float* u, float delta_s) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i * N + j] = initial((i + 1) * delta_s, (j + 1) * delta_s);
        }
    }
}

int main(int argc, char** argv) {
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
            case 'c':
                c = atof(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [--number-interior-points N] [--diffusion-coefficient c]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    double delta_s = 1.0 / (N + 1);
    double delta_t = 0.5 * (delta_s * delta_s) / (4.0 * c);

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    int NP, Rank;
    MPI_Init(&argc, &argv);
    MPI_Status Stat;
    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

    float* u_old = malloc(N * N * sizeof(float));
    float* u_new = malloc(N * N * sizeof(float));
    FILE* fp = NULL;

    if (Rank == 0) {
        InputData(u_old, delta_s);
        fp = fopen("heat_output.txt", "w");
    }

    int Nc = N / NP;
    float* u_old_c = malloc(N * Nc * sizeof(float));
    float* u_new_c = malloc(N * Nc * sizeof(float));

    MPI_Scatter(u_old, N * Nc, MPI_FLOAT, u_old_c, N * Nc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float* u_down = malloc(N * sizeof(float));
    float* u_up = malloc(N * sizeof(float));

    int step = 0;

    while (step < max_steps) {
        if (Rank == 0) step++;
        MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (Rank == 0) {
            for (int i = 0; i < N; i++) {
                u_up[i] = alpha0((i + 1) * delta_s);
            }
            MPI_Send(u_old_c + (Nc - 1) * N, N, MPI_FLOAT, Rank + 1, Rank + 1000, MPI_COMM_WORLD);
        } else if (Rank == NP - 1) {
            MPI_Recv(u_up, N, MPI_FLOAT, Rank - 1, Rank - 1 + 1000, MPI_COMM_WORLD, &Stat);
        } else {
            MPI_Sendrecv(u_old_c + (Nc - 1) * N, N, MPI_FLOAT, Rank + 1, Rank + 1000,
                         u_up, N, MPI_FLOAT, Rank - 1, Rank - 1 + 1000, MPI_COMM_WORLD, &Stat);
        }

        if (Rank == NP - 1) {
            for (int i = 0; i < N; i++) {
                u_down[i] = alpha1((i + 1) * delta_s);
            }
            MPI_Send(u_old_c, N, MPI_FLOAT, Rank - 1, Rank + 100, MPI_COMM_WORLD);
        } else if (Rank == 0) {
            MPI_Recv(u_down, N, MPI_FLOAT, Rank + 1, Rank + 1 + 100, MPI_COMM_WORLD, &Stat);
        } else {
            MPI_Sendrecv(u_old_c, N, MPI_FLOAT, Rank - 1, Rank + 100,
                         u_down, N, MPI_FLOAT, Rank + 1, Rank + 1 + 100, MPI_COMM_WORLD, &Stat);
        }

        LocalJacobi(u_new_c, u_old_c, Nc, u_down, u_up, Rank, delta_s, delta_t);

        for (int i = 0; i < Nc; i++) {
            for (int j = 0; j < N; j++) {
                u_old_c[i * N + j] = u_new_c[i * N + j];
            }
        }

        MPI_Gather(u_old_c, N * Nc, MPI_FLOAT, u_new, N * Nc, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (Rank == 0 && fp != NULL) {
            for (int i = 0; i < N * N; i++) {
                fprintf(fp, "%.6f ", u_new[i]);
            }
            fprintf(fp, "\n");
        }
    }

    if (Rank == 0 && fp != NULL) {
        fclose(fp);
    }

    if (Rank == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                              (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        printf("Simulation completed.\n");
        printf("Number of interior points (N): %d\n", N);
        printf("Diffusion coefficient (c): %.6f\n", c);
        printf("Total execution time: %.6f seconds\n", elapsed_time);
        printf("Number of iterations: %d\n", max_steps);
        // DisplayArray(u_new, N, N);
    }

    free(u_old);
    free(u_new);
    free(u_old_c);
    free(u_new_c);
    free(u_down);
    free(u_up);

    MPI_Finalize();
    return 0;
}
