#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <time.h>

/* Default problem parameters */
int N = 100;          /* Number of interior points per dimension */
double c = 0.1;       /* Diffusion coefficient */
int max_steps = 10000; /* Number of time iterations */

/* Boundary temperatures */
double alpha0(double y) { return 0.0; }  /* u(t,0,y) */
double alpha1(double y) { return sqrt(y / 4); }  /* u(t,1,y) */
double beta0(double x)  { return 100 * (0.7 + 0.3 * sin(5 * 3.14 * x / 4)); }  /* u(t,x,0) */
double beta1(double x)  { return 100 * cbrt(x / 4); }  /* u(t,x,1) */

/* Initial interior temperature */
double initial(double x, double y) {
    return 0.0;
}

void Derivative(double *u_new, double *u, double delta_s, double delta_t) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double uij = u[i * N + j];
            double down = (i == N - 1) ? alpha1((j + 1) * delta_s) : u[(i + 1) * N + j];
            double up = (i == 0) ? alpha0((j + 1) * delta_s) : u[(i - 1) * N + j];
            double left = (j == 0) ? beta0((i + 1) * delta_s) : u[i * N + (j - 1)];
            double right = (j == N - 1) ? beta1((i + 1) * delta_s) : u[i * N + (j + 1)];
            double laplace = right + left + up + down - 4.0 * uij;
            u_new[i * N + j] = uij + c * delta_t / (delta_s * delta_s) * laplace;
        }
    }
}

void InputData(double *u, double delta_s) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i * N + j] = initial((i + 1) * delta_s, (j + 1) * delta_s);
        }
    }
}

void print_usage(char *program_name) {
    printf("Usage: %s [--number-interior-points N] [--diffusion-coefficient C]\n", program_name);
    printf("Defaults: N = 100, C = 0.1\n");
}

int main(int argc, char **argv) {
    /* Parse command-line arguments */
    static struct option long_options[] = {
        {"number-interior-points", required_argument, 0, 'n'},
        {"diffusion-coefficient", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };
    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "n:c:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                if (N <= 0) {
                    fprintf(stderr, "Error: Number of interior points must be positive.\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'c':
                c = atof(optarg);
                if (c <= 0) {
                    fprintf(stderr, "Error: Diffusion coefficient must be positive.\n");
                    return EXIT_FAILURE;
                }
                break;
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    double delta_s = 1.0 / (N + 1);
    double delta_t = 0.5 * (delta_s * delta_s) / (4.0 * c);

    /* Allocate flat arrays of size N×N */
    double *u_old = malloc(N * N * sizeof(double));
    double *u_new = malloc(N * N * sizeof(double));

    if (u_old == NULL || u_new == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    /* Initialize u_old: set boundaries and initial interior */
    InputData(u_old, delta_s);

    /* Open the output file */
    FILE *fp = fopen("heat_output.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        free(u_old);
        free(u_new);
        return EXIT_FAILURE;
    }

    /* Start timing */
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    /* Time-stepping loop */
    for (int step = 0; step < max_steps; step++) {
        Derivative(u_new, u_old, delta_s, delta_t);

        /* Write the current state to the file */
        // for (int i = 0; i < N * N; i++) {
        //     fprintf(fp, "%.6f ", u_new[i]);
        // }
        // fprintf(fp, "\n");

        /* Swap pointers for next iteration */
        double *temp = u_old;
        u_old = u_new;
        u_new = temp;
    }

    /* End timing */
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    /* Close the file and free memory */
    // display u_new
    printf("Final temperature distribution:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.6f ", u_new[i * N + j]);
        }
    printf("\n");
}
    fclose(fp);
    free(u_old);
    free(u_new);

    /* Display configuration and timing */
    printf("Simulation completed.\n");
    printf("Number of interior points (N): %d\n", N);
    printf("Diffusion coefficient (c): %.6f\n", c);
    printf("Total execution time: %.6f seconds\n", elapsed_time);

    return EXIT_SUCCESS;
}
