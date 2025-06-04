#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Problem parameters */
#define c 0.1          /* Diffusion coefficient */
#define N 100          /* Number of interior points per dimension */
#define max_steps 10000 /* Number of time iterations */
#define delta_s (1.0 / (N + 1))
#define delta_t ((delta_s * delta_s) / (4.0 * c))

/* Boundary temperatures */
double alpha0(double y) { return 0.0; }  /* u(t,0,y) */
double alpha1(double y) { return sqrt(y/4); }  /* u(t,1,y) */
double beta0(double x)  { return 100*(0.7+0.3*sin(5*3.14*x/4)); }  /* u(t,x,0) */
double beta1(double x)  { return 100*cbrt(x/4); }  /* u(t,x,1) */

/* Initial interior temperature */
double initial(double x, double y) {
    return 0.0;
}

void Derivative(double *u_new, double *u) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double uij = u[i*N + j];
            double down = (i == N - 1) ? alpha1((j + 1) * delta_s) : u[(i + 1)*N + j];
            double up = (i == 0) ? alpha0((j + 1) * delta_s) : u[(i - 1)*N + j];
            double left = (j == 0) ? beta0((i + 1) * delta_s) : u[i*N + (j - 1)];
            double right = (j == N - 1) ? beta1((i + 1) * delta_s) : u[i*N + (j + 1)];
            double laplace = right + left + up + down - 4.0 * uij;
            u_new[i*N + j] = uij + c * delta_t / (delta_s * delta_s) * laplace;
        }
    }
}

void InputData(double *u) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            u[i*N + j] = initial((i + 1) * delta_s, (j + 1) * delta_s);
        }
    }
}

int main() {
    /* Allocate flat arrays of size N×N */
    double *u_old = malloc(N * N * sizeof(double));
    double *u_new = malloc(N * N * sizeof(double));

    if (u_old == NULL || u_new == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    /* Initialize u_old: set boundaries and initial interior */
    InputData(u_old);

    /* Open the output file */
    FILE *fp = fopen("heat_output.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        free(u_old);
        free(u_new);
        return 1;
    }

    /* Time-stepping loop */
    for (int step = 0; step < max_steps; step++) {
        Derivative(u_new, u_old);

        /* Write the current state to the file */
        for (int i = 0; i < N * N; i++) {
            fprintf(fp, "%.6f ", u_new[i]);
        }
        fprintf(fp, "\n");

        /* Swap pointers for next iteration */
        double *temp = u_old;
        u_old = u_new;
        u_new = temp;
    }

    /* Close the file and free memory */
    fclose(fp);
    free(u_old);
    free(u_new);

    return 0;
}
