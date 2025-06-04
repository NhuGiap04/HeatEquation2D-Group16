#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* problem parameters */
#define c 0.1          /* diffusion coefficient */
#define N 100       /* number of interior points per dimension */
#define max_steps 10000 /* number of time iterations */
#define delta_s 1.0 / (N + 1)
#define delta_t (delta_s * delta_s) / (4.0 * c)

/* boundary temperatures */
double alpha0(double y) { return 10.0; }  /* u(t,0,y) */
double alpha1(double y) { return 40.0; }  /* u(t,1,y) */
double beta0(double x)  { return 30.0; }  /* u(t,x,0) */
double beta1(double x)  { return 50.0; }  /* u(t,x,1) */

void DisplayArray(double *array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", array[i * cols + j]);
        }
        printf("\n");
    }
}

/* initial interior temperature */
double initial(double x, double y) {
    /* here: start from zero everywhere inside */
    return 0.0;
}

void Derivative(double *u_new, double *u) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double uij = u[i*N + j];
            double down = (i==N-1) ? alpha1((j+1) * delta_s) : u[(i+1)*N + j];
            double up = (i==0) ? alpha0((j+1) * delta_s) : u[(i-1)*N + j];
            double left = (j==0) ? beta0((i+1) * delta_s) : u[i*N + (j-1)];
            double right = (j==N-1) ? beta1((i+1) * delta_s) : u[i*N + (j+1)];
            double laplace = right + left + up + down - 4.0 * uij;
            u_new[i*N + j] = uij + c * delta_t / (delta_s * delta_s) * laplace;
        }
    }
}

void InputData(double *u){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            u[i*N + j] = initial((i+1) * delta_s, (j+1) * delta_s);
        }
    }
}

int main() {
    /* allocate flat arrays of size N×N */
    double *u_old = malloc(N * N * sizeof(double));
    double *u_new = malloc(N * N * sizeof(double));

    /* initialize u_old: set boundaries and initial interior */
    InputData(u_old);

    /* time-stepping loop */
    for (int step = 0; step < max_steps; step++) {
        Derivative(u_new, u_old);
        
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                u_old[i*N + j] = u_new[i*N + j];
            }
        }
    }

    /* output final temperature field to stdout */
    printf("Number of iterations: %d\n", max_steps);
    DisplayArray(u_new, N, N);

    return 0;
}
