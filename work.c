#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

/*
 * Un paso del método de Jacobi para la ecuación de Poisson
 *
 *   Argumentos:
 *     - N,M: dimensiones de la malla
 *     - Entrada: x es el vector de la iteración anterior, b es la parte derecha del sistema
 *     - Salida: t es el nuevo vector
 *
 *   Se asume que x,b,t son de dimensión (N+2)*(M+2), se recorren solo los puntos interiores
 *   de la malla, y en los bordes están almacenadas las condiciones de frontera (por defecto 0).
 */
void jacobi_step(int nloc, int M, double *x, double *b, double *t, MPI_Request reqs[], MPI_Status stats[])
{
  int i, j, ld = M + 2;

  // comunicacion persistente
  MPI_Startall(4, reqs);

  // calculos no dependientes entre procesos
  for (i = 2; i < nloc; i++) {
    for (j = 1; j <= M; j++) {
      t[i * ld + j] = (b[i * ld + j] + x[(i + 1) * ld + j] + x[(i - 1) * ld + j] + x[i * ld + (j + 1)] + x[i * ld + (j - 1)]) / 4.0;
    }
  }

  // esperamos a las comunicacines
  MPI_Waitall(4, reqs, stats);

  // calculos dependientes entre procesos
  // primera fila
  const int f1 = 1;
  for (j = 1; j <= M; j++) {
    t[f1 * ld +j] = (b[f1 * ld + j] + x[(f1 + 1) * ld + j] + x[(f1 - 1) * ld + j] + x[f1 * ld + (j + 1)] + x[f1 * ld + (j - 1)]) / 4.0;
  }

  // ultima fila
  const int flast = nloc;
  for (j = 1; j <= M; j++) {
    t[flast * ld + j] = (b[flast * ld + j] + x[(flast + 1) * ld + j] + x[(flast - 1) * ld + j] + x[flast * ld + (j + 1)] + x[flast * ld +(j - 1)]) / 4.0;
  }
}

/*
 * Método de Jacobi para la ecuación de Poisson
 *
 *   Suponemos definida una malla de (N+2)x(M+2) puntos, donde los puntos
 *   de la frontera tienen definida una condición de contorno.
 *
 *   Esta función resuelve el sistema Ax=b mediante el método iterativo
 *   estacionario de Jacobi. La matriz A no se almacena explícitamente y
 *   se aplica de forma implícita para cada punto de la malla. El vector
 *   x representa la solución de la ecuación de Poisson en cada uno de los
 *   puntos de la malla (incluyendo el contorno). El vector b es la parte
 *   derecha del sistema de ecuaciones, y contiene el término h^2*f.
 *
 *   Suponemos que las condiciones de contorno son igual a 0 en toda la
 *   frontera del dominio.
 */
void jacobi_poisson(int N, int M, double *x, double *b, MPI_Request reqs[], MPI_Status stats[])
{
  int i, j, k, ld = M + 2, conv, maxit = 10000;
  double *t, s, sloc, tol = 1e-6;

  t = (double*)calloc((N + 2) * (M + 2), sizeof(double));

  k = 0;
  conv = 0;

  while (!conv && k < maxit) {

    // calcula siguiente vector
    jacobi_step(N, M, x, b, t, reqs, stats);
    
    // criterio de parada: ||x_{k}-x_{k+1}||<tol
    s = 0.0;
    sloc = 0.0;
    for (i = 1; i <= N; i++) {
      for (j = 1; j <= M; j++) {
        sloc += (x[i * ld + j] - t[i * ld + j]) * (x[i * ld + j] - t[i * ld + j]);
      }
    }
  
    // calculamos para todo proceso la suma total de la s local de cada proceso
    MPI_Allreduce(&sloc, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    conv = (sqrt(s) < tol);
    
    // siguiente iteración
    k = k + 1;
    for (i = 1; i <= N; i++) {
      for (j = 1; j <= M; j++) {
        x[i * ld + j] = t[i * ld + j];
      }
    }
  }
  free(t);
}

int main(int argc, char **argv)
{
  int i, j, N = 60, M = 60, ld, size, rank, next, prev;
  double *x, *b, *t, h = 0.01, f = 1.5, t1, t2;
  MPI_Status stats[4];
  MPI_Request reqs[4];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if (rank == 0){
    t1 = MPI_Wtime();
  }

  // obtener rank procesos vecinos
  if (rank == 0) prev = MPI_PROC_NULL;
  else prev = rank - 1;
  if (rank == size - 1) next = MPI_PROC_NULL;
  else next = rank + 1;

  // extracción de argumentos
  if (argc > 1) { // el usuario ha indicado el valor de N
    if ((N = atoi(argv[1])) < 0) N = 60;
  }
  if (argc > 2) { // el usuario ha indicado el valor de M
    if ((M = atoi(argv[2])) < 0) M = 60;
  }

  // leading dimension
  ld = M + 2;
  int nloc = N / size;

  // reserva de memoria
  x = (double*)calloc((nloc + 2) * (M + 2), sizeof(double));
  b = (double*)calloc((nloc + 2) * (M + 2), sizeof(double));
  t = (double*)calloc(N * (M + 2), sizeof(double));

  // inicializar datos
  for (i = 1; i <= nloc; i++) {
    for (j = 1; j <= M; j++) {
      b[i * ld + j] = h * h * f;  // suponemos que la función f es constante en todo el dominio
    }
  }
  
  // comunicaciones persistentes 
  // con vecino anterior
  MPI_Recv_init(x, ld, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
  MPI_Send_init(x + ld, ld, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[1]);
  
  // con vecino siguiente
  MPI_Send_init(x + ld * nloc, ld, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[2]);
  MPI_Recv_init(x + ld * (nloc + 1), ld, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[3]);

  // resolución del sistema por el método de Jacobi
  jacobi_poisson(nloc, M, x, b, reqs, stats);

  // reunimos los resultados de cada proceso
  MPI_Gather(x + ld, nloc * ld, MPI_DOUBLE, t + rank * nloc * ld, nloc * ld, MPI_DOUBLE, 0,MPI_COMM_WORLD);

  if(rank == 0){
    // pintamos matriz resultado
    for (i = 0; i < N; i++) {
      for (j = 1; j <= M; j++) {
        printf("%g ", t[i * ld + j]);
      }
      printf("\n");
    }
  
    t2 = MPI_Wtime();
    printf("Tiempo transcurrido %f s.\n", t2 - t1);
  }
  
  MPI_Request_free(&reqs[0]);
  MPI_Request_free(&reqs[1]);
  MPI_Request_free(&reqs[2]);
  MPI_Request_free(&reqs[3]);

  free(x);
  free(b);
  free(t);
  MPI_Finalize();

  return 0;
}
