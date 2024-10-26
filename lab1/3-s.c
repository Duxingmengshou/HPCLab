#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4

void printMatrix(int *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = N / size;
    int *local_matrix = (int *)malloc(local_rows * N * sizeof(int));
    int *global_matrix = NULL;
    int *transposed_matrix = NULL;

    if (rank == 0)
    {
        global_matrix = (int *)malloc(N * N * sizeof(int));
        for (int i = 0; i < N * N; i++)
        {
            global_matrix[i] = i;
        }
        printMatrix(global_matrix, N, N);
    }

    MPI_Scatter(global_matrix, local_rows * N, MPI_INT, local_matrix, local_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_transposed = (int *)malloc(local_rows * N * sizeof(int));
    for (int i = 0; i < local_rows; i++)
    {
        for (int j = 0; j < N; j++)
        {
            local_transposed[j * local_rows + i] = local_matrix[i * N + j];
        }
    }

    if (rank == 0)
    {
        transposed_matrix = (int *)malloc(N * N * sizeof(int));
    }

    MPI_Gather(local_transposed, local_rows * N, MPI_INT, transposed_matrix, local_rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Transposed Matrix:\n");
        printMatrix(transposed_matrix, N, N);
        free(global_matrix);
        free(transposed_matrix);
    }

    free(local_matrix);
    free(local_transposed);

    MPI_Finalize();
    return 0;
}