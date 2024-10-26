#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define N 3 // 矩阵维度

void Matrix_print(double *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%.2f ", A[i * n + j]);
        printf("\n");
    }
    printf("\n");
}

void invert_matrix(double *A, double *I, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            I[i * n + j] = (i == j) ? 1.0 : 0.0; // 初始化单位矩阵
        }
    }

    for (int i = 0; i < n; i++)
    {
        double pivot = A[i * n + i];
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] /= pivot; // 归一化行
            I[i * n + j] /= pivot; // 同时归一化单位矩阵
        }
        for (int k = 0; k < n; k++)
        {
            if (k != i)
            {
                double factor = A[k * n + i];
                for (int j = 0; j < n; j++)
                {
                    A[k * n + j] -= factor * A[i * n + j]; // 消元
                    I[k * n + j] -= factor * I[i * n + j]; // 更新单位矩阵
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    int my_rank, comm_sz;
    double *Matrix_A, *Matrix_I;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    Matrix_A = (double *)malloc(N * N * sizeof(double));
    Matrix_I = (double *)malloc(N * N * sizeof(double));

    // 给矩阵A赋值
    if (my_rank == 0)
    {
        Matrix_A[0] = 4;
        Matrix_A[1] = 7;
        Matrix_A[2] = 2;
        Matrix_A[3] = 3;
        Matrix_A[4] = 6;
        Matrix_A[5] = 1;
        Matrix_A[6] = 2;
        Matrix_A[7] = 5;
        Matrix_A[8] = 3;

        // 输出原矩阵
        printf("Original Matrix:\n");
        Matrix_print(Matrix_A, N);
    }

    // 广播矩阵A
    MPI_Bcast(Matrix_A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 矩阵求逆
    invert_matrix(Matrix_A, Matrix_I, N);

    // 收集结果
    MPI_Gather(Matrix_I, N * N / comm_sz, MPI_DOUBLE, Matrix_I, N * N / comm_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 输出逆矩阵
    if (my_rank == 0)
    {
        printf("Inverse Matrix:\n");
        Matrix_print(Matrix_I, N);
    }

    free(Matrix_A);
    free(Matrix_I);

    MPI_Finalize();
    return 0;
}
