#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define M 10 // 矩阵维度
#define N 11
#define K 12

void Matrix_print(double *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%.1f ", A[i * n + j]);
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    int my_rank, comm_sz, line;
    double start, stop; // 计时时间
    MPI_Status status;

    double *Matrix_A, *Matrix_B, *Matrix_C, *ans, *buffer_A, *buffer_C, *result_Matrix;
    double alpha = 2, beta = 2; // 系数C=aA*B+bC

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    line = M / comm_sz; // 每个进程分多少行数据
    Matrix_A = (double *)malloc(M * N * sizeof(double));
    Matrix_B = (double *)malloc(N * K * sizeof(double));
    Matrix_C = (double *)malloc(M * K * sizeof(double));
    buffer_A = (double *)malloc(line * N * sizeof(double));   // A的均分行的数据
    buffer_C = (double *)malloc(line * K * sizeof(double));   // C的均分行的数据
    ans = (double *)malloc(line * K * sizeof(double));        // 保存部分数据计算结果
    result_Matrix = (double *)malloc(M * K * sizeof(double)); // 保存数据计算结果

    // 给矩阵A B,C赋值
    if (my_rank == 0)
    {
        start = MPI_Wtime();
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
                Matrix_A[i * N + j] = i + 1;
            for (int p = 0; p < K; p++)
                Matrix_C[i * K + p] = 1;
        }
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
                Matrix_B[i * K + j] = j + 1;
        }

        // 输出A,B,C
        Matrix_print(Matrix_A, M, N);
        Matrix_print(Matrix_B, N, K);
    }

    // 数据分发
    MPI_Scatter(Matrix_A, line * N, MPI_DOUBLE, buffer_A, line * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Matrix_C, line * K, MPI_DOUBLE, buffer_C, line * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // 数据广播
    MPI_Bcast(Matrix_B, N * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算 结果
    for (int i = 0; i < line; i++)
    {
        for (int j = 0; j < K; j++)
        {
            double temp = 0;
            for (int p = 0; p < N; p++)
                temp += buffer_A[i * N + p] * Matrix_B[p * K + j];
            ans[i * K + j] = alpha * temp + beta * buffer_C[i * K + j];
        }
    }
    // 结果聚集
    MPI_Gather(ans, line * K, MPI_DOUBLE, result_Matrix, line * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算A剩下的行数据
    if (my_rank == 0)
    {
        int rest = M % comm_sz;
        if (rest != 0)
        {
            for (int i = M - rest - 1; i < M; i++)
                for (int j = 0; j < K; j++)
                {
                    double temp = 0;
                    for (int p = 0; p < N; p++)
                        temp += Matrix_A[i * N + p] * Matrix_B[p * K + j];
                    result_Matrix[i * K + j] = alpha * temp + beta * Matrix_C[i * K + j];
                }
        }

        Matrix_print(result_Matrix, M, K);
        stop = MPI_Wtime();

        printf("rank:%d time:%lfs\n", my_rank, stop - start);
    }

    free(Matrix_A);
    free(Matrix_B);
    free(Matrix_C);
    free(ans);
    free(buffer_A);
    free(buffer_C);
    free(result_Matrix);

    MPI_Finalize();
    return 0;
}