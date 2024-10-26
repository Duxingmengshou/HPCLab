#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define M 10 // 矩阵行数
#define N 11 // 矩阵列数

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

    double *Matrix_A, *Vector_B, *result, *buffer_A, *ans;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    line = M / comm_sz; // 每个进程分多少行数据
    Matrix_A = (double *)malloc(M * N * sizeof(double));
    Vector_B = (double *)malloc(N * sizeof(double));
    result = (double *)malloc(M * sizeof(double));
    buffer_A = (double *)malloc(line * N * sizeof(double)); // A的均分行的数据
    ans = (double *)malloc(line * sizeof(double));          // 保存部分数据计算结果

    // 给矩阵A和向量B赋值
    if (my_rank == 0)
    {
        start = MPI_Wtime();
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
                Matrix_A[i * N + j] = i + 1; // 示例初始化
        }
        for (int i = 0; i < N; i++)
        {
            Vector_B[i] = i + 1; // 示例初始化
        }

        // 输出矩阵A和向量B
        Matrix_print(Matrix_A, M, N);
        printf("\n");
        for (int i = 0; i < N; i++)
        {
            printf("%.1f ", Vector_B[i]);
        }
        printf("\n");
    }

    // 数据分发
    MPI_Scatter(Matrix_A, line * N, MPI_DOUBLE, buffer_A, line * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // 数据广播
    MPI_Bcast(Vector_B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算结果
    for (int i = 0; i < line; i++)
    {
        double temp = 0;
        for (int j = 0; j < N; j++)
        {
            temp += buffer_A[i * N + j] * Vector_B[j];
        }
        ans[i] = temp;
    }

    // 结果聚集
    MPI_Gather(ans, line, MPI_DOUBLE, result, line, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算剩余行数据
    if (my_rank == 0)
    {
        int rest = M % comm_sz;
        if (rest != 0)
        {
            for (int i = M - rest; i < M; i++)
            {
                double temp = 0;
                for (int j = 0; j < N; j++)
                {
                    temp += Matrix_A[i * N + j] * Vector_B[j];
                }
                result[i] = temp;
            }
        }

        // 输出结果
        printf("\n");
        for (int i = 0; i < M; i++)
        {
            printf("%.1f ", result[i]);
        }
        printf("\n");
        stop = MPI_Wtime();
        printf("rank:%d time:%lfs\n", my_rank, stop - start);
    }

    free(Matrix_A);
    free(Vector_B);
    free(result);
    free(ans);
    free(buffer_A);

    MPI_Finalize();
    return 0;
}
