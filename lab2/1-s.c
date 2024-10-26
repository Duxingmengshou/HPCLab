#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct
{
    double *x;
    double *y;
    int start;
    int end;
    double *S_x;
    double *S_y;
    double *S_xy;
    double *S_xx;
} ThreadData;

void *calculate_sums(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    double local_S_x = 0.0, local_S_y = 0.0, local_S_xy = 0.0, local_S_xx = 0.0;

    for (int i = data->start; i < data->end; i++)
    {
        local_S_x += data->x[i];
        local_S_y += data->y[i];
        local_S_xy += data->x[i] * data->y[i];
        local_S_xx += data->x[i] * data->x[i];
    }

    // 合并结果
    *data->S_x += local_S_x;
    *data->S_y += local_S_y;
    *data->S_xy += local_S_xy;
    *data->S_xx += local_S_xx;

    return NULL;
}

void linear_fit(double *x, double *y, int n, int num_threads, double *k, double *b)
{
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    double S_x = 0.0, S_y = 0.0, S_xy = 0.0, S_xx = 0.0;

    int chunk_size = n / num_threads;

    // 创建线程
    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].x = x;
        thread_data[i].y = y;
        thread_data[i].start = i * chunk_size;
        thread_data[i].end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
        thread_data[i].S_x = &S_x;
        thread_data[i].S_y = &S_y;
        thread_data[i].S_xy = &S_xy;
        thread_data[i].S_xx = &S_xx;

        pthread_create(&threads[i], NULL, calculate_sums, &thread_data[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // 计算斜率 k 和截距 b
    *k = (n * S_xy - S_x * S_y) / (n * S_xx - S_x * S_x);
    *b = (S_y - (*k) * S_x) / n;
}

int main()
{
    int n = 10; // 数据点数量
    double x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double y[] = {2.2, 2.8, 3.6, 4.5, 5.1, 6.3, 7.8, 8.5, 9.1, 10.2};

    double k, b;
    int num_threads = 4; // 线程数量

    linear_fit(x, y, n, num_threads, &k, &b);

    printf("拟合直线: y = %.2fx + %.2f\n", k, b);
    return 0;
}
