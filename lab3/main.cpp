#include <iostream>
#include <vector>
#include <omp.h>

struct Point {
    double x;
    double y;
};

void fitAffineTransform(const std::vector<Point>& original, const std::vector<Point>& transformed, double& a, double& b, double& c, double& d, double& tx, double& ty) {
    int n = original.size();

    // 确保原始点和变换点数量一致
    if (n != transformed.size() || n < 3) {
        std::cerr << "Point sets must be of the same size and at least 3 points are required." << std::endl;
        return;
    }

    double sum_x = 0, sum_y = 0, sum_x_prime = 0, sum_y_prime = 0;
    double sum_xy = 0, sum_x2 = 0, sum_y2 = 0;

//     并行计算各个求和项
#pragma omp parallel for reduction(+:sum_x, sum_y, sum_x_prime, sum_y_prime, sum_xy, sum_x2, sum_y2)
    for (int i = 0; i < n; ++i) {
        double x = original[i].x;
        double y = original[i].y;
        double x_prime = transformed[i].x;
        double y_prime = transformed[i].y;

        sum_x += x;
        sum_y += y;
        sum_x_prime += x_prime;
        sum_y_prime += y_prime;
        sum_xy += x * y_prime;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

//     计算均值
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double mean_x_prime = sum_x_prime / n;
    double mean_y_prime = sum_y_prime / n;

//     计算变换参数
    a = (sum_xy - n * mean_x * mean_y_prime) / (sum_x2 - n * mean_x * mean_x);
    b = (sum_y_prime - mean_y_prime - a * mean_x) / mean_y;
    c = (sum_y_prime - mean_y_prime - a * mean_x) / mean_x;
    d = (sum_y - mean_y) / mean_y;

//     计算平移量
    tx = mean_x_prime - a * mean_x - b * mean_y;
    ty = mean_y_prime - c * mean_x - d * mean_y;
}

int main() {
    std::vector<Point> original = {{0, 0}, {1, 1}, {2, 2}};
    std::vector<Point> transformed = {{1, 2}, {2, 3}, {3, 6}};

    double a, b, c, d, tx, ty;
    fitAffineTransform(original, transformed, a, b, c, d, tx, ty);

    std::cout << "Affine Transform Parameters:" << std::endl;
    std::cout << "a: " << a << ", b: " << b << ", c: " << c << ", d: " << d << ", tx: " << tx << ", ty: " << ty << std::endl;

    return 0;
}