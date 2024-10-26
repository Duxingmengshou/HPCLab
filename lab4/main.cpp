#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>

using namespace cv;
using namespace std;

// 计算旋转150度的仿射变换矩阵
Mat computeRotationMatrix150(Point2f center) {
    double angle = 150.0; // 旋转角度
    double radians = angle * CV_PI / 180.0; // 转换为弧度

    // 计算旋转矩阵
    Mat rotationMatrix = (Mat_<double>(2, 3) <<
                                             cos(radians), -sin(radians), center.x - center.x * cos(radians) +
                                                                          center.y * sin(radians),
            sin(radians), cos(radians), center.y - center.x * sin(radians) - center.y * cos(radians));

    return rotationMatrix;
}

// 双线性插值
Vec3b bilinearInterpolation(const Mat &src, float x, float y) {
    int x1 = floor(x);
    int y1 = floor(y);
    int x2 = ceil(x);
    int y2 = ceil(y);

    if (x1 < 0 || x2 >= src.cols || y1 < 0 || y2 >= src.rows) {
        return Vec3b(0, 0, 0); // 边界处理
    }

    float a = x - x1;
    float b = y - y1;

    Vec3b I11 = src.at<Vec3b>(y1, x1);
    Vec3b I12 = src.at<Vec3b>(y1, x2);
    Vec3b I21 = src.at<Vec3b>(y2, x1);
    Vec3b I22 = src.at<Vec3b>(y2, x2);

    return (1 - a) * (1 - b) * I11 + a * (1 - b) * I12 + (1 - a) * b * I21 + a * b * I22;
}

// 图像变换
void applyAffineTransform(const Mat &src, Mat &dst, const Mat &affine) {
#pragma omp parallel for
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            // 应用仿射变换公式
            float x_src = affine.at<double>(0, 0) * x + affine.at<double>(0, 1) * y + affine.at<double>(0, 2);
            float y_src = affine.at<double>(1, 0) * x + affine.at<double>(1, 1) * y + affine.at<double>(1, 2);
            // 使用双线性插值获取目标像素的颜色
            dst.at<Vec3b>(y, x) = bilinearInterpolation(src, x_src, y_src);
        }
    }
}

int main() {
    // 读取图像
    Mat src = imread("E:\\_project\\CLion_project\\openmp-check\\input.jpg");
    if (src.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }

    // 计算旋转 150 度的仿射变换矩阵
    Point2f center(src.cols / 2.0f, src.rows / 2.0f); // 旋转中心为图像中心
    Mat affine = computeRotationMatrix150(center);

    // 创建输出图像
    Mat dst(src.rows, src.cols, src.type(), Scalar(0, 0, 0)); // 初始化为黑色背景

    // 应用仿射变换
    applyAffineTransform(src, dst, affine);

    // 显示结果
    namedWindow("Source Image", 0);
    namedWindow("Transformed Image", 0);
    imshow("Source Image", src);
    imshow("Transformed Image", dst);
    waitKey(0);
    return 0;
}

