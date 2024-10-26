cmake_minimum_required(VERSION 3.26)
project(openmp_check)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
set(OpenCV_DIR "H:/Programming/C++/OpenCV/OpenCV490/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} main.cpp)

target_compile_options(${PROJECT_NAME}
        PRIVATE
        /openmp
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${pthreads_ROOT_LIBRARIES})