cmake_minimum_required(VERSION 3.26)
project(openmp_check)

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)

target_compile_options(${PROJECT_NAME}
        PRIVATE
        /openmp
)

target_link_libraries(${PROJECT_NAME} ${pthreads_ROOT_LIBRARIES})