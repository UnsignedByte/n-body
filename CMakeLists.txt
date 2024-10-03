cmake_minimum_required(VERSION 3.10)
project(n-body LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O3)
include_directories(lib third-party/include)

add_library(glad third-party/src/glad.c)

add_executable(n-body main.cu)
target_link_libraries(n-body PRIVATE glfw GL GLEW glad)

# perf executables
add_executable(reduce-perf perf/reduce.cu)
add_executable(nbody-perf perf/nbody.cu)