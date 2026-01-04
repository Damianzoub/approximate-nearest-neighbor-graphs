cmake_minimum_required(VERSION 3.16)
project(hnsw_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
add_executable(
    hnsw_cpp
    src/main.cpp
    src/hnsw.cpp
    src/vector_utils.cpp
)

target_compile_options(hnsw_cpp PRIVATE -03 -Wall -Wextra -Wpedantic)