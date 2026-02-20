cmake_minimum_required(VERSION 3.18)
project(hnsw_darth_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(hnswDarth_cpp
  bindings.cpp
  hnswDarth.cpp

)

target_include_directories(hnswDarth_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)