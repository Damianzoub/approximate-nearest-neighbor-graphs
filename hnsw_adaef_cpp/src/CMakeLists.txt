cmake_minimum_required(VERSION 3.18)
project(hnsw_edaef LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(hnsw_edaef
  bindings.cpp
  hnsw_adaef.cpp
)

target_include_directories(hnsw_edaef PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)