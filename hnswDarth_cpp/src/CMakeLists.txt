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

# Locate LightGBM shared library (installed via pip into the active Python env)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import lightgbm, os; print(os.path.join(os.path.dirname(lightgbm.__file__), 'lib'))"
  OUTPUT_VARIABLE LGBM_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT LGBM_LIB_DIR)
  # Fallback: ask the build Python
  execute_process(
    COMMAND python3 -c
      "import lightgbm, os; print(os.path.join(os.path.dirname(lightgbm.__file__), 'lib'))"
    OUTPUT_VARIABLE LGBM_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

# lib_lightgbm.dylib uses an underscore prefix so find_library name is "_lightgbm"
find_library(LGBM_LIB
  NAMES _lightgbm
  HINTS "${LGBM_LIB_DIR}"
)
if(NOT LGBM_LIB)
  # Fallback: construct the path directly from the dir we already have
  set(LGBM_LIB "${LGBM_LIB_DIR}/lib_lightgbm.dylib")
  if(NOT EXISTS "${LGBM_LIB}")
    message(FATAL_ERROR "Cannot locate LightGBM library in ${LGBM_LIB_DIR}")
  endif()
endif()
message(STATUS "LightGBM library: ${LGBM_LIB}")

pybind11_add_module(hnswDarth_cpp
  bindings.cpp
  hnswDarth.cpp
)

target_include_directories(hnswDarth_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(hnswDarth_cpp PRIVATE ${LGBM_LIB})