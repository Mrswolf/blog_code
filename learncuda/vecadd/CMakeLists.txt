cmake_minimum_required(VERSION 3.17)
project(vecadd VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61;70)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

aux_source_directory(. SOURCES)
add_executable(vecadd ${SOURCES})

target_include_directories(vecadd PUBLIC /usr/local/cuda/include)
target_compile_options(vecadd PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_link_directories(vecadd PUBLIC /usr/local/cuda/lib64)
target_link_libraries(vecadd PRIVATE cudart)