cmake_minimum_required(VERSION 3.13)
project(dynlib VERSION 1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61;70)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

aux_source_directory(. SOURCES)
add_library(dynlib SHARED ${SOURCES})
target_compile_definitions(dynlib PRIVATE EXPORT_DLL)
target_include_directories(dynlib PUBLIC /usr/local/cuda/include)
target_compile_options(dynlib PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_link_directories(dynlib PUBLIC /usr/local/cuda/lib64)
target_link_libraries(dynlib PRIVATE cudart)
