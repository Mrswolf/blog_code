
cmake_minimum_required(VERSION 3.13)
project(HelloWorld VERSION 1.0 LANGUAGES C CXX)

add_subdirectory(dynlib)
add_subdirectory(stalib)

set(CMAKE_CXX_STANDARD 11)
file(GLOB SOURCES "*.cpp" "utils/*.cpp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
add_executable(helloworld ${SOURCES})
target_include_directories(helloworld PRIVATE includes)
target_link_directories(helloworld PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(helloworld PRIVATE stalib dynlib)
