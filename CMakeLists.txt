project(testenv)

cmake_minimum_required(VERSION 3.8.2)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wpedantic -pthread -O3")

add_executable(comparison ${PROJECT_SOURCE_DIR}/comparison.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(comparison PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(
    comparison
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/
    PRIVATE ${PROJECT_SOURCE_DIR}/src/
)

target_sources(
    comparison
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/CommitOriginalDataStructure.hpp
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/ThreadedMatrixVecMultiplication.hpp

    PRIVATE ${PROJECT_SOURCE_DIR}/src/CommitOriginalDataStructure.cpp
    PRIVATE ${PROJECT_SOURCE_DIR}/src/ThreadedMatrixVecMultiplication.cpp
)