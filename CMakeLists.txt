project(IsolatedMatMultCommit)

cmake_minimum_required(VERSION 3.8.2)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -Wextra -Wpedantic -pthread -O3")

SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
SET(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDA)

if(CUDA_FOUND)
    enable_language(CUDA)
    try_run(
    RUN_RESULT_VAR 
    COMPILE_RESULT_VAR
    ${CMAKE_CURRENT_BINARY_DIR}/ 
    ${PROJECT_SOURCE_DIR}/util/CheckIfCudaDeviceIsPresent.cu
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
    if("${COMPILE_RESULT_VAR}" AND ("${RUN_RESULT_VAR}" GREATER 0))
        message("-- Found : ${RUN_RESULT_VAR} cuda capable device(s)")    
        set(CUDA_HAVE_GPU TRUE)
    else()
        message("-- Found cuda packages but there's no cuda capable device")
        set(CUDA_HAVE_GPU FALSE)
    endif()

else()
    message("-- Cuda packages not found")
endif(CUDA_FOUND)

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
    PUBLIC  ${PROJECT_SOURCE_DIR}/include/ThreadedMatrixVecMultiplicationNoLUT.hpp

    PRIVATE ${PROJECT_SOURCE_DIR}/src/CommitOriginalDataStructure.cpp
    PRIVATE ${PROJECT_SOURCE_DIR}/src/ThreadedMatrixVecMultiplicationNoLUT.cpp
    PRIVATE ${PROJECT_SOURCE_DIR}/src/GpuMatrixMultiplication.cu
)