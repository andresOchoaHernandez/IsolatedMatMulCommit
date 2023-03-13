project(SparseDataStructures)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)

cmake_minimum_required(VERSION 3.8.2)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic")

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

find_package(OpenMP)

if(CUDA_FOUND AND CUDA_HAVE_GPU AND OpenMP_CXX_FOUND)
    
    #==================== COMPILE THE LIBRARY ALONE ====================
    add_library(LinearAlgebra SHARED)    

    target_include_directories(
        LinearAlgebra 
        
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/
        
        PRIVATE ${PROJECT_SOURCE_DIR}/src/
        PRIVATE ${PROJECT_SOURCE_DIR}/src/utils
    )
    
    target_sources(
        LinearAlgebra
        PUBLIC  ${PROJECT_SOURCE_DIR}/include/LinearAlgebra.hpp
        
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Vector.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/Matrix.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/CSRMatrix.cu
        PRIVATE ${PROJECT_SOURCE_DIR}/src/CSCMatrix.cu
        
        PRIVATE ${PROJECT_SOURCE_DIR}/src/utils.cpp
    )

    target_link_libraries(LinearAlgebra PRIVATE OpenMP::OpenMP_CXX ${CUDA_cusparse_LIBRARY})
    #===================================================================

    #============================= TESTING =============================
    enable_testing()

    add_executable(VectorUnitTest ${PROJECT_SOURCE_DIR}/test/VectorUnitTest.cpp)

    target_sources(
        VectorUnitTest
        PRIVATE  ${PROJECT_SOURCE_DIR}/test/MeasureTime.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/test/MeasureTime.cpp
    )
    
    target_link_libraries(VectorUnitTest LinearAlgebra)

    add_test(NAME vectorTest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/ COMMAND ./VectorUnitTest)


    add_executable(MatrixUnitTest ${PROJECT_SOURCE_DIR}/test/MatrixUnitTest.cpp)

    target_sources(
        MatrixUnitTest
        PRIVATE  ${PROJECT_SOURCE_DIR}/test/MeasureTime.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/test/MeasureTime.cpp
    )
    
    target_link_libraries(MatrixUnitTest LinearAlgebra)

    add_test(NAME matrixTest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/ COMMAND ./MatrixUnitTest)


    add_executable(CSRMatrixUnitTest ${PROJECT_SOURCE_DIR}/test/CSRMatrixUnitTest.cpp)

    target_sources(
        CSRMatrixUnitTest
        PRIVATE  ${PROJECT_SOURCE_DIR}/test/MeasureTime.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/test/MeasureTime.cpp
    )
    
    target_link_libraries(CSRMatrixUnitTest LinearAlgebra)

    add_test(NAME csrMatrixTest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/ COMMAND ./CSRMatrixUnitTest)

    add_executable(CSCMatrixUnitTest ${PROJECT_SOURCE_DIR}/test/CSCMatrixUnitTest.cpp)

    target_sources(
        CSCMatrixUnitTest
        PRIVATE  ${PROJECT_SOURCE_DIR}/test/MeasureTime.hpp
        PRIVATE ${PROJECT_SOURCE_DIR}/test/MeasureTime.cpp
    )
    
    target_link_libraries(CSCMatrixUnitTest LinearAlgebra)

    add_test(NAME cscMatrixTest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/ COMMAND ./CSCMatrixUnitTest)
    #===================================================================

endif()