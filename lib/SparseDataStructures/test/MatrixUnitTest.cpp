#include <cassert>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "MeasureTime.hpp"

void test_equlityOperator()
{
    //TODO:
}

void test_randomInit()
{
    //TODO:
}

void test_valInit()
{
    //TODO:
}

void test_vectorArithmethic()
{
    //TODO:
}

void test_constantArithmethic()
{
    //TODO:
}

void test_gpuArithmethic()
{
    //TODO:
}

void test_matrixVectorMultiplication()
{
    //TODO: both host and device methods
}

void test_matrixMultiplication()
{
    //TODO: both host and device methods
}

void test_matrixToCSRConversion()
{
    // TODO:
}

int main()
{
    std::cout << "Matrix test" << std::endl;

    test_equlityOperator();
    test_randomInit();
    test_valInit();
    test_vectorArithmethic();
    test_constantArithmethic();
    test_gpuArithmethic();
    test_matrixVectorMultiplication();
    test_matrixMultiplication();
    test_matrixToCSRConversion();
    
    return 0;
}