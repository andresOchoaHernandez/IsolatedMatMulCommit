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

int main()
{
    using LinearAlgebra::Vector;

    const unsigned size = 1000000;

    Vector a{size};

    a.valInit(1.0);

    Vector b{size};

    b.valInit(1.0);

    assert(a == b);

    test_equlityOperator();
    test_randomInit();
    test_valInit();
    test_vectorArithmethic();
    test_constantArithmethic();
    test_gpuArithmethic();
    
    return 0;
}