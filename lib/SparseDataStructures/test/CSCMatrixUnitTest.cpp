#include <cassert>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "MeasureTime.hpp"

void test_MatVecMultBigger()
{
    using LinearAlgebra::Matrix;
    using LinearAlgebra::CSCMatrix;
    using LinearAlgebra::Vector;

    const unsigned rows = 2000;
    const unsigned cols = 3000;


    Matrix a{rows,cols};
    a.randomInit(0.0,0.1);

    Vector x{cols};
    x.randomInit(3.0,4.0);
    

    Vector r1 = a.matrixVectorMult(x);

    CSCMatrix b = a.toCSCMatrix();

    Vector r2 = b.matrixVectorMult(x);

    assert(r1 == r1);
}

void test_MatVectorMult()
{
    using LinearAlgebra::Matrix;
    using LinearAlgebra::CSCMatrix;
    using LinearAlgebra::Vector;

    Matrix a{3u,5u};

    a.valInit(0);

    a[2] = 1; a[8] = 3;
    a[3] = 2; a[9] = 4;
    a[4] = 3; a[10] = 2;
    a[5] = 1; a[11] = 1;
    a[7] = 5; a[13] = 1;

    Vector x{5u};
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    x[3] = 4;
    x[4] = 5;

    Vector r = a.matrixVectorMult(x);

    CSCMatrix f = a.toCSCMatrix();

    Vector r1 = f.matrixVectorMult(x);

    assert(r1 == r);
}

void test_CorrectConstruction()
{
/*
    | 0 0 1 2 3 |
    | 1 0 5 3 4 |
    | 2 1 0 1 0 |

    cols | 0 2 3 5 8 10 
    rows | 1 2 2 0 1 0 1 2 0 1 
    vals | 1 2 1 1 5 2 3 1 3 4 
*/

    using LinearAlgebra::Matrix;
    using LinearAlgebra::CSCMatrix;

    Matrix a{3u,5u};

    a.valInit(0);

    a[2] = 1; a[8] = 3;
    a[3] = 2; a[9] = 4;
    a[4] = 3; a[10] = 2;
    a[5] = 1; a[11] = 1;
    a[7] = 5; a[13] = 1;

    CSCMatrix b = a.toCSCMatrix();

    const unsigned cols[6u] = {0u,2u,3u,5u,8u,10u};
    const unsigned rows[10u] = {1u,2u,2u,0u,1u,0u,1u,2u,0u,1u};
    const int vals[10] = {1,2,1,1,5,2,3,1,3,4};

    unsigned* colsVec = b.getColsArray();
    unsigned* rowsVec = b.getRowsArray();
    float*   valsVec = b.getValsArray();

    bool correct = true;

    for(unsigned i = 0u ; i < 6u ; i++)
    {
        if(cols[i] != colsVec[i])
            correct = false;
    }
    assert(correct);

    for(unsigned i = 0u ; i < 10u ; i++)
    {
        if(rows[i] != rowsVec[i])
            correct = false;
    }
    assert(correct);

    for(unsigned i = 0u ; i < 10u ; i++)
    {
        if(vals[i] != valsVec[i])
            correct = false;
    }
    assert(correct);
}

int main()
{
    test_CorrectConstruction();
    test_MatVectorMult();

    test_MatVecMultBigger();

    return 0;
}