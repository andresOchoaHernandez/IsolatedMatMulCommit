#include <cassert>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "MeasureTime.hpp"

void test_csrMatrixWithEmptyRows2()
{
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::Vector;

    CSRMatrix a{3,3,4};
    unsigned* rowsVec = a.getRowsArray();
    unsigned* colsVec = a.getColsArray();
    float*   valsVec = a.getValsArray();

    rowsVec[0] = 0; rowsVec[1] = 2; rowsVec[2] = 4; rowsVec[3] = 4;
    colsVec[0] = 0; colsVec[1] = 2; colsVec[2] = 0; colsVec[3] = 2;
    valsVec[0] = 1.0; valsVec[1] = 2.0; valsVec[2] = 1.0; valsVec[3] = 2.0;

    Vector x{3};
    x[0] = 1; x[1] = 2; x[2] = 3;

    Vector r1 = a.matrixVectorMult(x);

    CSRMatrix b{2,3,4};
    unsigned* r = b.getRowsArray();
    unsigned* c = b.getColsArray();
    float*   v = b.getValsArray();

    r[0] = 0; r[1] = 2; r[2] = 4;
    c[0] = 0; c[1] = 2; c[2] = 0; c[3] = 2;
    v[0] = 1.0; v[1] = 2.0; v[2] = 1.0; v[3] = 2.0;

    Vector r2 = b.matrixVectorMult(x);
}

void test_csrMatrixWithEmptyRows()
{
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::Vector;

    CSRMatrix a{3,3,4};
    unsigned* rowsVec = a.getRowsArray();
    unsigned* colsVec = a.getColsArray();
    float*   valsVec = a.getValsArray();

    rowsVec[0] = 0; rowsVec[1] = 1; rowsVec[2] = 1; rowsVec[3] = 4;
    colsVec[0] = 2; colsVec[1] = 0; colsVec[2] = 1; colsVec[3] = 2;
    valsVec[0] = 1.0; valsVec[1] = 3.0; valsVec[2] = 1.0; valsVec[3] = 4.0;

    Vector x{3};
    x[0] = 1; x[1] = 2; x[2] = 3;

    Vector r1 = a.matrixVectorMult(x);

    CSRMatrix b{2,3,4};
    unsigned* r = b.getRowsArray();
    unsigned* c = b.getColsArray();
    float*   v = b.getValsArray();

    r[0] = 0; r[1] = 1; r[2] = 4;
    c[0] = 2; c[1] = 0; c[2] = 1; c[3] = 2;
    v[0] = 1.0; v[1] = 3.0; v[2] = 1.0; v[3] = 4.0;

    Vector r2 = b.matrixVectorMult(x);
}

void test_another_test()
{
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::CSCMatrix;
    using LinearAlgebra::Matrix;

    Matrix a{3,5};
    a.randomInit(0,1);

    CSCMatrix b = a.toCSCMatrix();

    CSRMatrix f = b.toCSR();
}

void test_CSCToCSR()
{
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::CSCMatrix;
    using LinearAlgebra::Matrix;

    Matrix a{3u,5u};

    a.valInit(0);

    a[2] = 1; a[8] = 3;
    a[3] = 2; a[9] = 4;
    a[4] = 3; a[10] = 2;
    a[5] = 1; a[11] = 1;
    a[7] = 5; a[13] = 1;

    CSCMatrix b = a.toCSCMatrix();
    CSRMatrix res = b.toCSR();
}

void test_matrixVectorMultSpeedUp()
{
    using LinearAlgebra::Vector;
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::Matrix;
    using MeasureTime::Timer;

    Timer t;

    const unsigned rows    = 7000;
    const unsigned columns = 50000;

    Matrix a{rows,columns};
    a.randomInit(0,1);

    Vector x{columns};
    x.randomInit(100,200);

    t.begin();
    Vector r1 = a.matrixVectorMult(x);
    t.end("[Matrix]Sequential matrix vector multiplication");

    CSRMatrix f = a.toCSRMatrix();

    t.begin();
    Vector r2 = f.matrixVectorMult(x);
    t.end("[CSRMatrix]Sequential matrix vector multiplication");

    t.begin();
    Vector r3 = f.gpu_matrixVectorMult(x);
    t.end("[CSRMatrix]GPU matrix vector multiplication");

    t.begin();
    Vector r4 = f.gpu_cuSparse_matrixVectorMult(x);
    t.end("[CSRMatrix]CuSparse GPU matrix vector multiplication");

    assert(r1 == r2);
    assert(r1 == r3);
    assert(r1 == r4);
}

void test_matrixVectorMult()
{
    /*
    | 0 0 1 2 3 |
    | 1 0 5 3 4 |
    | 2 1 0 1 0 |

    rows | 0 3 7 10 
    cols | 2 3 4 0 2 3 4 0 1 3 
    vals | 1 2 3 1 5 3 4 2 1 1 
    */
    using LinearAlgebra::Vector;
    using LinearAlgebra::CSRMatrix;
    using LinearAlgebra::Matrix;

    Matrix a{3u,5u};

    a.valInit(0);

    a[2] = 1; a[8] = 3;
    a[3] = 2; a[9] = 4;
    a[4] = 3; a[10] = 2;
    a[5] = 1; a[11] = 1;
    a[7] = 5; a[13] = 1;

    CSRMatrix b = a.toCSRMatrix();

    Vector x{5u};
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    x[3] = 4;
    x[4] = 5;

    Vector r = b.matrixVectorMult(x);

    Vector r1 = b.gpu_matrixVectorMult(x);

    Vector r2 = b.gpu_cuSparse_matrixVectorMult(x);
    
    assert(r == r1);
    assert(r == r2);
    assert(r1 == r2);
}

int main()
{
    test_matrixVectorMult();
    test_matrixVectorMultSpeedUp();
    //test_CSCToCSR();
    //test_another_test();
    //test_csrMatrixWithEmptyRows();
    //test_csrMatrixWithEmptyRows2();
    
    return 0;
}