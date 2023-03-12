#pragma once

#include <ostream>

namespace LinearAlgebra{

    class Vector; class Matrix; class CSRMatrix;class CSCMatrix;

    class Vector
    {
        unsigned _len;
        
        float*  _vec;

        public:
            Vector(unsigned len);
            Vector(const Vector& vector);
            Vector(Vector&& v);
            ~Vector();

            void randomInit(float a,float b);
            void valInit(float val);

            Vector operator+(const Vector& other)const;
            Vector operator-(const Vector& other)const;
            Vector operator*(const Vector& other)const;
            Vector operator/(const Vector& other)const;

            Vector operator+(const float constant)const;
            Vector operator-(const float constant)const;
            Vector operator*(const float constant)const;
            Vector operator/(const float constant)const;

            Vector gpu_diff(const Vector& v2)const;
            Vector gpu_sum(const Vector& v2)const;

            unsigned len()const;
            float*  getVec();

            float& operator [](unsigned i);
            const float& operator [](unsigned i)const;
            bool operator==(const Vector& other) const;

            friend std::ostream& operator<<(std::ostream& stream, const Vector& operand);
    };

    class Matrix
    {
        unsigned _rows;
        unsigned _cols;

        float*  _data;

        public:
            Matrix(unsigned rows, unsigned cols);
            Matrix(const Matrix& matrix);
            Matrix(Matrix&& mat);
            ~Matrix();

            void randomInit(float a,float b);
            void valInit(float val);

            Matrix operator+(const Matrix& other)const;
            Matrix operator-(const Matrix& other)const;
            Matrix operator*(const Matrix& other)const;
            Matrix operator/(const Matrix& other)const;

            Matrix operator+(const float constant)const;
            Matrix operator-(const float constant)const;
            Matrix operator*(const float constant)const;
            Matrix operator/(const float constant)const;

            Vector matrixVectorMult(const Vector& v1)const;
            Matrix matrixMultiplication(const Matrix& mat)const;

            Vector gpu_matrixVectorMult(const Vector& v1)const;
            Matrix gpu_matrixMultiplication(const Matrix& mat)const;

            CSRMatrix toCSRMatrix() const;
            CSCMatrix toCSCMatrix() const;

            unsigned rows()const;
            unsigned cols()const;
            float*  data();

            float& operator [](unsigned i);
            const float& operator [](unsigned i)const;
            bool operator==(const Matrix& other) const;

            friend std::ostream& operator<<(std::ostream& stream, const Matrix& operand);
    };

    class CSRMatrix
    {
        unsigned  _nRows;
        unsigned  _nCols;
        unsigned  _nNzElems;

        unsigned *_rows;
        unsigned *_cols;
        float   *_vals;

        public:
            CSRMatrix(unsigned nRows,unsigned nCols, unsigned nNzElems);
            CSRMatrix(const CSRMatrix& matrix);
            CSRMatrix(CSRMatrix&& mat);
            ~CSRMatrix();

            void randomInit(float a,float b);

            Vector matrixVectorMult(const Vector& v1)const;

            Vector gpu_matrixVectorMult(const Vector& v1)const;
            Vector gpu_cuSparse_matrixVectorMult(const Vector& v1)const;

            Matrix toMatrix() const;
            CSCMatrix toCSC() const;

            unsigned  rows()const;
            unsigned  cols()const;
            unsigned  nonZeroElements()const;
            unsigned* getRowsArray();
            unsigned* getColsArray();
            float*   getValsArray();

            bool operator==(const CSRMatrix& other) const;

            friend std::ostream& operator<<(std::ostream& stream, const CSRMatrix& operand);
    };

    class CSCMatrix
    {
        unsigned  _nRows;
        unsigned  _nCols;
        unsigned  _nNzElems;

        unsigned *_cols;
        unsigned *_rows;
        float   *_vals;

        public:
            CSCMatrix(unsigned nRows,unsigned nCols, unsigned nNzElems);
            CSCMatrix(const CSCMatrix& matrix);
            CSCMatrix(CSCMatrix&& mat);
            ~CSCMatrix();

            void randomInit(float a,float b);

            Vector matrixVectorMult(const Vector& v1)const;

            Vector gpu_matrixVectorMult(const Vector& v1)const;

            Matrix toMatrix() const;
            CSRMatrix toCSR() const;

            unsigned  rows()const;
            unsigned  cols()const;
            unsigned  nonZeroElements()const;
            unsigned* getColsArray();
            unsigned* getRowsArray();
            float*   getValsArray();

            bool operator==(const CSCMatrix& other) const;

            friend std::ostream& operator<<(std::ostream& stream, const CSCMatrix& operand);
    };

    bool areFloatNearlyEqual(float a, float b);
}