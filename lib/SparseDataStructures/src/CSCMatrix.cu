#include <vector>
#include <bits/stdc++.h>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "CSCMatrixKernels.cu"

namespace LinearAlgebra
{
    CSCMatrix::CSCMatrix(unsigned nRows,unsigned nCols, unsigned nNzElems):
    _nRows{nRows},
    _nCols{nCols},
    _nNzElems{nNzElems},
    _cols{new unsigned[_nCols + 1u]},
    _rows{new unsigned[_nNzElems]},
    _vals{new float[_nNzElems]}
    {
        if((nNzElems < _nCols) || ((_nRows * _nCols) < _nNzElems)) throw std::runtime_error{"Non Zero elements must be at least equal to Cols but less than rows x cols"};
    }
    CSCMatrix::CSCMatrix(const CSCMatrix& matrix):
    _nRows{matrix._nRows},
    _nCols{matrix._nCols},
    _nNzElems{matrix._nNzElems},
    _cols{new unsigned[_nCols + 1u]},
    _rows{new unsigned[_nNzElems]},
    _vals{new float[_nNzElems]}
    {
        for(unsigned i = 0u ; i <= _nCols ; i++)
        {
            _cols[i] = matrix._cols[i];
        }

        for(unsigned i = 0u ; i < _nNzElems ; i++)
        {
            _rows[i] = matrix._rows[i];
            _vals[i] = matrix._vals[i];
        }
    }
    CSCMatrix::CSCMatrix(CSCMatrix&& mat)
    {
        _nRows    = mat._nRows;
        _nCols    = mat._nCols;
        _nNzElems = mat._nNzElems; 

        _cols = mat._cols;
        _rows = mat._rows;
        _vals = mat._vals;

        mat._nRows    = 0u;
        mat._nCols    = 0u;
        mat._nNzElems = 0u;

        mat._cols = nullptr;
        mat._rows = nullptr; 
        mat._vals = nullptr;
    }
    CSCMatrix::~CSCMatrix(){delete[] _cols;delete[] _rows;delete[] _vals;}

    void CSCMatrix::randomInit(float a,float b)
    {
        // TODO:
    }

    Vector CSCMatrix::matrixVectorMult(const Vector& v1)const
    {
        if(_nCols != v1.len()) throw std::runtime_error("Matrix and Vector's dimensions don't match!");
        
        Vector result{_nRows};

        result.valInit(0.0);

        for(unsigned i = 0u ; i < _nCols ; i++)
        {
            unsigned startColumn = _cols[i];
            unsigned endColumn   = _cols[i + 1u];

            for(unsigned j = startColumn ; j < endColumn ; j++)
            {
                result[_rows[j]] += _vals[j] * v1[i];
            }
        }

        return result;
    }

    Vector CSCMatrix::gpu_matrixVectorMult(const Vector& v1)const
    {
        // TODO:
        return {1};
    }

    Matrix CSCMatrix::toMatrix() const 
    {
        // TODO:
        return {1,1};
    }

    CSRMatrix CSCMatrix::toCSR() const
    {
        struct triplet
        {
            unsigned row;
            unsigned column;
            float   value;

            bool operator<(const triplet& other) const
            {
                return row == other.row ? column < other.column : row < other.row;
            }
        };
        
        CSRMatrix result{_nRows,_nCols,_nNzElems};

        std::vector<triplet> coordinates;

        unsigned columnIndex = 0u;

        for(unsigned i = 0u; i < _nCols ; i++)
        {
            unsigned startCol = _cols[i];
            unsigned endCol = _cols[i+1u];

            for(unsigned j = startCol ; j < endCol ; j++)
            {
                triplet elem;
                elem.row    = _rows[j];
                elem.column = columnIndex;
                elem.value  = _vals[j];
            
                coordinates.push_back(elem);
            }

            columnIndex++;
        }

        std::sort(coordinates.begin(),coordinates.end());

        unsigned* rowsVec = result.getRowsArray();
        unsigned* colsVec = result.getColsArray();
        float*   valsVec = result.getValsArray();

        unsigned rowIndex = 0u;
        rowsVec[rowIndex] = 0u;

        unsigned currentRow = 0u;

        for(unsigned element = 0u ; element < _nNzElems ; element++)
        {
            if(currentRow != coordinates[element].row)
            {
                rowsVec[++rowIndex] = element;
                currentRow = coordinates[element].row;
            }

            colsVec[element] = coordinates[element].column;
            valsVec[element] = coordinates[element].value;
        }

        rowsVec[_nRows] = _nNzElems;

        return result;
    }

    unsigned  CSCMatrix::rows()const
    {
        return _nRows;
    }
    unsigned  CSCMatrix::cols()const
    {
        return _nCols;
    }
    unsigned  CSCMatrix::nonZeroElements()const
    {
        return _nNzElems;
    }
    unsigned* CSCMatrix::getColsArray()
    {
        return _cols;
    }
    unsigned* CSCMatrix::getRowsArray()
    {
        return _rows;
    }
    float*   CSCMatrix::getValsArray()
    {
        return _vals;
    }

    bool CSCMatrix::operator==(const CSCMatrix& other) const
    {
        //TODO:
        return false;
    }

    std::ostream& operator<<(std::ostream& stream, const CSCMatrix& operand)
    {
        stream << "cols | ";

        for(unsigned i = 0u; i <= operand._nCols; i++)
        {
            stream << operand._cols[i] << " ";
        }

        stream << std::endl << "rows | ";

        for(unsigned i = 0u; i < operand._nNzElems; i++)
        {
            stream << operand._rows[i] << " ";        
        }

        stream << std::endl << "vals | ";

        for(unsigned i = 0u; i < operand._nNzElems; i++)
        {
            stream << operand._vals[i] << " ";        
        }

        stream << std::endl;

        return stream;
    }
}