#include <random>
#include <chrono>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "MatrixKernels.cu"

namespace LinearAlgebra
{
    Matrix::Matrix(unsigned rows, unsigned cols):_rows{rows},_cols{cols},_data{new float[_rows*_cols]}{}
    Matrix::Matrix(const Matrix& matrix):_rows{matrix._rows},_cols{matrix._cols},_data{new float[_rows*_cols]}
    {
        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows; i++)
        {
            for(unsigned j = 0u; j < _cols; j++)
            {
                _data[i*_cols + j] = matrix._data[i*_cols + j]; 
            }
        }
    }
    Matrix::Matrix(Matrix&& mat)
    {
        _rows = mat._rows;
        _cols = mat._cols;
        _data = mat._data;

        mat._rows = 0u;
        mat._cols = 0u;
        mat._data = nullptr;
    }
    Matrix::~Matrix(){delete[]_data;}

    Matrix Matrix::operator+(const Matrix& other)const
    {
        if(_rows != other.rows() || _cols != other.cols()) throw std::runtime_error("Matrix dimensions must be equal!");

        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] + other[i*_cols + j];
            }
        }

        return result;
    }
    Matrix Matrix::operator-(const Matrix& other)const
    {
        if(_rows != other.rows() || _cols != other.cols()) throw std::runtime_error("Matrix dimensions must be equal!");
        
        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] - other[i*_cols + j];
            }
        }

        return result;
    }
    Matrix Matrix::operator*(const Matrix& other)const
    {
        if(_rows != other.rows() || _cols != other.cols()) throw std::runtime_error("Matrix dimensions must be equal!");
        
        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] * other[i*_cols + j];
            }
        }

        return result;
    }
    Matrix Matrix::operator/(const Matrix& other)const
    {
        if(_rows != other.rows() || _cols != other.cols()) throw std::runtime_error("Matrix dimensions must be equal!");
        
        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] / other[i*_cols + j];
            }
        }

        return result;
    }

    Matrix Matrix::operator+(const float constant)const
    {
        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] + constant;
            }
        }

        return result;
    }
    Matrix Matrix::operator-(const float constant)const
    {
        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] - constant;
            }
        }

        return result;
    }
    Matrix Matrix::operator*(const float constant)const
    {
        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] * constant;
            }
        }

        return result;
    }
    Matrix Matrix::operator/(const float constant)const
    {
        Matrix result{_rows,_cols};

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < _cols ; j++)
            {
                result[i*_cols + j] = _data[i*_cols + j] / constant;
            }
        }

        return result;
    }

    Vector Matrix::gpu_matrixVectorMult(const Vector& v1) const
    {
        if(_cols != v1.len()) throw std::runtime_error{"Matrix dimensions and vector dimensions don't match"};

        Vector rv{_rows};

        float* matrix_device; float* v1_device; float* rv_device;

        cudaMalloc(&matrix_device,sizeof(float)*_rows*_cols);
        cudaMalloc(&v1_device,sizeof(float)*v1.len());
        cudaMalloc(&rv_device,sizeof(float)*rv.len());

        cudaMemcpy(matrix_device,_data,sizeof(float)*_rows*_cols,cudaMemcpyHostToDevice);
        cudaMemcpy(v1_device,&v1[0u],sizeof(float)*v1.len(),cudaMemcpyHostToDevice);

        const unsigned threadsPerBlock = 1024u;
        const unsigned numberOfBlocks = _rows < threadsPerBlock? 1u: (_rows % threadsPerBlock == 0u? _rows/threadsPerBlock:_rows/threadsPerBlock +1u);
        dim3 dimGrid(numberOfBlocks,1,1);
        dim3 dimBlock(threadsPerBlock,1,1);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        matrixVectorMultKernel<<<dimGrid,dimBlock>>>(matrix_device,v1_device,rv_device,_rows,_cols);
        cudaDeviceSynchronize();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Cuda kernel for matrix vector multiplication took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

        cudaMemcpy(&rv[0u],rv_device,sizeof(float)*rv.len(),cudaMemcpyDeviceToHost);

        cudaFree(matrix_device);
        cudaFree(v1_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;
    }
    Vector Matrix::matrixVectorMult(const Vector& v1) const
    {
        if(_cols != v1.len()) throw std::runtime_error{"Matrix dimensions and vector dimensions don't match"};

        Vector rv{_rows};

        #pragma omp parallel for
        for( unsigned i = 0u; i < _rows; i++ )
        {
            rv[i] = 0;

            float acc = 0;
            for( unsigned j = 0u; j < _cols; j++ )
            {
                acc += _data[i*_cols + j] * v1[j]; 
            }

            rv[i] = acc;
        }
        
        return rv;
    }

    Matrix Matrix::matrixMultiplication(const Matrix& mat)const
    {
        if(_cols != mat.rows()) throw std::runtime_error{"Matrix dimensions don't match"};

        Matrix result{_rows,mat.cols()};

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows ; i++)
        {
            for(unsigned j = 0u ; j < mat.cols(); j++)
            {
                float acc = 0;
                for(unsigned k = 0u ; k < _cols ; k++)
                {
                    acc += _data[ i*_cols + k] * mat[ mat.cols()*k + j];
                }
                result[i*mat.cols() + j] = acc;
            }
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Sequential matrix multiplication took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;


        return result;
    }

    Matrix Matrix::gpu_matrixMultiplication(const Matrix& mat)const
    {
        if(_cols != mat.rows()) throw std::runtime_error{"Matrix dimensions don't match"};
        //TODO:

        return mat;
    }


    void Matrix::randomInit(float a,float b)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> dist(a,b);

        #pragma omp parallel for
        for(unsigned i = 0u; i < _rows; i++)
        {
            for(unsigned j = 0u; j < _cols; j++)
            {
                _data[i*_cols + j] = dist(rng); 
            }
        }
    }
    
    void Matrix::valInit(float val)
    {
        #pragma omp parallel for
        for (unsigned i = 0u ; i <  _rows ; i++ )
        {
            for(unsigned j = 0u ;  j < _cols ; j++ )
            {
                _data[i*_cols + j] = val;
            }
        }
    }

    CSRMatrix Matrix::toCSRMatrix() const
    {
        unsigned nonZeroElements = 0u;
        bool emptyRow;

        for(unsigned i = 0u ; i < _rows ; i++)
        {
            emptyRow = true;

            for(unsigned j = 0u ; j < _cols ; j++)
            {
                if(_data[i*_cols + j] != 0)
                {
                    nonZeroElements++;
                    emptyRow = false;
                }
            }
            if(emptyRow) throw std::runtime_error{"CSRMatrix doesn't allow empty rows"};
        }

        CSRMatrix csrMatrix{_rows,_cols,nonZeroElements};

        unsigned* rowsVec = csrMatrix.getRowsArray();
        unsigned* colsVec = csrMatrix.getColsArray();
        float*   valsVec = csrMatrix.getValsArray();

        unsigned NzElemsIndex = 0u;

        rowsVec[0u] = 0u;

        for(unsigned i = 0u ; i < csrMatrix.rows() ; i++)
        {
            for(unsigned j = 0u ; j < csrMatrix.cols() ; j++)
            {
                if(_data[i*csrMatrix.cols() + j] != 0)
                {
                    colsVec[NzElemsIndex] = j; 
                    valsVec[NzElemsIndex] = _data[i*csrMatrix.cols() + j];
                    NzElemsIndex++; 
                }
            }

            rowsVec[i + 1u] = NzElemsIndex;
        }

        return csrMatrix;
    }

    CSCMatrix Matrix::toCSCMatrix() const
    {
        unsigned nonZeroElements = 0u;
        bool emptyColumn;

        for(unsigned j = 0u ; j < _cols ; j++)
        {
            emptyColumn = true; 

            for(unsigned i = 0u ; i < _rows ; i++)
            {
                if(_data[i*_cols + j] != 0)
                {
                    nonZeroElements++;
                    emptyColumn = false;
                }
            }

            if(emptyColumn) throw std::runtime_error{"CSCMatrix doesn't allow empty columns"};
        }

        CSCMatrix cscMatrix{_rows,_cols,nonZeroElements};

        unsigned* colsVec = cscMatrix.getColsArray();
        unsigned* rowsVec = cscMatrix.getRowsArray();
        float*   valsVec = cscMatrix.getValsArray();

        unsigned nonZeroElemIndex = 0u;
        colsVec[0u] = 0u;

        for(unsigned j = 0u ; j < _cols ; j++)
        {
            for(unsigned i = 0u ; i < _rows ; i++)
            {
                if(_data[i*cscMatrix.cols() + j] != 0)
                {
                    rowsVec[nonZeroElemIndex] = i;
                    valsVec[nonZeroElemIndex] = _data[i*cscMatrix.cols() + j];
                    nonZeroElemIndex++;
                }
            }

            colsVec[j + 1u] = nonZeroElemIndex;
        }
        
        return cscMatrix;
    }

    unsigned Matrix::rows()const{return _rows;}
    unsigned Matrix::cols()const{return _cols;}
    float*  Matrix::data(){return _data;}

    float& Matrix::operator [](unsigned i){return _data[i];}
    const float& Matrix::operator [](unsigned i)const{return _data[i];}

    bool Matrix::operator==(const Matrix& other) const
    {
        if(_rows != other.rows() || _cols != other.cols()) return false;

        for(unsigned i = 0u ; i < _rows ; i++)
        {
            for(unsigned j = 0u; j < _cols ; j++)
            {
                if(_data[i*_cols + j] != other[i*_cols + j]) return false;
            }
        }

        return true;
    }

    std::ostream& operator<<(std::ostream& stream, const Matrix& operand)
    {
        for(unsigned i = 0u; i < operand._rows ; i++)
        {
            for(unsigned j = 0u; j < operand._cols ; j++)
            {
                stream << operand[i*operand._cols + j] << " ";
            }

            stream << std::endl;
        }

        return stream;
    }
}