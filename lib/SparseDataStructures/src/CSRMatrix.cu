#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cusparse.h>

#include "LinearAlgebra.hpp"
#include "CSRMatrixKernels.cu"

namespace LinearAlgebra
{
    CSRMatrix::CSRMatrix(unsigned nRows,unsigned nCols, unsigned nNzElems):
    _nRows{nRows},
    _nCols{nCols},
    _nNzElems{nNzElems},
    _rows{new unsigned[_nRows + 1u]},
    _cols{new unsigned[_nNzElems]},
    _vals{new float[_nNzElems]}
    {
        if((nNzElems < _nRows) || ((_nRows * _nCols) < _nNzElems)) throw std::runtime_error{"Non Zero elements must be at least equal to Rows but less than rows x cols"};
    }

    CSRMatrix::CSRMatrix(const CSRMatrix& matrix):
    _nRows{matrix._nRows},
    _nCols{matrix._nCols},
    _nNzElems{matrix._nNzElems},
    _rows{new unsigned[_nRows + 1u]},
    _cols{new unsigned[_nNzElems]},
    _vals{new float[_nNzElems]}
    {
        for(unsigned i = 0u ; i <= _nRows ; i++)
        {
            _rows[i] = matrix._rows[i];
        }

        for(unsigned i = 0u; i < _nNzElems; i++)
        {
            _cols[i] = matrix._cols[i];
            _vals[i] = matrix._vals[i];
        }
    
    }
    CSRMatrix::CSRMatrix(CSRMatrix&& mat)
    {
        _nRows    = mat._nRows;
        _nCols    = mat._nCols;
        _nNzElems = mat._nNzElems; 

        _rows = mat._rows;
        _cols = mat._cols;
        _vals = mat._vals;

        mat._nRows    = 0u;
        mat._nCols    = 0u;
        mat._nNzElems = 0u;

        mat._rows = nullptr; 
        mat._cols = nullptr;
        mat._vals = nullptr;
    }
    CSRMatrix::~CSRMatrix(){delete[] _rows;delete[] _cols;delete[] _vals;}

    Vector CSRMatrix::gpu_matrixVectorMult(const Vector& v1) const
    {
        if(_nCols != v1.len()) throw std::runtime_error{"Matrix dimensions and vector dimensions don't match"};

        Vector rv{_nRows};

        unsigned* rows_device;
        unsigned* cols_device;
        float*   vals_device; 
        
        float* v1_device; 
        float* rv_device;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMalloc(&rows_device,sizeof(unsigned)*(_nRows + 1));
        cudaMalloc(&cols_device,sizeof(unsigned)*_nNzElems);
        cudaMalloc(&vals_device,sizeof(float)*_nNzElems);

        cudaMalloc(&v1_device,sizeof(float)*v1.len());
        cudaMalloc(&rv_device,sizeof(float)*rv.len());

        cudaMemcpy(rows_device,_rows,sizeof(unsigned)*(_nRows + 1),cudaMemcpyHostToDevice);
        cudaMemcpy(cols_device,_cols,sizeof(unsigned)*_nNzElems,cudaMemcpyHostToDevice);
        cudaMemcpy(vals_device,_vals,sizeof(float)*_nNzElems,cudaMemcpyHostToDevice);

        cudaMemcpy(v1_device,&v1[0u],sizeof(float)*v1.len(),cudaMemcpyHostToDevice);

        const unsigned threadsPerBlock = 64u;
        const unsigned numberOfBlocks = _nRows < threadsPerBlock? 64u: (_nRows % threadsPerBlock == 0u?_nRows:_nRows+1u);

        dim3 dimGrid(numberOfBlocks,1,1);
        dim3 dimBlock(threadsPerBlock,1,1);

        cudaEventRecord(start);
        csrMatrixVectorMultKernel<<<dimGrid,dimBlock>>>(rows_device,cols_device,vals_device,v1_device,rv_device,_nRows);
        cudaEventRecord(stop);


        cudaMemcpy(&rv[0u],rv_device,sizeof(float)*rv.len(),cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Cuda kernel for csr matrix vector multiplication took : " << milliseconds << " ms" << std::endl;


        cudaFree(rows_device);
        cudaFree(cols_device);
        cudaFree(vals_device);

        cudaFree(v1_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;
    }

    Vector CSRMatrix::gpu_cuSparse_matrixVectorMult(const Vector& v1)const
    {
        if(_nCols != v1.len()) throw std::runtime_error{"Matrix dimensions and vector dimensions don't match"};

        Vector rv{_nRows};

        unsigned* rows_device;
        unsigned* cols_device;
        float*   vals_device; 
        
        float* v1_device; 
        float* rv_device;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMalloc(&rows_device,sizeof(unsigned)*(_nRows + 1));
        cudaMalloc(&cols_device,sizeof(unsigned)*_nNzElems);
        cudaMalloc(&vals_device,sizeof(float)*_nNzElems);

        cudaMalloc(&v1_device,sizeof(float)*v1.len());
        cudaMalloc(&rv_device,sizeof(float)*rv.len());

        cudaMemcpy(rows_device,_rows,sizeof(unsigned)*(_nRows + 1),cudaMemcpyHostToDevice);
        cudaMemcpy(cols_device,_cols,sizeof(unsigned)*_nNzElems,cudaMemcpyHostToDevice);
        cudaMemcpy(vals_device,_vals,sizeof(float)*_nNzElems,cudaMemcpyHostToDevice);

        cudaMemset(rv_device,0.0f,sizeof(float)*rv.len());

        cudaMemcpy(v1_device,&v1[0u],sizeof(float)*v1.len(),cudaMemcpyHostToDevice);

        // =========== CUSPARSE ========== //
        cusparseHandle_t     handle = nullptr;
        cusparseSpMatDescr_t csrMatrixDesc;
        cusparseDnVecDescr_t v1Desc,rvDesc;
        void *buffer = nullptr;
        size_t sizeBuffer = 0;
        float alpha = 1.0f, beta = 0.0f;

        cusparseCreate(&handle);
        cusparseCreateCsr(&csrMatrixDesc,_nRows,_nCols,_nNzElems,rows_device,cols_device,vals_device,CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F);
        cusparseCreateDnVec(&v1Desc,v1.len(),v1_device,CUDA_R_32F);
        cusparseCreateDnVec(&rvDesc,rv.len(),rv_device,CUDA_R_32F);

        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha, csrMatrixDesc, v1Desc, &beta, rvDesc, CUDA_R_32F,CUSPARSE_CSRMV_ALG2, &sizeBuffer);
        cudaMalloc(&buffer, sizeBuffer);

        cudaEventRecord(start);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,csrMatrixDesc,v1Desc, &beta, rvDesc, CUDA_R_32F,CUSPARSE_CSRMV_ALG2,buffer); 
        cudaEventRecord(stop);

        cusparseDestroySpMat(csrMatrixDesc);
        cusparseDestroyDnVec(v1Desc);
        cusparseDestroyDnVec(rvDesc);

        cusparseDestroy(handle);

        // ========== ========= ========== //
        cudaMemcpy(&rv[0u],rv_device,sizeof(float)*rv.len(),cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "CuSparse CSR SpMV took : " << milliseconds << " ms" << std::endl;

        cudaFree(rows_device);
        cudaFree(cols_device);
        cudaFree(vals_device);

        cudaFree(v1_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;      
    }

    Vector CSRMatrix::matrixVectorMult(const Vector& v1) const
    {
        if(_nCols != v1.len()) throw std::runtime_error("Matrix and Vector's dimensions don't match!");

        Vector rv{_nRows};

        unsigned startRow = 0u; 
        unsigned endRow   = 0u;

        #pragma omp parallel for
        for(unsigned i = 0u; i < _nRows ; i++ )
        {
            startRow = _rows[i];
            endRow   = _rows[i + 1u];

            rv[i] = 0;

            float acc = 0;
            for(unsigned j = startRow; j < endRow; j++)
            {
                acc += _vals[j] * v1[_cols[j]];
            }

            rv[i] = acc; 
        }

        return rv;
    }
    void CSRMatrix::randomInit(float a,float b)
    {
        if(a == 0 || b == 0) throw std::runtime_error("a and b must be != 0");

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> vals_dist(a,b);
        std::uniform_int_distribution<std::mt19937::result_type> cols_dist(0,_nCols-1);

        std::vector<char> matrixIndexes(_nRows * _nCols,'0');

        for(unsigned i = 0u; i < _nRows ; i ++)
        {
            matrixIndexes[i*_nCols + cols_dist(rng)] = '1';
        }

        unsigned elemsToDistribute = _nNzElems - _nRows;
        unsigned randomIndex;
        bool rowIsFull = false;

        while(elemsToDistribute > 0u)
        {
            for(unsigned i = 0u ; i < _nRows; i++)
            {
                if(elemsToDistribute == 0u) break;

                randomIndex = cols_dist(rng) ;

                if(matrixIndexes[i*_nCols + randomIndex] == '1')
                {
                    for(unsigned j = 0u ; j < _nCols; j++ )
                    {
                        if(matrixIndexes[i*_nCols + j] == '0')
                        {
                            rowIsFull = false;
                            matrixIndexes[i*_nCols + j] = '1';
                            elemsToDistribute--;
                            break;
                        }
                        rowIsFull = true;
                    }

                    if(rowIsFull) continue;
                    
                }
                else
                {
                    matrixIndexes[i*_nCols + randomIndex] = '1';
                    elemsToDistribute--;
                }
            }
        }

        unsigned NzElemsIndex = 0u;

        _rows[0u] = 0u;

        for(unsigned i = 0u ; i < _nRows ; i++)
        {
            for(unsigned j = 0u ; j < _nCols ; j++)
            {
                if(matrixIndexes[i*_nCols + j] == '1')
                {
                    _cols[NzElemsIndex] = j; 
                    _vals[NzElemsIndex] = vals_dist(rng);
                    NzElemsIndex++; 
                }
            }
            _rows[i + 1u] = NzElemsIndex;
        }
    }

    Matrix CSRMatrix::toMatrix() const
    {
        Matrix result{_nRows,_nCols};
        result.valInit(0);

        unsigned startRow,endRow;

        for(unsigned i = 0u ; i < _nRows; i++)
        {
            startRow = _rows[i];
            endRow   = _rows[i+1u];
            for(unsigned j = startRow; j < endRow ; j++)
            {
                result[i*_nCols + _cols[j]] = _vals[j];
            }
        }

        return result;
    }

    CSCMatrix CSRMatrix::toCSC() const
    {
        //TODO:
        return {1u,1u,1u};
    } 

    unsigned  CSRMatrix::rows()const{return _nRows;}
    unsigned  CSRMatrix::cols()const{return _nCols;}
    unsigned  CSRMatrix::nonZeroElements()const{return _nNzElems;}
    unsigned* CSRMatrix::getRowsArray(){return _rows;}
    unsigned* CSRMatrix::getColsArray(){return _cols;}
    float*   CSRMatrix::getValsArray(){return _vals;}

    bool CSRMatrix::operator==(const CSRMatrix& other) const
    {
        //TODO:
        return false;
    }

    std::ostream& operator<<(std::ostream& stream, const CSRMatrix& operand)
    {
        stream << "rows | ";

        for(unsigned i = 0u; i <= operand._nRows; i++)
        {
            stream << operand._rows[i] << " ";
        }

        stream << std::endl << "cols | ";

        for(unsigned i = 0u; i < operand._nNzElems; i++)
        {
            stream << operand._cols[i] << " ";        
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