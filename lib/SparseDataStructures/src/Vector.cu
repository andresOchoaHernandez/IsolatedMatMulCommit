#include <random>
#include <chrono>
#include <iostream>

#include "LinearAlgebra.hpp"
#include "VectorKernels.cu"

namespace LinearAlgebra
{
    Vector::Vector(unsigned len):_len{len},_vec{new float[_len]}{}
    Vector::Vector(const Vector& vector):_len{vector._len},_vec{new float[_len]}
    {
        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++)
        {
            _vec[i] = vector._vec[i];
        }
    }
    Vector::Vector(Vector&& v)
    {
        _len = v._len;
        _vec = v._vec;
        v._len = 0u;
        v._vec = nullptr;
    }
    Vector::~Vector(){delete[] _vec;}


    Vector Vector::operator+(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};

        Vector result{_len};
        
        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] + other[i];
        }

        return result;
    }

    Vector Vector::operator-(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i]- other[i];
        }

        return result;
    }

    Vector Vector::operator*(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};
        
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i]*other[i];
        }

        return result;
    }

    Vector Vector::operator/(const Vector& other)const
    {
        if( _len != other.len()) throw std::runtime_error{"Vectors dimensions don't match"};
        
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] / other[i];
        }

        return result;
    }

    Vector Vector::operator+(const float constant)const
    {   
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] + constant;
        }

        return result;
    }

    Vector Vector::operator-(const float constant)const
    {
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] - constant;
        }

        return result;
    }

    Vector Vector::operator*(const float constant)const
    {
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] * constant;
        }

        return result;
    }

    Vector Vector::operator/(const float constant)const
    {
        Vector result{_len};

        #pragma omp parallel for
        for(unsigned i = 0u ; i < _len ; i++ )
        {
            result[i] = _vec[i] / constant;
        }

        return result;
    }


    Vector Vector::gpu_diff(const Vector& v2)const
    {
        if( _len != v2.len()) throw std::runtime_error{"Vectors dimensions don't match"};

        Vector rv{_len};

        float *v1_device;float *v2_device;float *rv_device;

        cudaMalloc(&v1_device,sizeof(float)*_len);
        cudaMalloc(&v2_device,sizeof(float)*v2.len());
        cudaMalloc(&rv_device,sizeof(float)*rv.len());

        cudaMemcpy(v1_device,_vec,sizeof(float)*_len,cudaMemcpyHostToDevice);
        cudaMemcpy(v2_device,&v2[0u],sizeof(float)*v2.len(),cudaMemcpyHostToDevice);

        const unsigned threadsPerBlock = 1024u;
        const unsigned numberOfBlocks = _len < threadsPerBlock? 1u: (_len % threadsPerBlock == 0u? _len/threadsPerBlock:_len/threadsPerBlock +1u);
        dim3 dimGrid(numberOfBlocks,1,1);
        dim3 dimBlock(threadsPerBlock,1,1);
        
        vectorDifKernel<<<dimGrid,dimBlock>>>(v1_device,v2_device,rv_device,_len);
        cudaDeviceSynchronize();

        cudaMemcpy(&rv[0u],rv_device,sizeof(float)*rv.len(),cudaMemcpyDeviceToHost);

        cudaFree(v1_device);
        cudaFree(v2_device);
        cudaFree(rv_device);

        cudaDeviceReset();

        return rv;
    }

    Vector Vector::gpu_sum(const Vector& v2)const
    {
        //TODO:
        return v2;
    }

    void Vector::randomInit(float a, float b)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> dist(a,b);

        #pragma omp parallel for
        for (unsigned i = 0u ; i < _len ; i++ )
            _vec[i] = dist(rng);
    }

    void Vector::valInit(float val)
    {
        #pragma omp parallel for
        for (unsigned i = 0u ; i < _len ; i++ )
            _vec[i] = val;
    }

    unsigned Vector::len()const{ return _len; }
    float* Vector::getVec(){ return _vec; }

    float& Vector::operator [](unsigned i){return _vec[i];}
    const float& Vector::operator [](unsigned i)const{return _vec[i];}

    bool Vector::operator==(const Vector& other) const
    {
        if(_len != other.len()) return false;

        for(unsigned i = 0u ; i < _len ; i++)
        {
            if(!areFloatNearlyEqual(_vec[i],other[i])) 
            {
                std::cout << "this > " << _vec[i] << " other > " << other[i] << std::endl;
                return false;
            }
        }

        return true;
    }

    std::ostream& operator<<(std::ostream& stream, const Vector& operand)
    {
        for(unsigned i = 0u ; i < operand._len ; i++)
            stream << operand[i] << " ";
    
        stream << std::endl;

        return stream;
    }
}