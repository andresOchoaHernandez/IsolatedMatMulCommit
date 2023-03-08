#include <iostream>
#include <chrono>
#include <thread>

#include "matrixVecMultiplication.hpp"

/*
#define THREADS_IN_SYS 12

void zeroInit(float* _vOUT, int M)
{
    for(int i = 0; i < M; i++) _vOUT[i] = 0.0;
}
*/

int main()
{
    const int threads = std::thread::hardware_concurrency();

    std::cout << threads << std::endl;
    /*
    int _nS       = 100;
    int _nF       = 283522;
    int _nR       = 1;
    int _nE       = 145434;
    int _nT       = 1;
    int _nV       = 53008;
    int _nI       = 2;
    int _n        = 16664340;
    int _ndirs    = 32761;

    int M = _nV*_nS;                    
    int N = _nR*_nF + _nT*_nE + _nI*_nV;
    
    float *_vIN          = new float[N];
    float *_vOUT         = new float[M]();

    uint32_t *_ICf        = new uint32_t[_n];
    uint32_t *_ICv        = new uint32_t[_n];
    uint16_t *_ICo        = new uint16_t[_n];
    float *_ICl           = new float[_n];

    uint32_t *_ECv        = new uint32_t[_nE]; 
    uint16_t *_ECo        = new uint16_t[_nE];

    uint32_t *_ISOv       = new uint32_t[_nV];

    float *_wmrSFP        = new float[_nR*_ndirs*_nS]; 
    float *_wmhSFP        = new float[_nT*_ndirs*_nS]; 
    float *_isoSFP        = new float[_nI*_nS];

    uint32_t* _ICthreads  = new uint32_t[THREADS_IN_SYS+1];
    uint32_t* _ECthreads  = new uint32_t[THREADS_IN_SYS+1];
    uint32_t* _ISOthreads = new uint32_t[THREADS_IN_SYS+1];

    float *_vOUT_CORRECT = new float[M];

    loadDataset(
        _nF,_n,_nE,_nV,_nS,_ndirs,_nI,_nR,_nT,N,M,
        _vIN,_vOUT_CORRECT,
        _ICf,_ICv,_ICo,_ICl,
        _ECv,_ECo,
        _ISOv,
        _wmrSFP,_wmhSFP,_isoSFP,
        _ICthreads,_ECthreads,_ISOthreads,THREADS_IN_SYS+1
    );

    //============================== SEQUENTIAL ================================================//
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    sequential_matVecMult(
        _nF,_n,_nE,_nV,_nS,_ndirs,_nI,_nR,_nT,N,M,
        _vIN,_vOUT,
        _ICf,_ICv,_ICo,_ICl,
        _ECv,_ECo,
        _ISOv,
        _wmrSFP,_wmhSFP,_isoSFP
    );
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto timeSequential = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Sequential matrix vector multiplication took: " << timeSequential << " ms" << std::endl;
    
    if(!verifyCorrectness(_vOUT,_vOUT_CORRECT,M))
    {
        std::cout << "Error in sequential implementation!" << std::endl;
    }

    // ============================== THREADED ================================================ //
    zeroInit(_vOUT,M);

    begin = std::chrono::steady_clock::now();
    threaded_matVecMult(
        _nF,_n,_nE,_nV,_nS,_ndirs,
        _vIN,_vOUT,
        _ICf,_ICv,_ICo,_ICl,
        _ECv,_ECo,
        _ISOv,
        _wmrSFP,_wmhSFP,_isoSFP,
        _ICthreads,_ECthreads,_ISOthreads
    );
    end = std::chrono::steady_clock::now();
    auto timeThreaded = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "threaded_matVecMult took: " << timeThreaded << " ms" << std::endl;

    if(!verifyCorrectness(_vOUT,_vOUT_CORRECT,M))
    {
        std::cout << "Error in threaded implementation !" << std::endl;
    }

    std::cout << "Speedup Sequential/Threaded : " << timeSequential / timeThreaded << "x" << std::endl;
 
    delete[] _vIN;   
    delete[] _vOUT;   
    delete[] _ICf; 
    delete[] _ICv; 
    delete[] _ICo; 
    delete[] _ICl;    
    delete[] _ECv; 
    delete[] _ECo; 
    delete[] _ISOv; 
    delete[] _wmrSFP;    
    delete[] _wmhSFP;    
    delete[] _isoSFP;    
    delete[] _ICthreads; 
    delete[] _ECthreads; 
    delete[] _ISOthreads;

    */

    return 0;
}
