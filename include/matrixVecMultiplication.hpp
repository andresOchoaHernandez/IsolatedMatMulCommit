#pragma once

#include <limits>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

/* =================================== WRAPPER OF ORIGINAL DATA STRUCTURE =========================================== */
class CommitOriginalDataStructure{

    int _nF;
    int _n;
    int _nE;
    int _nV;
    int _nS;
    int _ndirs;
    int _nI;
    int _nR;
    int _nT;
    int N;
    int M;

    std::vector<float> input;
    std::vector<float> output;

    /* IC */
    std::vector<uint32_t> icf;
    std::vector<uint32_t> icv;
    std::vector<uint32_t> ico;
    std::vector<float> icl;

    /* EC */
    std::vector<uint32_t> ecv;
    std::vector<uint32_t> eco;

    /* ISO */
    std::vector<uint32_t> isov;

    /* ===== LOOKUP TABLES ===== */
    
    /* IC */
    std::vector<float> wmrSFP;
    
    /* EC */
    std::vector<float> wmhSFP;

    /* ISO */
    std::vector<float> isoSFP;

    /* ===== THREADS DS   ===== */
    int _threads;
    std::vector<uint32_t> icThreads;
    std::vector<uint32_t> ecThreads;
    std::vector<uint32_t> isoThreads; 

    public:
        CommitOriginalDataStructure(int nF, int n, int nE, int nV, int nS, int ndirs,int nI,int nR, int nT,int threads);
        void loadDataset(std::string& inputPath,std::string& outputPath);
};


/*============================================== UTILS ===============================================================*/

void loadDataset(
    int _nF, int _n, int _nE, int _nV, int _nS, int _ndirs,int _nI,int _nR, int _nT,int N, int M,
    float *_vIN, float *_vOUT,
    uint32_t *_ICf, uint32_t *_ICv, uint16_t *_ICo, float *_ICl,
    uint32_t *_ECv, uint16_t *_ECo,
    uint32_t *_ISOv,
    float *_wmrSFP, float *_wmhSFP, float *_isoSFP,
    uint32_t* _ICthreads, uint32_t* _ECthreads, uint32_t* _ISOthreads,int threadsInSys
);

template<typename T>
bool areNearlyEqual(T a, T b) {
    const T normal_min = std::numeric_limits<T>::min();
    const T relative_error = 0.00001;
    if (!std::isfinite(a) || !std::isfinite(b))
    {
        return false;
    }

    T diff = std::abs(a - b);
    if (diff <= normal_min) 
        return true;

    T abs_a = std::abs(a);
    T abs_b = std::abs(b);

    return (diff / std::max(abs_a, abs_b)) <= relative_error;
}


template<typename T>
bool verifyCorrectness(const T *_vOUT,const T *_vOUT_CORRECT,int M)
{
    for(int i = 0; i < M ; i++)
    {
        if(!areNearlyEqual<T>(_vOUT[i], _vOUT_CORRECT[i]))
        {
            std::cout << "Error at index : " << i << std::endl;
            std::cout << "_vOUT ->  : " << _vOUT[i] << " _vOUT_CORRECT ->  : " << _vOUT_CORRECT[i]<< std::endl;
            return false;
        }
    }
    return true;
}

/*============================================== MATVEC ORIGINAL =====================================================*/

void sequential_matVecMult(
    int _nF, int _n, int _nE, int _nV, int _nS, int _ndirs,int _nI,int _nR, int _nT,int N, int M,
    float *_vIN, float *_vOUT,
    uint32_t *_ICf, uint32_t *_ICv, uint16_t *_ICo, float *_ICl,
    uint32_t *_ECv,uint16_t *_ECo,
    uint32_t *_ISOv,
    float *_wmrSFP, float *_wmhSFP,float *_isoSFP
);

void threaded_matVecMult(
    int _nF, int _n, int _nE, int _nV, int _nS, int _ndirs,
    float *_vIN, float *_vOUT,
    uint32_t *_ICf, uint32_t *_ICv, uint16_t *_ICo, float *_ICl,
    uint32_t *_ECv, uint16_t *_ECo,
    uint32_t *_ISOv,
    float *_wmrSFP, float *_wmhSFP, float *_isoSFP,
    uint32_t* _ICthreads, uint32_t* _ECthreads, uint32_t* _ISOthreads
);