#pragma once

#include <limits>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

#include "LinearAlgebra.hpp"

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
    std::vector<uint16_t> ico;
    std::vector<float> icl;

    /* EC */
    std::vector<uint32_t> ecv;
    std::vector<uint16_t> eco;

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

        void sequentialMatrixMultiplication();
        void threadedMatrixMultiplication();

        LinearAlgebra::CSCMatrix transformToCSC();
        LinearAlgebra::CSRMatrix transformToCSR();

        void CSRSequentialMatrixMultiplication(const LinearAlgebra::CSRMatrix& csrmatrix);
        void CSRGpuMatrixMultiplication(const LinearAlgebra::CSRMatrix& csrmatrix);
};