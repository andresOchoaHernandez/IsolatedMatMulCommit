#include <iostream>
#include <thread>

#include "CommitOriginalDataStructure.hpp"

int main()
{
    int _nS       = 100;
    int _nF       = 283522;
    int _nR       = 1;
    int _nE       = 145434;
    int _nT       = 1;
    int _nV       = 53008;
    int _nI       = 2;
    int _n        = 15825021;
    int _ndirs    = 500;

    const int threads = 1;

    CommitOriginalDataStructure originalDataStructure(_nF, _n, _nE, _nV, _nS, _ndirs,_nI,_nR, _nT,threads);

    
    originalDataStructure.loadDataset();

    originalDataStructure.prepareBatchedLUT();

    /*
    originalDataStructure.sequentialMatrixMultiplication();
    originalDataStructure.threadedMatrixMultiplication();

    originalDataStructure.generateIndexesVector();

    originalDataStructure.gpuMatrixMultiplication();
    */

    return 0;
}
