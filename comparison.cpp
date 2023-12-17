#include <iostream>
#include <thread>

#include "CommitOriginalDataStructure.hpp"

int main()
{
    int _nS       = 1;
    int _nF       = 2937738;
    int _nR       = 1;
    int _nE       = 0;
    int _nT       = 0;
    int _nV       = 274948;
    int _nI       = 0;
    int _n        = 153015729;
    int _ndirs    = 1;

    const int threads = 1;

    CommitOriginalDataStructure originalDataStructure(_nF, _n, _nE, _nV, _nS, _ndirs,_nI,_nR, _nT,threads);
    
    originalDataStructure.sequentialMatrixMultiplication();
    originalDataStructure.threadedMatrixMultiplication();
    originalDataStructure.gpuMatrixMultiplication();
    
    return 0;
}
