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
    int _n        = 16664340;
    int _ndirs    = 32761;

    const int threads = std::thread::hardware_concurrency();

    CommitOriginalDataStructure originalDataStructure(_nF, _n, _nE, _nV, _nS, _ndirs,_nI,_nR, _nT,threads);

    std::string input  = "../dataset/input/";
    std::string output = "../dataset/output/";

    originalDataStructure.loadDataset(input,output);

    originalDataStructure.sequentialMatrixMultiplication();
    originalDataStructure.threadedMatrixMultiplication();

    return 0;
}