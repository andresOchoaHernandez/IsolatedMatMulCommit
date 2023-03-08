#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <type_traits>
#include <limits>
#include <cmath>

#include "matrixVecMultiplication.hpp"

template<typename T>
void loadArray(std::string& path,T* array,int len)
{
    std::ifstream  data(path);
    std::string line;

    for(int i = 0 ; i < len ; i++)
    {
        std::getline(data,line);

        if(line.empty())break;

        if(std::is_same<T,double>::value)
        {
            array[i] = std::stod(line); 
        }
        else if(std::is_same<T,float>::value)
        {
            array[i] = std::stof(line);
        }
        else if(std::is_same<T,uint32_t>::value)
        {
            array[i] = static_cast<uint32_t>(std::stoul(line)); 
        }
        else if(std::is_same<T,uint16_t>::value)
        {
            array[i] = static_cast<uint16_t>(std::stoul(line)); 
        }
        
    }
}

void loadDataset(
    int _nF, int _n, int _nE, int _nV, int _nS, int _ndirs,int _nI,int _nR, int _nT,int N, int M,
    float *_vIN, float *_vOUT,
    uint32_t *_ICf, uint32_t *_ICv, uint16_t *_ICo, float *_ICl,
    uint32_t *_ECv, uint16_t *_ECo,
    uint32_t *_ISOv,
    float *_wmrSFP, float *_wmhSFP, float *_isoSFP,
    uint32_t* _ICthreads, uint32_t* _ECthreads, uint32_t* _ISOthreads,int threadsInSys
)
{
    std::string inputPath  = "/home/andres/IsolatedMatMulCommit/dataset/input/";
    int lenInputString = inputPath.size();

    std::string outputPath = "/home/andres/IsolatedMatMulCommit/dataset/output/";
    int lenOutputString = outputPath.size();

    loadArray<float>(inputPath.append("vectorIn.csv"),_vIN,N);
    inputPath = inputPath.substr(0,lenInputString);
        
    loadArray<float>(outputPath.append("vectorOut.csv"),_vOUT,M);

    loadArray<uint32_t>(inputPath.append("icf.csv"),_ICf,_n);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("icv.csv"),_ICv,_n);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint16_t>(inputPath.append("ico.csv"),_ICo,_n);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("icl.csv"),_ICl,_n);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("ecv.csv"),_ECv,_nE);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint16_t>(inputPath.append("eco.csv"),_ECo,_nE);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("isov.csv"),_ISOv,_nV);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("wmrsfp.csv"),_wmrSFP,_nR*_ndirs*_nS);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("wmhsfp.csv"),_wmhSFP,_nT*_ndirs*_nS);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("isosfp.csv"),_isoSFP,_nI*_nS);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("icthreads.csv"),_ICthreads,threadsInSys);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("ecthreads.csv"),_ECthreads,threadsInSys);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("isothreads.csv"),_ISOthreads,threadsInSys);
}