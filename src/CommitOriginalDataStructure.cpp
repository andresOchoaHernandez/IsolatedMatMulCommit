#include <fstream>
#include <chrono>
#include <iomanip>
#include <bits/stdc++.h>

#include "CommitOriginalDataStructure.hpp"
#include "ThreadedMatrixVecMultiplication.hpp"

CommitOriginalDataStructure::CommitOriginalDataStructure(int nF, int n, int nE, int nV, int nS, int ndirs,int nI,int nR, int nT,int threads):
_nF{nF},
_n{n},
_nE{nE},
_nV{nV},
_nS{nS},
_ndirs{ndirs},
_nI{nI},
_nR{nR},
_nT{nT},
M{_nV*_nS},
N{_nR*_nF + _nT*_nE + _nI*_nV},
icf(_n),
icv(_n),
ico(_n),
icl(_n),
ecv(_nE),
eco(_nE),
isov(_nV),
wmrSFP(_nR*_ndirs*_nS),
wmhSFP(_nT*_ndirs*_nS),
isoSFP(_nI*_nS),
_threads(threads),
icThreads(_threads + 1),
ecThreads(_threads + 1),
isoThreads(_threads + 1),
input(N),
output(M),
icIndexes(_nV,0),
ecIndexes(_nV,0),
batchedLUTs(1 + ((nS-1)/SAMPLE_TILE_LENGTH))
{}

template<typename T>
void loadArray(const std::string& path,std::vector<T>& array)
{
    std::ifstream data(path);
    std::string line;

    for(size_t i = 0 ; i < array.size() ; i++)
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

void CommitOriginalDataStructure::loadDataset()
{
std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
       #pragma omp single
       {
            #pragma omp task
            {
                loadArray<float>("../dataset/input/vectorIn.csv",input);
            }
            #pragma omp task
            {
                loadArray<uint32_t>("../dataset/input/icf.csv",icf);
            }
            #pragma omp task
            {
                loadArray<uint32_t>("../dataset/input/icv.csv",icv);
            }
            #pragma omp task
            {
                loadArray<uint16_t>("../dataset/input/ico.csv",ico);
            }
            #pragma omp task
            {
                loadArray<float>("../dataset/input/icl.csv",icl);
            }
            #pragma omp task
            {
                loadArray<uint32_t>("../dataset/input/ecv.csv",ecv);
            }
            #pragma omp task
            {
                loadArray<uint16_t>("../dataset/input/eco.csv",eco);
            }
            #pragma omp task
            {
                loadArray<uint32_t>("../dataset/input/isov.csv",isov);
            }
            #pragma omp task
            {
                loadArray<float>("../dataset/input/wmrsfp.csv",wmrSFP);
            }
            #pragma omp task
            {
                loadArray<float>("../dataset/input/wmhsfp.csv",wmhSFP);
            }
            #pragma omp task
            {
                loadArray<float>("../dataset/input/isosfp.csv",isoSFP);
            }
            #pragma omp task
            {
                loadArray<uint32_t>("../dataset/input/icthreads.csv",icThreads);
            }
            #pragma omp task
            {
                loadArray<uint32_t>("../dataset/input/ecthreads.csv",ecThreads);
            }
            #pragma omp task
            {
                loadArray<uint32_t>("../dataset/input/isothreads.csv",isoThreads);
            }
       } 
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    long int timeLoading = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    
    /* INITIALIZE CORRECT OUTPUT ARRAY */
    threaded_matVecMult(
        _nF, _n, _nE, _nV, _nS, _ndirs,
        input.data(),output.data(),
        icf.data(),icv.data(),ico.data(),icl.data(),
        ecv.data(),eco.data(),
        isov.data(),
        wmrSFP.data(),wmhSFP.data(),isoSFP.data(),
        icThreads.data(),ecThreads.data(),isoThreads.data()
    );

    std::cout << "------------------ Loading dataset ------------------"     << std::endl
              << "| time => " << timeLoading    << " ms" << std::endl
              << "-----------------------------------------------------"     << std::endl;
}

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
bool verifyCorrectness(const std::vector<T>& correct,const std::vector<T>& obtained)
{
    if (correct.size() != obtained.size())
    {
        std::cout << "Correct and obtained vectors don't have same size!" << std::endl;
        return false;
    }

    for(size_t i = 0;i < correct.size();i++)
    {
        if(!areNearlyEqual<T>(correct[i],obtained[i]))
        {
            std::cout << std::fixed << std::setprecision(6) <<
                      "Error found : correct[" << i << "] = " << correct[i] << ", obtained[" << i << "] = " << obtained[i] << std::endl; 
            return false;
        }
    }
    return true;
}

float calculateAverageAbsoluteError(const std::vector<float>& correct,const std::vector<float>& obtained)
{
    if (correct.size() != obtained.size())
    {
        std::cout << "Correct and obtained vectors don't have same size!" << std::endl;
        return false;
    }

    float accAbsErr = 0.0f;

    for(size_t i = 0;i < correct.size();i++)
    {
        accAbsErr += std::abs(correct[i] - obtained[i]);
    }
    return accAbsErr/static_cast<float>(correct.size());
}

void printResult(const std::string& message,const std::vector<float>& correct,const std::vector<float>& obtained,bool correctness, long int time){

    const std::string upperSepSx  = "------------------ ";
    const std::string upperSepDx  = " ------------------";
    const std::string downerSep(upperSepSx.length()*2+message.length(),'-'); 

    std::cout << upperSepSx << message << upperSepDx                                    << std::endl
              << "| correct     => " << ((correctness)? "true":"false")                 << std::endl
              << "| time        => " << time << " ms"                                   << std::endl
              << "| avg abs err => " << calculateAverageAbsoluteError(correct,obtained) << std::endl
              << downerSep                                                              << std::endl;
}

void CommitOriginalDataStructure::sequentialMatrixMultiplication(){

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    std::vector<float> outputVector(output.size(),0);

    float accumulator;
    int xIndex;

    /* IC */
    for(int segment = 0 ; segment < _n ; segment++)
    {
        for(int sample = 0; sample < _nS; sample++)
        {
            accumulator = 0.0f;
            for(int radii = 0 ; radii <  _nR; radii++ )
            {
                accumulator += input[icf[segment] + radii] * wmrSFP[ (radii*_ndirs*_nS) + (ico[segment] * _nS + sample)];
            }
            outputVector[icv[segment]*_nS + sample] += icl[segment]* accumulator;
        }
    }

    /* EC */
    xIndex = _nR * _nF;
    for(int segment = 0; segment < _nE; segment++)
    {
        for(int sample = 0; sample < _nS ; sample++)
        {
            accumulator = 0.0f;
            for(int ec = 0; ec < _nT ; ec++)
            {
                accumulator += input[xIndex + ec * _nE] * wmhSFP[(ec*_ndirs*_nS) + (eco[segment] * _nS + sample)];
            }
            outputVector[ecv[segment]*_nS + sample] += accumulator;
        }
        xIndex++;
    }

    /* ISO */
    xIndex = _nR*_nF + _nT*_nE;
    for(int i = 0; i < _nV ; i++)
    {
        for(int sample = 0 ; sample < _nS ; sample++)
        {
            accumulator = 0.0f;
            for(int iso = 0; iso < _nI ; iso++)
            {
                accumulator += input[xIndex + iso*_nV] * isoSFP[iso*_nS + sample];
            }
            outputVector[isov[i]*_nS + sample] += accumulator;
        }
        xIndex++;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    long int time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    printResult("Sequential matrix multiplication",output,outputVector,verifyCorrectness<float>(output,outputVector),time);
}

void CommitOriginalDataStructure::threadedMatrixMultiplication(){

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<float> outputVector(output.size(),0);
    
    threaded_matVecMult(
        _nF, _n, _nE, _nV, _nS, _ndirs,
        input.data(),outputVector.data(),
        icf.data(),icv.data(),ico.data(),icl.data(),
        ecv.data(),eco.data(),
        isov.data(),
        wmrSFP.data(),wmhSFP.data(),isoSFP.data(),
        icThreads.data(),ecThreads.data(),isoThreads.data()
    );

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    long int time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    printResult("Threaded matrix multiplication",output,outputVector,verifyCorrectness<float>(output,outputVector),time);
}

void CommitOriginalDataStructure::generateIndexesVector()
{
    if(_n > 0 && _nR > 0)
    {
        uint32_t voxel = icv[0];
        for(int segment = 0; segment < _n; segment++)
        {
            if(icv[segment] != voxel)
            {
                icIndexes[voxel] = segment;
                voxel = icv[segment];
            }
        }

        icIndexes[_nV-1] =_n;
    }

    if(_nE > 0 && _nT > 0)
    {
        uint32_t voxel = ecv[0];
        for(int segment = 0; segment < _nE; segment++)
        {
            if(ecv[segment] != voxel)
            {
                ecIndexes[voxel] = segment;
                voxel = ecv[segment];
            }
        }

        ecIndexes[_nV-1] =_nE;
    }
}

void CommitOriginalDataStructure::prepareBatchedLUT()
{
    for(unsigned batch = 0; batch < batchedLUTs.size(); batch++)
    {

        LUTBatch LUT;

        unsigned SAMPLE_BATCH_OFFSET = batch * SAMPLE_TILE_LENGTH;

        /* IC */
        for(unsigned radius = 0; radius < _nR ; radius++)
        {
            for(unsigned direction = 0 ; direction < _ndirs ; direction++)
            {
                for(unsigned sample = 0 ; sample < SAMPLE_TILE_LENGTH ; sample++)
                {
                    if(SAMPLE_BATCH_OFFSET + sample < _nS)
                    {
                        LUT.wmrSFP.push_back(wmrSFP[(radius*_ndirs*_nS) + (direction * _nS + SAMPLE_BATCH_OFFSET + sample)]);
                    }
                    else
                    {
                        LUT.wmrSFP.push_back(1.0f);
                    }
                }
            }
        }

        /* EC */
        for(unsigned ec = 0; ec < _nT ; ec++)
        {
            for(unsigned direction = 0 ; direction < _ndirs ; direction++)
            {
                for(unsigned sample = 0 ; sample < SAMPLE_TILE_LENGTH ; sample++)
                {
                    if(SAMPLE_BATCH_OFFSET + sample < _nS)
                    {
                        LUT.wmhSFP.push_back(wmhSFP[(ec*_ndirs*_nS) + (direction * _nS + SAMPLE_BATCH_OFFSET + sample)]);
                    }
                    else
                    {
                        LUT.wmhSFP.push_back(1.0f);
                    }
                }
            }
        }

        /* ISO */
        for(unsigned iso = 0; iso < _nI ; iso++)
        {
            for(unsigned sample = 0 ; sample < SAMPLE_TILE_LENGTH ; sample++)
            {
                if(SAMPLE_BATCH_OFFSET + sample < _nS)
                {
                    LUT.isoSFP.push_back(isoSFP[(iso*_nS) + SAMPLE_BATCH_OFFSET + sample]);
                }
                else
                {
                    LUT.isoSFP.push_back(1.0f);
                }
            }
        }

        batchedLUTs[batch] = LUT;
    }
}