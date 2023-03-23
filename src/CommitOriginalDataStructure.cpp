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
ecIndexes(_nV,0)
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

void CommitOriginalDataStructure::loadDataset(std::string& inputPath)
{
    unsigned lenInputString = inputPath.size();

    loadArray<float>(inputPath.append("vectorIn.csv"),input);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("icf.csv"),icf);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("icv.csv"),icv);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint16_t>(inputPath.append("ico.csv"),ico);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("icl.csv"),icl);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("ecv.csv"),ecv);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint16_t>(inputPath.append("eco.csv"),eco);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("isov.csv"),isov);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("wmrsfp.csv"),wmrSFP);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("wmhsfp.csv"),wmhSFP);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<float>(inputPath.append("isosfp.csv"),isoSFP);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("icthreads.csv"),icThreads);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("ecthreads.csv"),ecThreads);
    inputPath = inputPath.substr(0,lenInputString);

    loadArray<uint32_t>(inputPath.append("isothreads.csv"),isoThreads);

    /* INITIALIZE CORRECT OUTPUT ARRAY */
    float accumulator;
    int xIndex;
    for(int segment = 0 ; segment < _n ; segment++)
    {
        for(int sample = 0; sample < _nS; sample++)
        {
            accumulator = 0.0f;
            for(int radii = 0 ; radii <  _nR; radii++ )
            {
                accumulator += input[icf[segment] + radii] * wmrSFP[ (radii*_ndirs*_nS) + (ico[segment] * _nS + sample)];
            }
            output[icv[segment]*_nS + sample] += icl[segment]* accumulator;
        }
    }
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
            output[ecv[segment]*_nS + sample] += accumulator;
        }
        xIndex++;
    }
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
            output[isov[i]*_nS + sample] += accumulator;
        }
        xIndex++;
    }
}

template<typename T>
bool areNearlyEqual(T a, T b) {
    const T normal_min = std::numeric_limits<T>::min();
    const T relative_error = 0.000002;
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

/*==================== TESTING VOXEL DIVISION =====================*/
bool testVoxelDivision(const std::vector<uint32_t>& voxelIndexes,const std::vector<int>& helperIndexes)
{
    int initialIndex = 0;
    for(int helperIndex : helperIndexes)
    {
        uint32_t voxel = voxelIndexes[initialIndex];
        for(int index = initialIndex; index < helperIndex; index++ )
        {
            if(voxel != voxelIndexes[index]){
                return false;
            }
        }
        initialIndex = helperIndex;
    }

    return true;
}
/*=================================================================*/

void CommitOriginalDataStructure::orderByVoxel()
{
    struct IcSection
    {
        uint32_t fiber;
        uint32_t voxel;
        uint16_t orientation;
        float    length;

        bool operator<(const IcSection& other)
        {
            return voxel < other.voxel;
        }
    };

    std::vector<IcSection> icSegments;

    for(int seg = 0; seg < _n; seg++)
    {
        icSegments.push_back({.fiber=icf[seg],.voxel=icv[seg],.orientation=ico[seg],.length=icl[seg]});
    }

    std::sort(icSegments.begin(),icSegments.end());

    for(int seg = 0; seg < _n; seg++)
    {
        icf[seg] = icSegments[seg].fiber;
        icv[seg] = icSegments[seg].voxel;
        ico[seg] = icSegments[seg].orientation;
        icl[seg] = icSegments[seg].length;
    }

    uint32_t voxel = icv[0];
    for(int segment = 0; segment < _n; segment++)
    {
        if(icv[segment] != voxel)
        {
            icIndexes[voxel] = segment;
            voxel = icv[segment];
        }
    }

    icIndexes[_nV-1] =_n-1;

    if(!testVoxelDivision(icv,icIndexes)){std::cout << "Error in voxel division for ic section" << std::endl;}

    struct EcSection
    {
        uint32_t voxel;
        uint16_t orientation;

        bool operator<(const EcSection& other)
        {
            return voxel < other.voxel;
        }
    };

    std::vector<EcSection> ecSegments;

    for(int seg = 0; seg < _nE; seg++)
    {
        ecSegments.push_back({.voxel=ecv[seg],.orientation=eco[seg]});
    }

    std::sort(ecSegments.begin(),ecSegments.end());

    for(int seg = 0; seg < _nE; seg++)
    {
        ecv[seg] = ecSegments[seg].voxel;
        eco[seg] = ecSegments[seg].orientation;
    }

    voxel = ecv[0];
    for(int segment = 0; segment < _nE; segment++)
    {
        if(ecv[segment] != voxel)
        {
            ecIndexes[voxel] = segment;
            voxel = ecv[segment];
        }
    }

    ecIndexes[_nV-1] =_nE-1;

    if(!testVoxelDivision(ecv,ecIndexes)){std::cout << "Error in voxel division for ec section" << std::endl;}

    std::sort(isov.begin(),isov.end());
}