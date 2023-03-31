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
output(M)
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

    begin = std::chrono::steady_clock::now();
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
    end = std::chrono::steady_clock::now();
    long int timeInitOutput = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << "------------------ Loading dataset ------------------"     << std::endl
              << "| time                     => " << timeLoading    << " ms" << std::endl
              << "| time initializing output => " << timeInitOutput << " ms" << std::endl
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
                      "Error found : correct[" << i << "] = " << correct[i] << ", obatined[" << i << "] = " << obtained[i] << std::endl; 
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

LinearAlgebra::CSCMatrix CommitOriginalDataStructure::transformToCSC()
{
    /* IC */
    struct Segment
    {
        uint32_t fiber;
        uint32_t voxel;
        uint16_t orientation;
        float    length;

        bool operator<(const Segment& other){ return fiber == other.fiber? voxel < other.voxel : fiber < other.fiber; }
    };

    struct SegmentsPerFiber
    {
        uint32_t fiber;
        unsigned segments;
    };

    std::vector<Segment> segments;

    for(int i = 0 ; i < _n ; i++)
    {
        Segment s;
        s.fiber       = icf[i];
        s.voxel       = icv[i];
        s.orientation = ico[i];
        s.length      = icl[i];

        segments.push_back(s);
    }

    std::sort(segments.begin(),segments.end());

    std::vector<SegmentsPerFiber> segmentsperfiber;
    unsigned totalVoxelsCrossedByFibers = 0u;

    {
        std::deque<Segment> queue(segments.begin(),segments.end());

        for(int fiber = 0 ; fiber < _nF ; fiber++)
        {   
            std::vector<uint32_t> voxels;

            unsigned segmentscount = 0u;

            for(Segment seg : queue)
            {
                if(seg.fiber != static_cast<uint32_t>(fiber)) break;
                
                voxels.push_back(seg.voxel);
                queue.pop_front();

                segmentscount++;
            }

            totalVoxelsCrossedByFibers += std::set<uint32_t>(voxels.begin(),voxels.end()).size();
        
            SegmentsPerFiber count;
            count.fiber = fiber;
            count.segments = segmentscount;

            segmentsperfiber.push_back(count);
        }
    }

    unsigned nonZeroElements = static_cast<unsigned>(_nS) *
        (totalVoxelsCrossedByFibers*static_cast<unsigned>(_nR)+static_cast<unsigned>(_nT)*static_cast<unsigned>(_nE)+static_cast<unsigned>(_nV)*static_cast<unsigned>(_nI));

    LinearAlgebra::CSCMatrix cscALinearOperator{static_cast<unsigned>(M),static_cast<unsigned>(N),nonZeroElements}; 

    unsigned* colsVec = cscALinearOperator.getColsArray();
    unsigned* rowsVec = cscALinearOperator.getRowsArray();
    float*   valsVec = cscALinearOperator.getValsArray();

    unsigned segmentsOffset = 0u;
    
    unsigned columnIndex = 0u;
    unsigned nonZeroValueIndex = 0u;

    colsVec[columnIndex++] = 0u;

    for(SegmentsPerFiber spf : segmentsperfiber)
    {
        for(int radii = 0 ; radii < _nR ; radii++)
        {
            std::map<uint32_t,std::vector<Segment>> segmentsinvoxel;

            for(unsigned segment = segmentsOffset ; segment < segmentsOffset + spf.segments ; segment++)
            {
                if(segmentsinvoxel.find(segments[segment].voxel) == segmentsinvoxel.end())
                {
                    segmentsinvoxel[segments[segment].voxel] = std::vector<Segment>{segments[segment]};
                }
                else
                {
                    segmentsinvoxel[segments[segment].voxel].push_back(segments[segment]);
                }
            }

            for(const auto& pair : segmentsinvoxel)
            {
                unsigned rowStart = pair.first * static_cast<unsigned>(_nS);
                
                std::vector<float> nonZeroValue(_nS,0.0);

                for(const auto& seg : pair.second)
                {
                    for(unsigned sample = 0u ; sample < static_cast<unsigned>(_nS) ; sample++)
                    {
                        nonZeroValue[sample] += seg.length * wmrSFP[(radii*_ndirs*_nS) + (seg.orientation*_nS + sample)];       
                    }
                }

                for(unsigned sample = 0u ; sample < static_cast<unsigned>(_nS) ; sample++)
                {
                    rowsVec[ nonZeroValueIndex ] = rowStart + sample;
                    valsVec[ nonZeroValueIndex ] = nonZeroValue[sample];
                    
                   nonZeroValueIndex++;
                }
            }

            colsVec[columnIndex++] = nonZeroValueIndex;
        }
        segmentsOffset += spf.segments;
    }

    std::vector<Segment>().swap(segments);
    std::vector<SegmentsPerFiber>().swap(segmentsperfiber);

    /* EC */
    for(int ec = 0; ec < _nT ; ec++)
    {
        for(int ecSegment = 0 ; ecSegment < _nE ; ecSegment++)
        {
            for(int sample = 0; sample < _nS ; sample++)
            {
                rowsVec[ nonZeroValueIndex ] = ecv[ecSegment] * static_cast<unsigned>(_nS) + sample;
                valsVec[ nonZeroValueIndex ] = wmhSFP[(ec*_ndirs*_nS) + (eco[ecSegment] * _nS + sample)];
                nonZeroValueIndex++;
            }
            colsVec[columnIndex++] = nonZeroValueIndex;
        }
    }

    /* ISO */ 
    for(int iso = 0 ; iso < _nI ; iso++)
    {
        for(int voxel = 0 ; voxel < _nV ; voxel++)
        {
            for(int sample = 0; sample < _nS; sample++)
            {
                rowsVec[ nonZeroValueIndex ] = isov[voxel] * static_cast<unsigned>(_nS) + sample;
                valsVec[ nonZeroValueIndex ] = isoSFP[iso*_nS + sample];

                nonZeroValueIndex++;
            }
            colsVec[columnIndex++] = nonZeroValueIndex;
        }
    }

    return cscALinearOperator;
}

LinearAlgebra::CSRMatrix CommitOriginalDataStructure::transformToCSR()
{

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

     /* IC */
    struct Segment
    {
        uint32_t fiber;
        uint32_t voxel;
        uint16_t orientation;
        float    length;

        bool operator<(const Segment& other){ return fiber == other.fiber? voxel < other.voxel : fiber < other.fiber; }
    };

    struct SegmentsPerFiber
    {
        uint32_t fiber;
        unsigned segments;
    };

    std::vector<Segment> segments;

    for(int i = 0 ; i < _n ; i++)
    {
        Segment s;
        s.fiber       = icf[i];
        s.voxel       = icv[i];
        s.orientation = ico[i];
        s.length      = icl[i];

        segments.push_back(s);
    }

    std::sort(segments.begin(),segments.end());

    std::deque<Segment> queue(segments.begin(),segments.end());

    std::vector<SegmentsPerFiber> segmentsperfiber;

    unsigned totalVoxelsCrossedByFibers = 0u;

    for(int fiber = 0 ; fiber < _nF ; fiber++)
    {   
        std::vector<uint32_t> voxels;

        unsigned segmentscount = 0u;

        for(Segment seg : queue)
        {
            if(seg.fiber != static_cast<uint32_t>(fiber)) break;
            
            voxels.push_back(seg.voxel);
            queue.pop_front();

            segmentscount++;
        }

        totalVoxelsCrossedByFibers += std::set<uint32_t>(voxels.begin(),voxels.end()).size();
    
        SegmentsPerFiber count;
        count.fiber = fiber;
        count.segments = segmentscount;

        segmentsperfiber.push_back(count);
    }

    std::deque<Segment>().swap(queue);

    unsigned nonZeroElements = static_cast<unsigned>(_nS) *
        (totalVoxelsCrossedByFibers*static_cast<unsigned>(_nR)+static_cast<unsigned>(_nT)*static_cast<unsigned>(_nE)+static_cast<unsigned>(_nV)*static_cast<unsigned>(_nI));

    struct duo
    {
        unsigned column;
        float value;
    };

    std::map<unsigned,std::vector<duo>> rows;

    unsigned segmentsOffset = 0u;
    unsigned columnIndex = 0u;

    for(SegmentsPerFiber spf : segmentsperfiber)
    {
        for(int radii = 0 ; radii < _nR ; radii++)
        {
            std::map<uint32_t,std::vector<Segment>> segmentsinvoxel;

            for(unsigned segment = segmentsOffset ; segment < segmentsOffset + spf.segments ; segment++)
            {
                if(segmentsinvoxel.find(segments[segment].voxel) == segmentsinvoxel.end())
                {
                    segmentsinvoxel[segments[segment].voxel] = std::vector<Segment>{segments[segment]};
                }
                else
                {
                    segmentsinvoxel[segments[segment].voxel].push_back(segments[segment]);
                }
            }

            for(const auto& pair : segmentsinvoxel)
            {
                unsigned rowStart = pair.first * static_cast<unsigned>(_nS);
                
                std::vector<float> nonZeroValue(_nS,0.0);

                for(const auto& seg : pair.second)
                {
                    for(unsigned sample = 0u ; sample < static_cast<unsigned>(_nS) ; sample++)
                    {
                        nonZeroValue[sample] += seg.length * wmrSFP[(radii*_ndirs*_nS) + (seg.orientation*_nS + sample)];       
                    }
                }

                for(unsigned sample = 0u ; sample < static_cast<unsigned>(_nS) ; sample++)
                {
                    unsigned row = rowStart + sample;
                    unsigned column = columnIndex;
                    float value = nonZeroValue[sample];

                    if(rows.find(row) == rows.end())
                    {
                        duo elem;
                        elem.column = column;
                        elem.value = value;
                        rows[row] = std::vector<duo>{elem};
                    }
                    else
                    {
                        duo elem;
                        elem.column = column;
                        elem.value = value;
                        rows[row].push_back(elem);
                    }
                }
            }
            columnIndex++;
        }
        segmentsOffset += spf.segments;
    }

    std::vector<Segment>().swap(segments);
    std::vector<SegmentsPerFiber>().swap(segmentsperfiber);

    /* EC */
    for(int ec = 0; ec < _nT ; ec++)
    {
        for(int ecSegment = 0 ; ecSegment < _nE ; ecSegment++)
        {
            for(int sample = 0; sample < _nS ; sample++)
            {
                unsigned row = ecv[ecSegment] * static_cast<unsigned>(_nS) + sample;
                unsigned column = columnIndex;
                float value = wmhSFP[(ec*_ndirs*_nS) + (eco[ecSegment] * _nS + sample)];

                if(rows.find(row) == rows.end())
                {
                    duo elem;
                    elem.column = column;
                    elem.value = value;
                    rows[row] = std::vector<duo>{elem};
                }
                else
                {
                    duo elem;
                    elem.column = column;
                    elem.value = value;
                    rows[row].push_back(elem);
                }
            }
            columnIndex++;
        }
    }

    /* ISO */ 
    for(int iso = 0 ; iso < _nI ; iso++)
    {
        for(int voxel = 0 ; voxel < _nV ; voxel++)
        {
            for(int sample = 0; sample < _nS; sample++)
            {

                unsigned row = isov[voxel] * static_cast<unsigned>(_nS) + sample;
                unsigned column = columnIndex;
                float value = isoSFP[iso*_nS + sample];

                if(rows.find(row) == rows.end())
                {
                    duo elem;
                    elem.column = column;
                    elem.value = value;
                    rows[row] = std::vector<duo>{elem};
                }
                else
                {
                    duo elem;
                    elem.column = column;
                    elem.value = value;
                    rows[row].push_back(elem);
                }
            }
            columnIndex++;
        }
    }

    using LinearAlgebra::CSRMatrix;

    CSRMatrix result{static_cast<unsigned>(M),static_cast<unsigned>(N),nonZeroElements};
    
    unsigned* rowsVec = result.getRowsArray();
    unsigned* colsVec = result.getColsArray();
    float*    valsVec = result.getValsArray();

    unsigned rowsVecIndex = 0u;
    unsigned nonZeroElementIndex = 0u;

    rowsVec[rowsVecIndex++] = 0u;

    for(auto& element : rows )
    {
        for(const auto& duos : element.second )
        {
            colsVec[nonZeroElementIndex] = duos.column;
            valsVec[nonZeroElementIndex] = duos.value;

            nonZeroElementIndex++;
        }
        rowsVec[rowsVecIndex++] = nonZeroElementIndex;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    long int time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << "------------------- Trasformation to CSR format -------------------" << std::endl;
    std::cout << "| time => " << time << " ms" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;

    return result;
}

void CommitOriginalDataStructure::CSRSequentialMatrixMultiplication(const LinearAlgebra::CSRMatrix& csrmatrix)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    LinearAlgebra::Vector inputVector{static_cast<unsigned>(input.size())};

    for(unsigned i = 0; i < inputVector.len(); i++)
    {
        inputVector[i] = input[i];
    }

    LinearAlgebra::Vector outputVector = csrmatrix.matrixVectorMult(inputVector);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    long int time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::vector<float> _outputVector(outputVector.len());

    for (size_t i = 0; i < _outputVector.size();i++)
    {
        _outputVector[i] = outputVector[i];
    }

    printResult("Sequential CSR matrix multiplication",output,_outputVector,verifyCorrectness<float>(output,_outputVector),time);
}

void CommitOriginalDataStructure::CSRGpuMatrixMultiplication(const LinearAlgebra::CSRMatrix& csrmatrix)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    LinearAlgebra::Vector inputVector{static_cast<unsigned>(input.size())};

    for(unsigned i = 0; i < inputVector.len(); i++)
    {
        inputVector[i] = input[i];
    }

    LinearAlgebra::Vector outputVector = csrmatrix.gpu_cuSparse_matrixVectorMult(inputVector);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    long int time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::vector<float> _outputVector(outputVector.len());

    for (size_t i = 0; i < _outputVector.size();i++)
    {
        _outputVector[i] = outputVector[i];
    }

    printResult("[CuSparse] Gpu CSR matrix multiplication",output,_outputVector,verifyCorrectness<float>(output,_outputVector),time);
}