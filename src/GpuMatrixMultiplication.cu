#include "CommitOriginalDataStructure.hpp"

#include <iomanip>

/* ========================================================================================== */
template<typename T>
bool areNearlyEqualGpu(T a, T b) {
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
bool verifyCorrectnessGpu(const std::vector<T>& correct,const std::vector<T>& obtained)
{
    if (correct.size() != obtained.size())
    {
        std::cout << "Correct and obtained vectors don't have same size!" << std::endl;
        return false;
    }

    for(size_t i = 0;i < correct.size();i++)
    {
        if(!areNearlyEqualGpu<T>(correct[i],obtained[i]))
        {
            std::cout << std::fixed << std::setprecision(6) <<
                      "Error found : correct[" << i << "] = " << correct[i] << ", obtained[" << i << "] = " << obtained[i] << ", abs err : " << std::abs(correct[i] - obtained[i])<< std::endl; 
            return false;
        }
    }
    return true;
}

float gpuCalculateAverageAbsoluteError(const std::vector<float>& correct,const std::vector<float>& obtained)
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

void printResultGpu(const std::string& message,const std::vector<float>& correct,const std::vector<float>& obtained,bool correctness,float kernelTime,float totalTime){

    const std::string upperSepSx  = "------------------ ";
    const std::string upperSepDx  = " ------------------";
    const std::string downerSep(upperSepSx.length()*2+message.length(),'-'); 

    std::cout << upperSepSx << message << upperSepDx                                       << std::endl
              << "| correct        => " << ((correctness)? "true":"false")                    << std::endl
              << "| kernel time    => " << kernelTime << " ms"                             << std::endl
              << "| total  time    => " << totalTime  << " ms"                             << std::endl
              << "| avg abs err    => " << gpuCalculateAverageAbsoluteError(correct,obtained) << std::endl
              << downerSep                                                                 << std::endl;

}
/* ========================================================================================== */

#define CUDAERRCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
    QUADDRO P6000 => 
                        max_threads_per_sm : 2048
                        max_blocks_per_sm  : 32
                        -------------------------
                        threads per block to achieve max occupancy : 64

                        n_sm : 30
                        -------------------------
                        to achieve full GPU occupancy : 960 (or a multiple of it) blocks of 64 threads

    RTX 2060      => 
                        max_threads_per_sm : 1024
                        max_blocks_per_sm  : 16
                        -------------------------
                        threads per block to achieve max occupancy : 64

                        n_sm : 30
                        -------------------------
                        to achieve full GPU occupancy : 480 (or a multiple of it) blocks of 64 threads

after IC: 1688.344360
after EC: 181036.250000
after ISO: 449822.250000
*/

#define THREADS_PER_BLOCK 100
#define BLOCKS 4800

__global__ void icMatrixMultiplication(
    int nS,int nV,
    uint32_t* icfDevice, uint32_t* icvDevice, uint16_t* icoDevice, float* iclDevice, int nR,int nF,
    float* wmrSFPDevice, int ndirs,
    int* icIndexesDevice,
    float* xDevice,
    float* yDevice
)
{
    const int startIcSegment = (blockIdx.x == 0)?0:icIndexesDevice[blockIdx.x-1];
    const int endIcSegment   = icIndexesDevice[blockIdx.x];

    float accumulator = 0.0f;

    for(int icsegment = startIcSegment; icsegment < endIcSegment; icsegment++)
    {
        int fiber       = icfDevice[icsegment];
        int voxel       = icvDevice[icsegment];
        int orientation = icoDevice[icsegment];
        float length    = iclDevice[icsegment];

        for (int radii = 0; radii < nR; radii++)
        {
            accumulator += xDevice[fiber + radii]*wmrSFPDevice[radii*ndirs*nS + orientation*nS + threadIdx.x]*length;
        }

        yDevice[voxel * nS + threadIdx.x] += accumulator;
    }
}
__global__ void ecMatrixMultiplication(
    int nS,int nV,int nR,int nF,
    uint32_t* ecvDevice, uint16_t* ecoDevice, int nT,int nE,
    float* wmhSFPDevice,int ndirs,
    int* ecIndexesDevice,
    float* xDevice,
    float* yDevice
)
{
    const int startEcSegment = (blockIdx.x == 0)?0:ecIndexesDevice[blockIdx.x-1];
    const int endEcSegment   = ecIndexesDevice[blockIdx.x];

    float accumulator = 0.0f;

    int xIndex = nR*nF + startEcSegment;
    for(int ecsegment = startEcSegment; ecsegment < endEcSegment; ecsegment++)
    {
        int voxel       = ecvDevice[ecsegment]; 
        int orientation = ecoDevice[ecsegment];

        for (int tortuosity = 0; tortuosity < nT; tortuosity++)
        {
            accumulator += xDevice[xIndex + tortuosity*nE]*wmhSFPDevice[tortuosity*ndirs*nS + orientation * nS + threadIdx.x];
        }
        xIndex++;
        
        yDevice[voxel * nS + threadIdx.x] += accumulator;
    }
}
__global__ void isoMatrixMultiplication(
    int nS,int nV,int nR,int nF,int nE,int nT,
    uint32_t* isovDevice, int nI,
    float* isoSFPDevice,int ndirs,
    int* isoIndexesDevice,
    float* xDevice,
    float* yDevice
)
{
    const int startIsoSegment = (blockIdx.x == 0)?0:isoIndexesDevice[blockIdx.x-1];
    const int endIsoSegment   = isoIndexesDevice[blockIdx.x];

    int previousVoxel = -1;

    float accumulator = 0.0f;

    for(int isosegment = startIsoSegment; isosegment < endIsoSegment; isosegment++)
    {
        int voxel = isovDevice[isosegment];

        for (int iso = 0; iso < nI; iso++)
        {
            accumulator += xDevice[(nR*nF + nT*nE + voxel) + iso*nV]*isoSFPDevice[iso * nS + threadIdx.x];
        }
        if(previousVoxel != voxel)
        {
            yDevice[voxel * nS + threadIdx.x] = accumulator;
        }

        previousVoxel = voxel;
    }
}

void CommitOriginalDataStructure::generateHelperVectors(std::vector<int>& icIndexes,std::vector<int>& ecIndexes,std::vector<int>& isoIndexes)
{
    const int icStride  = _n/BLOCKS;
    const int ecStride  = _nE/BLOCKS;
    const int isoStride = _nV/BLOCKS;

    for(int i = 0; i < BLOCKS - 1;i++)
    {
        /* IC */
        int currIndex = (i+1)*icStride;
        int currVoxel  = icv[currIndex];
        int leftVoxel  = icv[currIndex-1];

        if (leftVoxel != currVoxel)
        {
            icIndexes[i] = currIndex;
        }
        else if(leftVoxel == currVoxel)
        {
            while(icv[currIndex] == currVoxel)
            {
                currIndex++;                
            }

            icIndexes[i] = currIndex;
        }

        /* EC */
        currIndex = (i+1)*ecStride;
        currVoxel  = ecv[currIndex];
        leftVoxel  = ecv[currIndex-1];
        if (leftVoxel != currVoxel)
        {
            ecIndexes[i] = currIndex;
        }
        else if(leftVoxel == currVoxel)
        {
            while(ecv[currIndex] == currVoxel)
            {
                currIndex++;                
            }

            ecIndexes[i] = currIndex;
        }

        /* ISO */
        isoIndexes[i] = (i+1)*isoStride;
    }

    icIndexes[BLOCKS-1]  = _n;
    ecIndexes[BLOCKS-1]  = _nE;
    isoIndexes[BLOCKS-1] = _nV;
}

void CommitOriginalDataStructure::gpuMatrixMultiplication()
{
    /* GENERATE HELPER VECTORS */
    std::vector<int> icIndexes(BLOCKS,0);
    std::vector<int> ecIndexes(BLOCKS,0);
    std::vector<int> isoIndexes(BLOCKS,0);

    generateHelperVectors(icIndexes,ecIndexes,isoIndexes);

    /* CREATING EVENTS HANDLERS TO MEASURE EXECUTION TIME */
    cudaEvent_t totalStart,totalStop;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);

    cudaEvent_t kernelStart,kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(totalStart);

    /* IC */
    uint32_t* icfDevice; uint32_t* icvDevice; uint16_t* icoDevice; float* iclDevice;

    CUDAERRCHECK(cudaMalloc(&icfDevice,sizeof(uint32_t)*icf.size()))
    CUDAERRCHECK(cudaMalloc(&icvDevice,sizeof(uint32_t)*icv.size()))
    CUDAERRCHECK(cudaMalloc(&icoDevice,sizeof(uint16_t)*ico.size()))
    CUDAERRCHECK(cudaMalloc(&iclDevice,sizeof(float)*icl.size()))

    CUDAERRCHECK(cudaMemcpy(icfDevice,icf.data(),sizeof(uint32_t)*icf.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(icvDevice,icv.data(),sizeof(uint32_t)*icv.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(icoDevice,ico.data(),sizeof(uint16_t)*ico.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(iclDevice,icl.data(),sizeof(float)*icl.size(),cudaMemcpyHostToDevice))

    /* EC */
    uint32_t* ecvDevice; uint16_t* ecoDevice;

    CUDAERRCHECK(cudaMalloc(&ecvDevice,sizeof(uint32_t)*ecv.size()))
    CUDAERRCHECK(cudaMalloc(&ecoDevice,sizeof(uint16_t)*eco.size()))

    CUDAERRCHECK(cudaMemcpy(ecvDevice,ecv.data(),sizeof(uint32_t)*ecv.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(ecoDevice,eco.data(),sizeof(uint16_t)*eco.size(),cudaMemcpyHostToDevice))

    /* ISO */
    uint32_t* isovDevice;

    CUDAERRCHECK(cudaMalloc(&isovDevice,sizeof(uint32_t)*isov.size()))

    CUDAERRCHECK(cudaMemcpy(isovDevice,isov.data(),sizeof(uint32_t)*isov.size(),cudaMemcpyHostToDevice))

    /* LOOKUP TABLE */
    float* wmrSFPDevice;float* wmhSFPDevice;float* isoSFPDevice;

    CUDAERRCHECK(cudaMalloc(&wmrSFPDevice,sizeof(float)*wmrSFP.size()))
    CUDAERRCHECK(cudaMalloc(&wmhSFPDevice,sizeof(float)*wmhSFP.size()))
    CUDAERRCHECK(cudaMalloc(&isoSFPDevice,sizeof(float)*isoSFP.size()))

    CUDAERRCHECK(cudaMemcpy(wmrSFPDevice,wmrSFP.data(),sizeof(float)*wmrSFP.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(wmhSFPDevice,wmhSFP.data(),sizeof(float)*wmhSFP.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(isoSFPDevice,isoSFP.data(),sizeof(float)*isoSFP.size(),cudaMemcpyHostToDevice))

    /* HELPER ARRAYS */
    int* icIndexesDevice;int* ecIndexesDevice;int* isoIndexesDevice;

    CUDAERRCHECK(cudaMalloc(&icIndexesDevice,sizeof(int)*icIndexes.size()))
    CUDAERRCHECK(cudaMalloc(&ecIndexesDevice,sizeof(int)*ecIndexes.size()))
    CUDAERRCHECK(cudaMalloc(&isoIndexesDevice,sizeof(int)*isoIndexes.size()))

    CUDAERRCHECK(cudaMemcpy(icIndexesDevice,icIndexes.data(),sizeof(int)*icIndexes.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(ecIndexesDevice,ecIndexes.data(),sizeof(int)*ecIndexes.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(isoIndexesDevice,isoIndexes.data(),sizeof(int)*isoIndexes.size(),cudaMemcpyHostToDevice))

    /* INPUT */
    float* xDevice;
    
    CUDAERRCHECK(cudaMalloc(&xDevice,sizeof(float)*input.size()))

    CUDAERRCHECK(cudaMemcpy(xDevice,input.data(),sizeof(float)*input.size(),cudaMemcpyHostToDevice))

    /* RESULT */
    float* yDevice;

    CUDAERRCHECK(cudaMalloc(&yDevice,sizeof(float)*output.size()))
    
    CUDAERRCHECK(cudaMemset(yDevice,0.0f,sizeof(float)*output.size()))

    /* BLOCKS AND THREAD ORGANIZATION */
    dim3 dimGrid(BLOCKS,1,1);
    dim3 dimBlock(THREADS_PER_BLOCK,1,1);

    cudaEventRecord(kernelStart);

    icMatrixMultiplication<<<dimGrid,dimBlock>>>(_nS,_nV,icfDevice,icvDevice,icoDevice,iclDevice,_nR,_nF,wmrSFPDevice,_ndirs,icIndexesDevice,xDevice,yDevice);
    ecMatrixMultiplication<<<dimGrid,dimBlock>>>(_nS,_nV,_nR,_nF,ecvDevice,ecoDevice,_nT,_nE,wmhSFPDevice,_ndirs,ecIndexesDevice,xDevice,yDevice);
    //isoMatrixMultiplication<<<dimGrid,dimBlock>>>(_nS,_nV,_nR,_nF,_nE,_nT,isovDevice,_nI,isoSFPDevice,_ndirs,isoIndexesDevice,xDevice,yDevice);
    
    cudaEventRecord(kernelStop);

    /* COPYING BACK THE RESULT */
    std::vector<float> obtainedResult(output.size(),0.0f);
    CUDAERRCHECK(cudaMemcpy(obtainedResult.data(),yDevice,sizeof(float)*output.size(),cudaMemcpyDeviceToHost))

    /* FREEING MEMORY */
    cudaFree(icfDevice);cudaFree(icvDevice);cudaFree(icoDevice);cudaFree(iclDevice);
    cudaFree(ecvDevice);cudaFree(ecoDevice);
    cudaFree(isovDevice);
    cudaFree(wmrSFPDevice);cudaFree(wmhSFPDevice);cudaFree(isoSFPDevice);

    cudaEventRecord(totalStop);

    /* VERIFYING CORRECTNESS OF THE RESULT */
    cudaEventSynchronize(kernelStop);
    float kernelMilliseconds = 0;
    cudaEventElapsedTime(&kernelMilliseconds,kernelStart,kernelStop);
    
    cudaEventSynchronize(totalStop);
    float totalMilliseconds = 0;
    cudaEventElapsedTime(&totalMilliseconds,totalStart,totalStop);
    
    printResultGpu("Gpu matrix multiplication",output,obtainedResult,verifyCorrectnessGpu<float>(output,obtainedResult),kernelMilliseconds,totalMilliseconds);

    cudaDeviceReset();
}