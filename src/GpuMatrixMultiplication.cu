#include "CommitOriginalDataStructure.hpp"

#include <iomanip>

/* ========================================================================================== */
template<typename T>
bool areNearlyEqual(T a, T b) {
    const T normal_min = std::numeric_limits<T>::min();
    const T relative_error = 0.000009;
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

void printResult(const std::string& message,bool correctness,float kernelTime,float totalTime){

    const std::string upperSepSx  = "------------------ ";
    const std::string upperSepDx  = " ------------------";
    const std::string downerSep(upperSepSx.length()*2+message.length(),'-'); 

    std::cout << upperSepSx << message << upperSepDx                       << std::endl
              << "| correct        => " << ((correctness)? "true":"false") << std::endl
              << "| kernel time    => " << kernelTime << " ms"             << std::endl
              << "| total  time    => " << totalTime  << " ms"             << std::endl
              << downerSep                                                 << std::endl;  

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

__global__ void commitMatrixMultiplication(
    const uint32_t* icfDevice, const uint32_t* icvDevice, const uint16_t* icoDevice, const float* iclDevice, int nR,
    const uint32_t* ecvDevice, const uint16_t* ecoDevice, int nT,
    const uint32_t* isovDevice, int nI,
    const float* wmrSFPDevice, const float* wmhSFPDevice, const float* isoSFPDevice,int ndirs,
    const int* icIndexesDevice, const int* ecIndexesDevice, const int* isoIndexesDevice,
    float* xDevice,
    float* yDevice
)
{
    const int voxel       = blockIdx.x * blockDim.x;
    const int voxelOffset = voxel + threadIdx.x;

    /* IC */
    __shared__ float acc[100];
    acc[ threadIdx.x ] = 0.0f;

    for(int icsegment = voxel==0?0:icIndexesDevice[voxel-1]; icsegment < icIndexesDevice[voxel]; icsegment++)
    {
        for(int radii = 0; radii < nR; radii++)
        {
            acc[threadIdx.x] += 
                iclDevice[icsegment] * 
                wmrSFPDevice[(radii*ndirs * blockDim.x) + (icoDevice[icsegment]* blockDim.x + threadIdx.x)] * 
                xDevice[icfDevice[icsegment] + radii];
        }
    }
    yDevice[voxelOffset] += acc[threadIdx.x]; // TODO: IDEA, COMPUTAZIONI NEL CICLO IN SHARED MEMORY, 1 SOLA SCRITTURA IN GLOBAL DOPO
    /* EC */
    for(int ecsegment = voxel==0?0:ecIndexesDevice[voxel-1]; ecsegment < ecIndexesDevice[voxel]; ecsegment++)
    {
        for(int tortuosity = 0; tortuosity < nT; tortuosity++)
        {
        }
    }
    /* ISO */ 
    for(int isosegment = voxel==0?0:isoIndexesDevice[voxel-1]; isosegment < isoIndexesDevice[voxel]; isosegment++)
    {
        for(int iso = 0; iso < nI; iso++)
        {
        }
    }
}

void CommitOriginalDataStructure::gpuMatrixMultiplication()
{
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
    const int blocks = _nV;
    const int threadsPerBlock = _nS;

    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(threadsPerBlock,1,1);

    cudaEventRecord(kernelStart);
    commitMatrixMultiplication<<<dimGrid,dimBlock>>>(
        icfDevice,icvDevice,icoDevice,iclDevice,_nR,
        ecvDevice,ecoDevice,_nT,
        isovDevice,_nI,
        wmrSFPDevice,wmhSFPDevice,isoSFPDevice,_ndirs,
        icIndexesDevice,ecIndexesDevice,isoIndexesDevice,
        xDevice,
        yDevice
    );
    cudaEventRecord(kernelStop);

    /* COPYING BACK THE RESULT */
    std::vector<float> obtainedResult(output.size(),0.0f);
    CUDAERRCHECK(cudaMemcpy(obtainedResult.data(),yDevice,sizeof(float)*output.size(),cudaMemcpyDeviceToHost))

    /* FREEING MEMORY */
    cudaFree(icfDevice);cudaFree(icvDevice);cudaFree(icoDevice);cudaFree(iclDevice);
    cudaFree(ecvDevice);cudaFree(ecoDevice);
    cudaFree(isovDevice);
    cudaFree(wmrSFPDevice);cudaFree(wmhSFPDevice);cudaFree(isoSFPDevice);
    cudaFree(icIndexesDevice);cudaFree(ecIndexesDevice);cudaFree(isoIndexesDevice);

    cudaEventRecord(totalStop);

    /* VERIFYING CORRECTNESS OF THE RESULT */
    cudaEventSynchronize(kernelStop);
    float kernelMilliseconds = 0;
    cudaEventElapsedTime(&kernelMilliseconds,kernelStart,kernelStop);
    
    cudaEventSynchronize(totalStop);
    float totalMilliseconds = 0;
    cudaEventElapsedTime(&totalMilliseconds,totalStart,totalStop);
    
    printResult("Gpu matrix multiplication", verifyCorrectness<float>(output,obtainedResult),kernelMilliseconds,totalMilliseconds);

    cudaDeviceReset();
}