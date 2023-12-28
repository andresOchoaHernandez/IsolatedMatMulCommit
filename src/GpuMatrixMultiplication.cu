#include "CommitOriginalDataStructure.hpp"

#include <iomanip>
#include <cassert>

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


struct LUTBatchDevice{
    float* wmrSFP;
    float* wmhSFP;
    float* isoSFP;
};

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
    unsigned nR,unsigned ndirs, LUTBatchDevice lutBatchesDevice,
    float* yDevice
)
{
    /* SHARED MEMORY BUFFERS */
    extern __shared__ float LUTBuffer[];

    /* IC SECTION */
    for(unsigned radii = 0 ; radii < nR ; radii++)
    {
        for(unsigned direction = 0 ; direction < ndirs ; direction++)
        {
            LUTBuffer[(radii*ndirs*SAMPLE_TILE_LENGTH) + (direction*SAMPLE_TILE_LENGTH + threadIdx.x)] = lutBatchesDevice.wmrSFP[(radii*ndirs*SAMPLE_TILE_LENGTH) + (direction*SAMPLE_TILE_LENGTH + threadIdx.x)];
        }
    }
    __syncthreads();
}

cudaDeviceProp getCudaDeviceProps(int deviceId){

  cudaDeviceProp deviceProps;

  cudaError_t cu_err = cudaGetDeviceProperties(&deviceProps, deviceId);
  if(cudaSuccess != cu_err){
    printf("Unable to get cudaGetDeviceProperties for device ID %d : error num %d - %s\n", deviceId, (int) cu_err, cudaGetErrorString(cu_err));
    exit(EXIT_FAILURE);
  }

  return deviceProps;
}

void CommitOriginalDataStructure::gpuMatrixMultiplication()
{

    /* CHECK IF SHARED MEMORY PER BLOCK IS ENOUGH TO FIT LUT IN IT (ASSUMING THE MACHINE IS EQUIPPED WITH ONLY ONE GPU)*/
    cudaDeviceProp deviceProps;
    CUDAERRCHECK(cudaGetDeviceProperties(&deviceProps, 0))
    assert(_nR*_ndirs*SAMPLE_TILE_LENGTH*sizeof(float) < deviceProps.sharedMemPerBlock);

    cudaEvent_t totalStart,totalStop;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);

    cudaEvent_t kernelStart,kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(totalStart);

    /* LUTS BATCHES ALLOCATION & COPY TO GLOBAL MEMORY */
    LUTBatchDevice* lutBatchesDevice;
    CUDAERRCHECK(cudaMallocManaged((void **)&lutBatchesDevice, sizeof(LUTBatchDevice)*batchedLUTs.size()))

    for(int sampleTile = 0 ; sampleTile < batchedLUTs.size() ; sampleTile++)
    {
        float* wmrSFPDevice;
        CUDAERRCHECK(cudaMalloc(&wmrSFPDevice,sizeof(float)*batchedLUTs[sampleTile].wmrSFP.size()))
        CUDAERRCHECK(cudaMemcpy(wmrSFPDevice,batchedLUTs[sampleTile].wmrSFP.data(),sizeof(float)*batchedLUTs[sampleTile].wmrSFP.size(),cudaMemcpyHostToDevice))

        float* wmhSFPDevice;
        CUDAERRCHECK(cudaMalloc(&wmhSFPDevice,sizeof(float)*batchedLUTs[sampleTile].wmhSFP.size()))
        CUDAERRCHECK(cudaMemcpy(wmhSFPDevice,batchedLUTs[sampleTile].wmhSFP.data(),sizeof(float)*batchedLUTs[sampleTile].wmhSFP.size(),cudaMemcpyHostToDevice))

        float* isoSFPDevice;
        CUDAERRCHECK(cudaMalloc(&isoSFPDevice,sizeof(float)*batchedLUTs[sampleTile].isoSFP.size()))
        CUDAERRCHECK(cudaMemcpy(isoSFPDevice,batchedLUTs[sampleTile].isoSFP.data(),sizeof(float)*batchedLUTs[sampleTile].isoSFP.size(),cudaMemcpyHostToDevice))

        lutBatchesDevice[sampleTile].wmrSFP = wmrSFPDevice;
        lutBatchesDevice[sampleTile].wmhSFP = wmhSFPDevice; 
        lutBatchesDevice[sampleTile].isoSFP = isoSFPDevice; 
    }


    /* RESULT */
    float* yDevice;
    CUDAERRCHECK(cudaMalloc(&yDevice,sizeof(float)*output.size()))
    CUDAERRCHECK(cudaMemset(yDevice,0.0f,sizeof(float)*output.size()))

    /* BLOCKS AND THREAD ORGANIZATION */
    const int blocks = _nV;
    const int threadsPerBlock = SAMPLE_TILE_LENGTH;

    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(threadsPerBlock,1,1);

    cudaEventRecord(kernelStart);

    for(unsigned sampleTile = 0 ; sampleTile < batchedLUTs.size() ; sampleTile++)
    {
        commitMatrixMultiplication<<<dimGrid,dimBlock,_nR*_ndirs*SAMPLE_TILE_LENGTH*sizeof(float)>>>(
            _nR,_ndirs,lutBatchesDevice[sampleTile],
            yDevice
        );
    }

    cudaEventRecord(kernelStop);

    /* COPYING BACK THE RESULT */
    std::vector<float> obtainedResult(output.size(),0.0f);
    CUDAERRCHECK(cudaMemcpy(obtainedResult.data(),yDevice,sizeof(float)*output.size(),cudaMemcpyDeviceToHost))

    /* FREEING MEMORY */
    for(int sampleTile = 0 ; sampleTile < batchedLUTs.size() ; sampleTile++)
    {
        CUDAERRCHECK(cudaFree(lutBatchesDevice[sampleTile].wmrSFP))
        CUDAERRCHECK(cudaFree(lutBatchesDevice[sampleTile].wmhSFP))
        CUDAERRCHECK(cudaFree(lutBatchesDevice[sampleTile].isoSFP))
    }
    CUDAERRCHECK(cudaFree(lutBatchesDevice))
    CUDAERRCHECK(cudaFree(yDevice))

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