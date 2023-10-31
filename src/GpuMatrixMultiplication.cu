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
*/

__global__ void commitMatrixMultiplication(
    uint32_t* icfDevice, float* iclDevice,
    int* icIndexesDevice,
    float* xDevice,
    float* yDevice
)
{
    /* SHARED MEMORY BUFFERS */
    extern __shared__ float buffer[];
    int* xBuffer = (int*)buffer;
    float* lengthsBuffer = &buffer[32];

    const int voxel = blockIdx.x;

    const int startIcSegment = (voxel == 0) ? 0 : icIndexesDevice[voxel-1];
    const int endIcSegment   = icIndexesDevice[voxel];
    const int totalIcSegments  = endIcSegment - startIcSegment;

    const int TOTAL_IC_TILES = 1 + ((totalIcSegments-1)/32);

    float result = 0.0f;

    for(int tile = 0; tile < TOTAL_IC_TILES; tile++)
    {
        int segmentIndex = startIcSegment + tile * 32 + threadIdx.x;


        /*==================================================================*/
        /*                 PART TO OPTIMIZE                                 */

        if(segmentIndex < endIcSegment)
        {
            xBuffer[threadIdx.x]  = xDevice[icfDevice[segmentIndex]];
            lengthsBuffer[threadIdx.x] = iclDevice[segmentIndex];
        }

        /* CALCULATING MULTIPLICATION */
        float accumulator = 0.0f;
        
        if(segmentIndex < endIcSegment)
        {
            accumulator = xBuffer[threadIdx.x] * lengthsBuffer[threadIdx.x];
        }
        /*==================================================================*/

        /* REDUCTION */
        for (int offset = 16; offset > 0; offset /= 2)
            accumulator += __shfl_down_sync(0xffffffff, accumulator, offset);
        
        result += accumulator;
    }

    /* WRITING OUT THE RESULT */
    if(threadIdx.x == 0)
    {
        yDevice[voxel] = result;
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
    uint32_t* icfDevice; float* iclDevice;

    CUDAERRCHECK(cudaMalloc(&icfDevice,sizeof(uint32_t)*icf.size()))
    CUDAERRCHECK(cudaMalloc(&iclDevice,sizeof(float)*icl.size()))

    CUDAERRCHECK(cudaMemcpy(icfDevice,icf.data(),sizeof(uint32_t)*icf.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(iclDevice,icl.data(),sizeof(float)*icl.size(),cudaMemcpyHostToDevice))

    /* HELPER INDEXES */
    int* icIndexesDevice;

    CUDAERRCHECK(cudaMalloc(&icIndexesDevice,sizeof(int)*icIndexes.size()))
    CUDAERRCHECK(cudaMemcpy(icIndexesDevice,icIndexes.data(),sizeof(int)*icIndexes.size(),cudaMemcpyHostToDevice))

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
    const int threadsPerBlock = 32;

    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(threadsPerBlock,1,1);

    cudaEventRecord(kernelStart);
    commitMatrixMultiplication<<<dimGrid,dimBlock,2*32*sizeof(float)>>>(
        icfDevice,iclDevice,
        icIndexesDevice,
        xDevice,
        yDevice
    );
    cudaEventRecord(kernelStop);

    /* COPYING BACK THE RESULT */
    std::vector<float> obtainedResult(output.size(),0.0f);
    CUDAERRCHECK(cudaMemcpy(obtainedResult.data(),yDevice,sizeof(float)*output.size(),cudaMemcpyDeviceToHost))

    /* FREEING MEMORY */
    cudaFree(icfDevice);cudaFree(iclDevice);
    cudaFree(icIndexesDevice);

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