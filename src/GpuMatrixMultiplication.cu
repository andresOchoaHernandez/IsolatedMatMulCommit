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

#define WMRSFP_SIZE 16000

__constant__ float WMRSFP[WMRSFP_SIZE];

__global__ void commitMatrixMultiplication(
    int nS,
    int nV,
    uint32_t* icfDevice, uint32_t* icvDevice, uint16_t* icoDevice, float* iclDevice, int nR,int nF,
    uint32_t* ecvDevice, uint16_t* ecoDevice, int nT,int nE,
    uint32_t* isovDevice, int nI,
    cudaTextureObject_t WMHSFP, cudaTextureObject_t ISOSFP,int ndirs,
    int* icIndexesDevice, int* ecIndexesDevice,
    float* xDevice,
    float* yDevice
)
{
    /* SHARED MEMORY BUFFERS */
    extern __shared__ float buffer[];
    int* fibers = (int*)buffer;
    int* orientations =(int*)&buffer[nS];
    float* lengths = &buffer[2*nS];

    /* MANDATORY INDEXES */
    const int voxel  = blockIdx.x;
    const int sample = threadIdx.x;
    const int yIndex = voxel * nS + sample;

    /* IC INDEXES */
    const int startIcSegment = (voxel==0)?0:icIndexesDevice[voxel-1];
    const int endIcSegment   = icIndexesDevice[voxel];
    const int totalIcSegments  = endIcSegment - startIcSegment;

    const int TOTAL_IC_TILES = 1 + ((totalIcSegments-1)/nS);
    const int lastIcSegment = startIcSegment + (TOTAL_IC_TILES-1) * nS + nS; 
    const int diff = lastIcSegment - endIcSegment;

    /* EC INDEXES */
    const int startEcSegment = (voxel==0)?0:ecIndexesDevice[voxel-1];
    const int endEcSegment   = ecIndexesDevice[voxel];

    /* MULTIPLICATION */
    float accumulator = 0.0f;

    for(int tile = 0; tile < TOTAL_IC_TILES; tile++)
    {
        int segmentIndex = startIcSegment + tile * nS + threadIdx.x;

        if(segmentIndex < endIcSegment)
        {
            fibers[threadIdx.x]       = icfDevice[segmentIndex];
            orientations[threadIdx.x] = icoDevice[segmentIndex];
            lengths[threadIdx.x]      = iclDevice[segmentIndex];
        }
        __syncthreads();

        for (int radii = 0; radii < nR; radii++)
        {
            for(int icsegment = 0; icsegment < nS-diff*static_cast<int>(tile == TOTAL_IC_TILES-1); icsegment++)
            {
                int xIndex = fibers[icsegment] + nF*radii;
                int lookupTableIndex = radii*ndirs*nS + orientations[icsegment] * nS + sample;

                accumulator += xDevice[xIndex]*WMRSFP[lookupTableIndex]*lengths[icsegment];
            }
        }
        __syncthreads();
    }
    for (int tortuosity = 0; tortuosity < nT; tortuosity++)
    {
        int xIndex = nR*nF + tortuosity*nE + startEcSegment;

        for(int ecsegment = startEcSegment; ecsegment < endEcSegment; ecsegment++)
        {
            int lookupTableIndex = tortuosity*ndirs*nS + ecoDevice[ecsegment] * nS + sample;

            accumulator += xDevice[xIndex]*tex1Dfetch<float>(WMHSFP,lookupTableIndex);
            xIndex++;
        }
    }
    for (int iso = 0; iso < nI; iso++)
    {   
        int xIndex = nR*nF + nT*nE + voxel + iso*nV;
        int lookupTableIndex = iso * nS + sample;

        accumulator += xDevice[xIndex]*tex1Dfetch<float>(ISOSFP,lookupTableIndex);
    }

    /* FINAL RESULT */
    yDevice[yIndex] = accumulator;
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

    /* IC */
    CUDAERRCHECK( cudaMemcpyToSymbol(WMRSFP,wmrSFP.data(),sizeof(float)*wmrSFP.size(), 0, cudaMemcpyHostToDevice) );

    /* EC & ISO SECTIONS */
    float* wmhSFPDevice;float* isoSFPDevice;

    CUDAERRCHECK(cudaMalloc(&wmhSFPDevice,sizeof(float)*wmhSFP.size()))
    CUDAERRCHECK(cudaMalloc(&isoSFPDevice,sizeof(float)*isoSFP.size()))

    CUDAERRCHECK(cudaMemcpy(wmhSFPDevice,wmhSFP.data(),sizeof(float)*wmhSFP.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(isoSFPDevice,isoSFP.data(),sizeof(float)*isoSFP.size(),cudaMemcpyHostToDevice))

    /* TEXTURE OBJECTS */

    /* EC SECTION LOOKUPTABLE */
    cudaResourceDesc resDescWMHSFP;
    memset(&resDescWMHSFP,0,sizeof(resDescWMHSFP));
    resDescWMHSFP.resType = cudaResourceTypeLinear;
    resDescWMHSFP.res.linear.devPtr = wmhSFPDevice;
    resDescWMHSFP.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDescWMHSFP.res.linear.desc.x = 32;
    resDescWMHSFP.res.linear.sizeInBytes = sizeof(float)*wmhSFP.size();

    cudaTextureDesc texDescWMHSFP;
    memset(&texDescWMHSFP, 0, sizeof(texDescWMHSFP));
    texDescWMHSFP.readMode = cudaReadModeElementType;

    cudaTextureObject_t WMHSFP=0;
    cudaCreateTextureObject(&WMHSFP, &resDescWMHSFP, &texDescWMHSFP, NULL);

    /* ISO SECTION LOOKUPTABLE */
    cudaResourceDesc resDescISOSFP;
    memset(&resDescISOSFP,0,sizeof(resDescISOSFP));
    resDescISOSFP.resType = cudaResourceTypeLinear;
    resDescISOSFP.res.linear.devPtr = isoSFPDevice;
    resDescISOSFP.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDescISOSFP.res.linear.desc.x = 32;
    resDescISOSFP.res.linear.sizeInBytes = sizeof(float)*isoSFP.size();

    cudaTextureDesc texDescISOSFP;
    memset(&texDescISOSFP, 0, sizeof(texDescISOSFP));
    texDescISOSFP.readMode = cudaReadModeElementType;

    cudaTextureObject_t ISOSFP=0;
    cudaCreateTextureObject(&ISOSFP, &resDescISOSFP, &texDescISOSFP, NULL);

    /* HELPER ARRAYS */
    int* icIndexesDevice;int* ecIndexesDevice;

    CUDAERRCHECK(cudaMalloc(&icIndexesDevice,sizeof(int)*icIndexes.size()))
    CUDAERRCHECK(cudaMalloc(&ecIndexesDevice,sizeof(int)*ecIndexes.size()))

    CUDAERRCHECK(cudaMemcpy(icIndexesDevice,icIndexes.data(),sizeof(int)*icIndexes.size(),cudaMemcpyHostToDevice))
    CUDAERRCHECK(cudaMemcpy(ecIndexesDevice,ecIndexes.data(),sizeof(int)*ecIndexes.size(),cudaMemcpyHostToDevice))

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
    commitMatrixMultiplication<<<dimGrid,dimBlock,3*_nS*sizeof(float)>>>(
        _nS,
        _nV,
        icfDevice,icvDevice,icoDevice,iclDevice,_nR,_nF,
        ecvDevice,ecoDevice,_nT,_nE,
        isovDevice,_nI,
        WMHSFP,ISOSFP,_ndirs,
        icIndexesDevice,ecIndexesDevice,
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
    cudaFree(wmhSFPDevice);cudaFree(isoSFPDevice);
    cudaFree(icIndexesDevice);cudaFree(ecIndexesDevice);

    cudaDestroyTextureObject(WMHSFP);cudaDestroyTextureObject(ISOSFP);

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