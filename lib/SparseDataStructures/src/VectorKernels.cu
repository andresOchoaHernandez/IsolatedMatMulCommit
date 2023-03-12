__global__ void vectorDifKernel(const float* v1, const float* v2, float* rv, const unsigned size)
{
    const unsigned globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(globalIndex >= size) return;

    rv[globalIndex] = v1[globalIndex] - v2[globalIndex];
}