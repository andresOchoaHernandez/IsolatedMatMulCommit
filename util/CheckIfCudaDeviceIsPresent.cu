int main(void)
{
    int deviceCount;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    return e == cudaSuccess ? deviceCount : -1;
}