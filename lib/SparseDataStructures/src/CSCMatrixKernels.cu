__global__ void cscMatrixVectorMultKernel(const unsigned* cscCols,const unsigned* cscRows, const int* cscVals, const int* v1, int* rv,const unsigned cols)
{
    // TODO: what about race conditions?
}