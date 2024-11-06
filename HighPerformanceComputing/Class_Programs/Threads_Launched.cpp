#include <iostream>
#include <cuda.h>

__global__ void kernelComputeIndex() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread ID: %d\n", idx);
}

int main() {
    int threadsPerBlock = 16;
    int blocksPerGrid = 4;

    kernelComputeIndex<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    std::cout << "Total threads launched: " << blocksPerGrid * threadsPerBlock << std::endl;

    return 0;
}
