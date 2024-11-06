#include <iostream>
#include <cuda.h>

__global__ void kernelPrintf() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread ID: %d\n", idx);
}

int main() {
    int threadsPerBlock = 32;
    int blocksPerGrid = 8; // Total threads = 32 * 8 = 256

    kernelPrintf<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    return 0;
}
