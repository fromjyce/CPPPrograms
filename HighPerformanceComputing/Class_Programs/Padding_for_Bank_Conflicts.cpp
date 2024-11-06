#include <iostream>
#include <cuda.h>

__global__ void preventBankConflicts() {
    __shared__ int array[32 + 1]; // Adding padding to avoid conflicts

    int idx = threadIdx.x;
    array[idx] = idx;  // Each thread writes its index value to shared memory
    __syncthreads();

    printf("Thread %d reads array[%d] = %d\n", idx, idx, array[idx]);
}

int main() {
    preventBankConflicts<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}
