#include <iostream>
#include <cuda.h>

__global__ void warpReadDemo(int *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 32) {  // Only the first warp (32 threads) accesses memory
        printf("Thread %d reads value: %d\n", idx, array[idx]);
    }
}

int main() {
    int h_array[32];
    for (int i = 0; i < 32; i++) h_array[i] = i * 10;

    int *d_array;
    cudaMalloc((void**)&d_array, 32 * sizeof(int));
    cudaMemcpy(d_array, h_array, 32 * sizeof(int), cudaMemcpyHostToDevice);

    warpReadDemo<<<1, 32>>>(d_array); // Launch with 32 threads in a single block
    cudaDeviceSynchronize();

    cudaFree(d_array);
    return 0;
}
