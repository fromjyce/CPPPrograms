#include <iostream>
#include <cuda.h>

#define N 256
__global__ void initializeToZero(int *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        array[idx] = 0;
    }
}


__global__ void addIndexToArray(int *array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        array[idx] += idx;
    }
}

int main() {
    int *d_array;
    int size = N * sizeof(int);
    cudaMalloc((void **)&d_array, size);
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    initializeToZero<<<blocksPerGrid, threadsPerBlock>>>(d_array);
    cudaDeviceSynchronize();

    addIndexToArray<<<blocksPerGrid, threadsPerBlock>>>(d_array);
    cudaDeviceSynchronize();

    int h_array[N];
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    std::cout << "Array after adding index values:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_array);

    return 0;
}
