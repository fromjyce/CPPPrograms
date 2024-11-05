#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolve1D(float *input, float *output, float *kernel, int inputSize, int kernelSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfKernel = kernelSize / 2;
    if (idx >= inputSize) return;

    float sum = 0;
    for (int j = -halfKernel; j <= halfKernel; ++j) {
        int index = idx + j;
        if (index >= 0 && index < inputSize)
            sum += input[index] * kernel[halfKernel + j];
    }
    output[idx] = sum;
}

void testConfiguration(int N, int threads, int kernelSize) {
    printf("Running with N = %d, Threads per block = %d\n", N, threads);

    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float h_kernel[] = {0.2, 0.2, 0.2, 0.2, 0.2};

    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    convolve1D<<<numBlocks, threads>>>(d_input, d_output, d_kernel, N, kernelSize);
    cudaDeviceSynchronize();


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for N = %d, Threads = %d: %f ms\n", N, threads, milliseconds);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_input);
    free(h_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int kernelSize = 5;
    int Ns[] = {1024, 2048, 8192}; // Different input sizes
    int threads[] = {512, 1024, 2048}; // Different thread configurations

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            testConfiguration(Ns[i], threads[j], kernelSize);
        }
    }

    return 0;
}
