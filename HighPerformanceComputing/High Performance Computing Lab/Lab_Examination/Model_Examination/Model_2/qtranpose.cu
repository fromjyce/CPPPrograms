#include <stdio.h>
#include <ctime>

#define N 1024
#define DIM 32

__global__ void transpose(int *org, int *trans) {
    __shared__ int shared[DIM][DIM];
    int x = blockIdx.x * DIM + threadIdx.x;
    int y = blockIdx.y * DIM + threadIdx.y;
    if (x < N && y < N) {
        shared[threadIdx.y][threadIdx.x] = org[y * N + x];
    }
    __syncthreads();
    if (x < N && y < N) {
        trans[x * N + y] = shared[threadIdx.x][threadIdx.y];
    }
}

void transposeCPU(int *org, int *trans) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            trans[j * N + i] = org[i * N + j];
        }
    }
}

int main() {
    int *h_org, *h_trans;
    int *d_org, *d_trans;

    h_org = (int*)malloc(N * N * sizeof(int));
    h_trans = (int*)malloc(N * N * sizeof(int));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_org[i * N + j] = i * N + j;
        }
    }
    cudaMalloc((void**)&d_org, N * N * sizeof(int));
    cudaMalloc((void**)&d_trans, N * N * sizeof(int));
    cudaMemcpy(d_org, h_org, N * N * sizeof(int), cudaMemcpyHostToDevice);
    int block_size = 32;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    transpose<<<gridDim, blockDim>>>(d_org, d_trans);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Time for N=%d, Threads=%d: %.2f ms\n", N, block_size * block_size, milliseconds);


    cudaMemcpy(h_trans, d_trans, N * N * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_org);
    cudaFree(d_trans);


    clock_t cpu_start = clock();
    transposeCPU(h_org, h_trans);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Time for N=%d: %.2f ms\n", N, cpu_time);


    printf("Original Matrix (5x5 subset):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d ", h_org[i * N + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Matrix (5x5 subset):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d ", h_trans[i * N + j]);
        }
        printf("\n");
    }


    free(h_org);
    free(h_trans);

    return 0;
}
