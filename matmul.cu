#include <cstdio>
#include "lib/helper_cuda.cuh"
#define N 10000

// Kernel definition
__global__ void MatAdd(float *A, float *B,
                       float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
    {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

int main()
{
    unsigned int size = N * N;
    unsigned int mem_size = sizeof(float) * size;

    // Allocate input vectors h_A and h_B in host memory
    float *h_A, *h_B, *h_C;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size));
    checkCudaErrors(cudaMallocHost(&h_B, mem_size));
    checkCudaErrors(cudaMallocHost(&h_C, mem_size));

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    float count = 0;

    // Initialize input matrices
    for (int i = 0; i < size; i++)
    {
        h_A[i] = count++;
        h_B[i] = count++;
    }

    // Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, mem_size));
    checkCudaErrors(cudaMalloc(&d_B, mem_size));
    checkCudaErrors(cudaMalloc(&d_C, mem_size));

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Copy host vectors to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice));

    // Invoke kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    getLastCudaError("Kernel execution failed");

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost));

    // Verify result
    for (int i = 0; i < size; i++)
    {
        if (h_A[i] + h_B[i] != h_C[i])
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}