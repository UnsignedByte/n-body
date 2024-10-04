#include <cstdio>
#include <cassert>
#include "helper_cuda.cuh"
#include "reduce.cuh"
#include <chrono>
#define TESTS 10000
#define N 1000000
#define EPSILON 1e-5
#define BLOCK_SIZE 256

int main()
{
  statDevice();

  // Allocate the array
  float *h_data;
  checkCudaErrors(cudaMallocHost(&h_data, N * sizeof(float)));
  for (int i = 0; i < N; i++)
  {
    // Make the data non-uniform
    h_data[i] = (i % 10) + 1;
  }

  // The sum of this arry should be N * 11/2 + (N % 10) * (N % 10 + 1) / 2
  float expected = 0.;
  for (int i = 0; i < N; i++)
  {
    expected += h_data[i];
  }

  // Size of the grid in blocks
  const unsigned long gridSize = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

  // Allocate device memory
  float *d_data, *d_result;
  checkCudaErrors(cudaMalloc(&d_data, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_result, gridSize * sizeof(float)));

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  checkCudaErrors(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  for (int i = 0; i < TESTS; ++i)
  {

    float result = dispatch_reduce<BLOCK_SIZE>(d_data, d_result, N);

    if (abs(result - expected) > EPSILON)
    {
      printf("Test failed at iteration %d: %f != %f\n", i, result, expected);
      return EXIT_FAILURE;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  // Print the time
  printf("Average time (ms): %lf\n", (double)std::chrono::duration_cast<std::chrono::nanoseconds>((end - start) / TESTS).count() / 1000000.);

  printf("Test passed\n");

  // Free memory
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_result));
  checkCudaErrors(cudaFreeHost(h_data));
}