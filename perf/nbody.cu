#include <stdio.h>
#include <chrono>

#define TESTS 1000
#define N 108 * 512 * 6
#define BLOCK_SIZE 512
#define GRID_SIZE 108

#include "step.cuh"

float randf()
{
  return (float)((double)rand() / RAND_MAX);
}

int main()
{
  statDevice();

  // Screen limits
  float width = 1920;
  float height = 1080;

  printf("Expected Memory usage: %lu MB\n", (sizeof(float) * N * 6) / 1024 / 1024);

  // Allocate px and py in host memory
  float2 *h_p;
  checkCudaErrors(cudaMallocHost(&h_p, sizeof(float2) * N));

  // Fill with random values in the screen
  for (int i = 0; i < N; i++)
  {
    h_p[i].x = randf() * width;
    h_p[i].y = randf() * height;
  }

  // Allocate vectors in device memory
  float2 *d_p, *d_v, *d_f;
  checkCudaErrors(cudaMalloc(&d_p, sizeof(float2) * N));
  checkCudaErrors(cudaMalloc(&d_v, sizeof(float2) * N));
  checkCudaErrors(cudaMalloc(&d_f, sizeof(float2) * N));

  // Copy host positions to device
  checkCudaErrors(cudaMemcpy(d_p, h_p, sizeof(float2) * N, cudaMemcpyHostToDevice));

  // Vectors can be initialized with zeros
  checkCudaErrors(cudaMemset(d_v, 0, sizeof(float2) * N));

  // Force vectors do not need to be initialized

  cudaStream_t compute_stream;
  // Allocate the stream
  checkCudaErrors(cudaStreamCreate(&compute_stream));

  auto start = std::chrono::high_resolution_clock::now();

  // Launch the kernel
  for (int i = 0; i < TESTS; i++)
  {
    step<BLOCK_SIZE, GRID_SIZE>(compute_stream, d_p, d_v, d_f);
  }

  // Wait for the stream to finish
  checkCudaErrors(cudaStreamSynchronize(compute_stream));

  auto end = std::chrono::high_resolution_clock::now();

  // Print the time
  printf("Average time (ms): %lf\n", (double)std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count() / 1000000. / TESTS);
  printf("Expected FPS: %lf\n", 1000000000. * TESTS / (double)std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count());

  // Free memory
  checkCudaErrors(cudaFree(d_p));
  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_f));
}