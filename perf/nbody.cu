#include <stdio.h>
#include <chrono>

#define TESTS 1000
#define N 108 * 512 * 6
#define BLOCK_SIZE 512
#define GRID_SIZE 108

#include "step.hu"

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
  float *h_px, *h_py;
  checkCudaErrors(cudaMallocHost(&h_px, sizeof(float) * N));
  checkCudaErrors(cudaMallocHost(&h_py, sizeof(float) * N));

  // Fill with random values in the screen
  for (int i = 0; i < N; i++)
  {
    h_px[i] = randf() * width;
    h_py[i] = randf() * height;
  }

  // Allocate vectors in device memory
  float *d_px, *d_py, *d_vx, *d_vy, *d_fx, *d_fy;
  checkCudaErrors(cudaMalloc(&d_px, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_py, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_vx, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_vy, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_fx, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_fy, sizeof(float) * N));

  // Copy host positions to device
  checkCudaErrors(cudaMemcpy(d_px, h_px, sizeof(float) * N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_py, h_py, sizeof(float) * N, cudaMemcpyHostToDevice));

  // Vectors can be initialized with zeros
  checkCudaErrors(cudaMemset(d_vx, 0, sizeof(float) * N));
  checkCudaErrors(cudaMemset(d_vy, 0, sizeof(float) * N));

  // Force vectors do not need to be initialized

  cudaStream_t compute_stream;
  // Allocate the stream
  checkCudaErrors(cudaStreamCreate(&compute_stream));

  auto start = std::chrono::high_resolution_clock::now();

  // Launch the kernel
  for (int i = 0; i < TESTS; i++)
  {
    step<BLOCK_SIZE, GRID_SIZE>(compute_stream, d_px, d_py, d_vx, d_vy, d_fx, d_fy);
  }

  // Wait for the stream to finish
  checkCudaErrors(cudaStreamSynchronize(compute_stream));

  auto end = std::chrono::high_resolution_clock::now();

  // Print the time
  printf("Average time (ms): %lf\n", (double)std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count() / 1000000. / TESTS);
  printf("Expected FPS: %lf\n", 1000000000. * TESTS / (double)std::chrono::duration_cast<std::chrono::nanoseconds>((end - start)).count());

  // Free memory
  checkCudaErrors(cudaFree(d_px));
  checkCudaErrors(cudaFree(d_py));
  checkCudaErrors(cudaFree(d_vx));
  checkCudaErrors(cudaFree(d_vy));
  checkCudaErrors(cudaFree(d_fx));
  checkCudaErrors(cudaFree(d_fy));
}