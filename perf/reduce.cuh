#ifndef REDUCE_H
#define REDUCE_H

#include <stdio.h>
#include "helper_cuda.cuh"
#define FULL_MASK 0xffffffff

/// Add reduction taken from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__device__ void reduceWarp(float *sdata, const unsigned int tid)
{
  // No need for synchronization here, as we fit into a single warp
  if (blockSize >= 64)
  {
    sdata[tid] += sdata[tid + 32];
  }
  if (blockSize >= 32)
    sdata[tid] += __shfl_down_sync(FULL_MASK, sdata[tid], 16);
  if (blockSize >= 16)
    sdata[tid] += __shfl_down_sync(FULL_MASK, sdata[tid], 8);
  if (blockSize >= 8)
    sdata[tid] += __shfl_down_sync(FULL_MASK, sdata[tid], 4);
  if (blockSize >= 4)
    sdata[tid] += __shfl_down_sync(FULL_MASK, sdata[tid], 2);
  if (blockSize >= 2)
    sdata[tid] += __shfl_down_sync(FULL_MASK, sdata[tid], 1);
}

template <unsigned int blockSize>
__global__ void reduce(const float *g_idata, float *g_odata, const unsigned int n)
{
  __shared__ float sdata[blockSize];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  sdata[tid] = 0;
  while (i < n)
  {
    sdata[tid] += g_idata[i];
    if (i + blockSize < n)
    {
      sdata[tid] += g_idata[i + blockSize];
    }
    i += gridSize;
  }

  __syncthreads();

  if (blockSize >= 1024)
  {
    if (tid < 512)
    {
      sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
  }

  if (blockSize >= 512)
  {
    if (tid < 256)
    {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }

  if (blockSize >= 256)
  {
    if (tid < 128)
    {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }

  if (blockSize >= 128)
  {
    if (tid < 64)
    {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32)
  {
    reduceWarp<blockSize>(sdata, tid);
  }

  if (tid == 0)
  {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template <unsigned int blockSize>
float dispatch_reduce(float *d_idata, float *d_odata, unsigned int size)
{
  // amount of shared memory needed for the block
  unsigned long sharedMemSize = blockSize * sizeof(float);

  // Size of the grid in blocks
  unsigned long gridSize = (size + blockSize * 2 - 1) / (blockSize * 2);

  // Invoke kernel
  reduce<blockSize><<<gridSize, blockSize, sharedMemSize>>>(d_idata, d_odata, size);

  getLastCudaError("First Reduction execution failed");

  float result = 0.;
  // Reduce again to get the final result
  if (gridSize > 1)
  {
    reduce<blockSize><<<1, blockSize, sharedMemSize>>>(d_odata, d_odata, gridSize);

    getLastCudaError("Second reduction execution failed");

    checkCudaErrors(cudaMemcpy(&result, d_odata, sizeof(float), cudaMemcpyDeviceToHost));
  }

  return result;
}

#endif