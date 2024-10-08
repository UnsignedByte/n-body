#ifndef STEP_H
#define STEP_H

#include <stdio.h>
#include "helper_cuda.cuh"
#define FULL_MASK 0xffffffff
#ifndef GRAVITY
#define GRAVITY 0.00001f
#endif
#ifndef SOFTENING
#define SOFTENING 0.01f
#endif
#ifndef DELTA_T
#define DELTA_T 0.01f
#endif
#ifndef N
#define N 1024 * 16
#endif

template <unsigned int blockSize>
__global__ void nbody_forces(const float2 *__restrict__ p, float2 *__restrict__ f)
{
  const unsigned int tid = threadIdx.x;
  const unsigned int laneid = tid & 31;

  for (unsigned int i = blockIdx.x * blockSize + tid; i < N; i += gridDim.x * blockSize)
  {

    // Register accumulators
    float fx_acc = 0.0f, fy_acc = 0.0f;

    // Move target data to registers
    float x = p[i].x;
    float y = p[i].y;

    // Iterate one block at a time to allow for shared memory coalescing
    for (unsigned int block_i = 0; block_i < N; block_i += blockSize)
    {
      // Load shared memory
      const float2 my_p = p[block_i + laneid];

#pragma unroll
      // Compute forces in the wrap
      for (unsigned int j = 0; j < 32; j++)
      {
        long long np_l = __shfl_sync(FULL_MASK, *reinterpret_cast<const long long *>(&my_p), j);
        float2 np = *reinterpret_cast<const float2 *>(&np_l);
        float dx = np.x - x;
        float dy = np.y - y;
        float distsq = dx * dx + dy * dy;
        distsq += SOFTENING;
        float inv_dist = rsqrtf(distsq);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;

        float fmag = inv_dist3 * GRAVITY;

        fx_acc += dx * fmag;
        fy_acc += dy * fmag;
      }
    }
    // Move accumulators to global memory

    f[i].x = fx_acc;
    f[i].y = fy_acc;
  }
}

__global__ void update_pos(float2 *__restrict__ p, float2 *__restrict__ v, const float2 *__restrict__ f)
{

  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
  {
    float ax = f[i].x;
    float ay = f[i].y;
    float vx_i = v[i].x;
    float vy_i = v[i].y;

    vx_i += ax * DELTA_T;
    vy_i += ay * DELTA_T;

    p[i].x += vx_i * DELTA_T;
    p[i].y += vy_i * DELTA_T;

    v[i].x = vx_i;
    v[i].y = vy_i;
  }
}

// Execute one step
template <unsigned int threads, unsigned int blocks>
void step(cudaStream_t compute_stream,
          float2 *d_p, float2 *d_v, float2 *d_f)
{
  // Compute forces
  nbody_forces<threads><<<blocks, threads, 0, compute_stream>>>(d_p, d_f);
  getLastCudaError("Force computation failed");

  // Update positions
  update_pos<<<blocks, threads, 0, compute_stream>>>(d_p, d_v, d_f);

  getLastCudaError("Update position failed");
}

#endif