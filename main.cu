#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.hu"
#include "step.hu"

#define N 1024 * 1024 * 32
#define BLOCK_SIZE 256
#define GRID_SIZE 1024 * 8

float randf()
{
  return (float)((double)rand() / RAND_MAX);
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
  glViewport(0, 0, width, height);
}

int main()
{
  unsigned int width = 800,
               height = 600;

  GLFWwindow *window;

  /* Initialize the library */
  if (!glfwInit())
    return -1;

  /* Create a windowed mode window and its OpenGL context */
  window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
  if (!window)
  {
    glfwTerminate();
    return -1;
  }

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  glViewport(0, 0, width, height);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  cudaSetDevice(0);

  // Allocate px and py in host memory
  float2 *h_p;
  checkCudaErrors(cudaMallocHost(&h_p, sizeof(float2) * N));

  // Fill with random values in the screen
  for (int i = 0; i < N; i++)
  {
    h_p[i].x = randf() * width;
    h_p[i].y = randf() * height;
  }

  size_t positions_size = N * sizeof(float2);

  // Create buffer object and register it with CUDA
  cudaGraphicsResource_t positions_resource;
  GLuint positions;
  glGenBuffers(1, &positions);
  // Copy to the openGL buffer
  glBindBuffer(GL_ARRAY_BUFFER, positions);
  glBufferData(GL_ARRAY_BUFFER, positions_size, h_p, GL_DYNAMIC_DRAW);

  // Register the buffer with CUDA
  cudaGraphicsGLRegisterBuffer(&positions_resource, positions, cudaGraphicsRegisterFlagsWriteDiscard);

  // Allocate vectors in device memory
  float *d_vx, *d_vy, *d_fx, *d_fy;
  checkCudaErrors(cudaMalloc(&d_vx, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_vy, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_fx, sizeof(float) * N));
  checkCudaErrors(cudaMalloc(&d_fy, sizeof(float) * N));

  // Vectors must be initialized with zeros
  checkCudaErrors(cudaMemset(d_vx, 0, sizeof(float) * N));
  checkCudaErrors(cudaMemset(d_vy, 0, sizeof(float) * N));
  // Force vectors do not need to be initialized

  cudaStream_t compute_stream;
  // Allocate the stream
  checkCudaErrors(cudaStreamCreate(&compute_stream));

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
    float2 *d_p;
    cudaGraphicsMapResources(1, &positions_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_p, &positions_size, positions_resource);

    // step<BLOCK_SIZE, GRID_SIZE>(compute_stream, d_p, d_vx, d_vy, d_fx, d_fy);

    cudaGraphicsUnmapResources(1, &positions_resource, 0);
    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw the particles
    glBindBuffer(GL_ARRAY_BUFFER, positions);

    glVertexPointer(2, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, N);
    glDisableClientState(GL_VERTEX_ARRAY);

    /* Swap front and back buffers */
    glfwSwapBuffers(window);

    /* Poll for and process events */
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}