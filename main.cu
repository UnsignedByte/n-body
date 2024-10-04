#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.hu"

#define N 108 * 512 * 6
#define BLOCK_SIZE 512
#define GRID_SIZE 108
#define GRAVITY 1.f
#define SOFTENING 0.001f
#define DELTA_T 0.01f

#include "step.hu"

float randf()
{
  return (float)((double)rand() / RAND_MAX);
}

void resize(GLFWwindow *window, int width, int height)
{
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, width, height, 0, -1, 1);
}

int main()
{
  unsigned int width = 2560,
               height = 1400;

  GLFWwindow *window;

  /* Initialize the library */
  if (!glfwInit())
    return -1;

  /* Create a windowed mode window and its OpenGL context */
  char window_name[100];
  sprintf(window_name, "Particle simulation with %d particles", N);
  window = glfwCreateWindow(width, height, window_name, NULL, NULL);
  if (!window)
  {
    glfwTerminate();
    return -1;
  }

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  resize(window, width, height);
  glfwSetFramebufferSizeCallback(window, resize);

  cudaSetDevice(0);

  statDevice();

  // Allocate px and py in host memory
  float2 *h_p;
  float2 *d_p, *d_v, *d_f;
  checkCudaErrors(cudaMallocHost(&h_p, sizeof(float2) * N));

  // Fill with random values in the screen
  for (int i = 0; i < N; i++)
  {
    h_p[i].x = randf() * width;
    h_p[i].y = randf() * height;
  }

  printf("Initialized %d particles\n", N);

  if (GLenum err = glewInit() != GLEW_OK)
  {
    printf("Failed to initialize GLEW: error code %u\n", err);
    return -1;
  }

  size_t positions_size = N * sizeof(float2);

  GLuint positions;
  glGenBuffers(1, &positions);
  // Copy to the openGL buffer
  glBindBuffer(GL_ARRAY_BUFFER, positions);
  glBufferData(GL_ARRAY_BUFFER, positions_size, h_p, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

#ifdef __WSL__
  // WSL does not support OpenGL interoperability
  checkCudaErrors(cudaMalloc(&d_p, positions_size));
  checkCudaErrors(cudaMemcpy(d_p, h_p, positions_size, cudaMemcpyHostToDevice));
#else
  // Create buffer object and register it with CUDA
  cudaGraphicsResource_t positions_resource;

  // Register the buffer with CUDA
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&positions_resource, positions, cudaGraphicsRegisterFlagsNone));
#endif

  // Allocate vectors in device memory
  checkCudaErrors(cudaMalloc(&d_v, positions_size));
  checkCudaErrors(cudaMalloc(&d_f, positions_size));

  // Vectors must be initialized with zeros
  checkCudaErrors(cudaMemset(d_v, 0, positions_size));
  // Force vectors do not need to be initialized

  cudaStream_t compute_stream;
  // Allocate the stream
  checkCudaErrors(cudaStreamCreate(&compute_stream));

  glColor3f(1.0f, 1.0f, 1.0f);
  glPointSize(1.0f);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
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

#ifdef __WSL__
    // Launch the kernel
    step<BLOCK_SIZE, GRID_SIZE>(compute_stream, d_p, d_v, d_f);
    // Wait for the kernel to finish
    checkCudaErrors(cudaStreamSynchronize(compute_stream));

    // Copy the results back to the host
    checkCudaErrors(cudaMemcpy(h_p, d_p, positions_size, cudaMemcpyDeviceToHost));

    // Copy the results to the OpenGL buffer
    glBindBuffer(GL_ARRAY_BUFFER, positions);
    glBufferData(GL_ARRAY_BUFFER, positions_size, h_p, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

#else
    // Launch the kernel
    checkCudaErrors(cudaGraphicsMapResources(1, &positions_resource, compute_stream));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_p, &positions_size, positions_resource));

    step<BLOCK_SIZE, GRID_SIZE>(compute_stream, d_p, d_v, d_f);

    // Wait for the kernel to finish
    checkCudaErrors(cudaStreamSynchronize(compute_stream));
    cudaGraphicsUnmapResources(1, &positions_resource, compute_stream);
#endif
  }

  glfwTerminate();

  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_f));
  return 0;
}