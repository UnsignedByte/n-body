#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.cuh"

#define N 108 * 512 * 6
#define BLOCK_SIZE 512
#define GRID_SIZE 108
#define GRAVITY 0.001f
#define SOFTENING 0.001f
#define DELTA_T 0.01f

#include "step.cuh"

unsigned int window_width = 2560,
             window_height = 1400;

float height = 10.;
float width = 10.;
bool paused = true;

float2 center = {0, 0};

bool mouse1_down = false;

double mouse_x, mouse_y;

float randf()
{
  return (float)((double)rand() / RAND_MAX);
}

void update_view()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(center.x - width / 2, center.x + width / 2, center.y - height / 2, center.y + height / 2, -1, 1);
}

void resize(GLFWwindow *window, int new_window_width, int new_window_height)
{
  window_width = new_window_width;
  window_height = new_window_height;

  // Adjust width to the aspect ratio
  width = height * (float)window_width / (float)window_height;

  glViewport(0, 0, window_width, window_height);
  update_view();
}

void input_handler(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);

  if (key == GLFW_KEY_W && action != GLFW_RELEASE)
  {
    center.y += height / 50.;
    update_view();
  }
  if (key == GLFW_KEY_S && action != GLFW_RELEASE)
  {
    center.y -= height / 50.;
    update_view();
  }
  if (key == GLFW_KEY_A && action != GLFW_RELEASE)
  {
    center.x -= height / 50.;
    update_view();
  }
  if (key == GLFW_KEY_D && action != GLFW_RELEASE)
  {
    center.x += height / 50.;
    update_view();
  }
  if (key == GLFW_KEY_SPACE && action != GLFW_RELEASE)
  {
    paused = !paused;
  }
}

void scroll_handler(GLFWwindow *window, double xoffset, double yoffset)
{
  double normalized_x = 2.f * mouse_x / window_width - 1.f;
  double normalized_y = 1.f - 2.f * mouse_y / window_height;

  // get the position of the mouse in the world coordinates
  float xpos = normalized_x * width / 2 + center.x;
  float ypos = normalized_y * height / 2 + center.y;

  // get the new width
  width *= 1 + yoffset / 10.;
  height *= 1 + yoffset / 10.;

  // adjust the center to keep the mouse position fixed
  center.x = xpos - normalized_x * width / 2;
  center.y = ypos - normalized_y * height / 2;
  update_view();
}

void mouse_handler(GLFWwindow *window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS)
  {
    mouse1_down = true;
  }
  if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_RELEASE)
  {
    mouse1_down = false;
  }
}

void pan_handler(GLFWwindow *window, double xpos, double ypos)
{
  if (mouse1_down)
  {
    center.x -= (xpos - mouse_x) * width / window_width;
    center.y += (ypos - mouse_y) * height / window_height;
    update_view();
  }

  mouse_x = xpos;
  mouse_y = ypos;
}

int main()
{
  unsigned int window_width = 2560,
               window_height = 1400;

  float aspect_ratio = (float)window_width / (float)window_height;

  GLFWwindow *window;

  /* Initialize the library */
  if (!glfwInit())
    return -1;

  /* Create a windowed mode window and its OpenGL context */
  char window_name[100];
  sprintf(window_name, "Particle simulation with %d particles", N);
  window = glfwCreateWindow(window_width, window_height, window_name, NULL, NULL);
  if (!window)
  {
    glfwTerminate();
    return -1;
  }

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  resize(window, window_width, window_height);
  glfwSetFramebufferSizeCallback(window, resize);

  /// Set up callbacks
  glfwSetKeyCallback(window, input_handler);
  glfwSetScrollCallback(window, scroll_handler);
  glfwSetMouseButtonCallback(window, mouse_handler);
  glfwSetCursorPosCallback(window, pan_handler);

  cudaSetDevice(0);

  statDevice();

  // Allocate px and py in host memory
  float2 *h_p, *h_v;
  float2 *d_p, *d_v, *d_f;
  checkCudaErrors(cudaMallocHost(&h_p, sizeof(float2) * N));
  checkCudaErrors(cudaMallocHost(&h_v, sizeof(float2) * N));

  // Fill with random values in the screen
  for (int i = 0; i < N; i++)
  {
    float theta = randf() * 2 * M_PI;
    float distance = 0;
    int num_samples = 3;
    for (int j = 0; j < num_samples; j++)
    {
      distance += randf();
    }
    distance = sqrtf(distance / num_samples);

    float cosine = cosf(theta);
    float sine = sinf(theta);

    h_p[i].x = distance * cosine;
    h_p[i].y = distance * sine;

    float orbital_velocity = sqrtf(10. / distance);

    // Velocity around the center
    h_v[i].x = -sine * orbital_velocity;
    h_v[i].y = cosine * orbital_velocity;
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

  // Copy velocities to the device
  checkCudaErrors(cudaMemcpy(d_v, h_v, positions_size, cudaMemcpyHostToDevice));

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

    if (paused)
    {
      continue;
    }

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