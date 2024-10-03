#include "render.hh"
#include <GLFW/glfw3.h>
exern "C"
{
#include <glad/glad.h>
}

void render_thread()
{
  GLFWwindow *window;

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  /* Initialize the library */
  if (!glfwInit())
    return -1;

  /* Create a windowed mode window and its OpenGL context */
  window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
  if (!window)
  {
    glfwTerminate();
    return -1;
  }

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window))
  {
    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);

    /* Swap front and back buffers */
    glfwSwapBuffers(window);

    // Register the buffers with CUDA
    cudaGraphicsResource *resource;
    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, buffer, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    /* Poll for and process events */
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}