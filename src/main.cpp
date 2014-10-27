// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  bool loadedScene = false;
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "mesh")==0){
      //renderScene = new scene(data);
      mesh = new obj();
      objLoader* loader = new objLoader(data, mesh);
      mesh->buildVBOs();
	  meshes.push_back(mesh);
      delete loader;
      loadedScene = true;
    }
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
	system("PAUSE");
    return 0;
  }

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  // Launch CUDA/GL
  if (init(argc, argv)) {
    // GLFW main loop
    mainLoop();
  }

  system("PAUSE");
  return 0;
}

void mainLoop() {
  while(!glfwWindowShouldClose(window)){
    glfwPollEvents();
    runCuda();

    time_t seconds2 = time (NULL);

    if(seconds2-seconds >= 1){

        fps = fpstracker/(seconds2-seconds);
        fpstracker = 0;
        seconds = seconds2;
    }

    string title = "CIS565 Rasterizer | " + utilityCore::convertIntToString((int)fps) + " FPS";
		glfwSetWindowTitle(window, title.c_str());
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glClear(GL_COLOR_BUFFER_BIT);   

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
    glfwSwapBuffers(window);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  dptr=NULL;

  vbo = mesh->getVBO();
  vbosize = mesh->getVBOsize();

  float newcbo[] = {0.0, 1.0, 0.0, 
                    0.0, 0.0, 1.0, 
                    1.0, 0.0, 0.0};
  cbo = newcbo;
  cbosize = 9;

  glm::mat4 rotationM = glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(1.0f, 0.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 20.0f-0.5f*frame, glm::vec3(0.0f, 1.0f, 0.0f))*glm::rotate(glm::mat4(1.0f), 0.0f, glm::vec3(0.0f, 0.0f, 1.0f));

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  nbo = mesh->getNBO();
  nbosize = mesh->getNBOsize();

  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), rotationM, frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, eye, center, view, lightpos, mode, barycenter);
  cudaGLUnmapBufferObject(pbo);

  vbo = NULL;
  cbo = NULL;
  ibo = NULL;

  frame++;
  fpstracker++;

}
  
//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(int argc, char* argv[]) {
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
      return false;
  }

  width = 800;
  height = 800;
  window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
  if (!window){
      glfwTerminate();
      return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetMouseButtonCallback(window,MouseClickCallback);
  glfwSetCursorEnterCallback(window,CursorEnterCallback);
  glfwSetCursorPosCallback(window,CursorCallback);

  // Set up GL context
  glewExperimental = GL_TRUE;
  if(glewInit()!=GLEW_OK){
    return false;
  }

  // Initialize other stuff
  initVAO();
  initTextures();
  initCuda();
  initPBO();
  
  GLuint passthroughProgram;
  passthroughProgram = initShader();

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  return true;
}

void initPBO(){
  // set up vertex data parameter
  int num_texels = width*height;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;
    
  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo);

}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice(0);

  // Clean up on program exit
  atexit(cleanupCuda);
}

void initTextures(){
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() {
  const char *attribLocations[] = { "Position", "Tex" };
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;
  
  glUseProgram(program);
  if ((location = glGetUniformLocation(program, "u_image")) != -1)
  {
    glUniform1i(location, 0);
  }

  return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  kernelCleanup();
  cudaDeviceReset();
  #ifdef __APPLE__
  glfwTerminate();
  #endif
  exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char* description){
    fputs(description, stderr);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
	if(key == GLFW_KEY_W && action == GLFW_PRESS){
		eye.y += 0.1f;
	}
	if(key == GLFW_KEY_S && action == GLFW_PRESS){
		eye.y -= 0.1f;
	}
	if(key == GLFW_KEY_A && action == GLFW_PRESS){
		eye.x += 0.1f;
	}
	if(key == GLFW_KEY_D && action == GLFW_PRESS){
		eye.x -= 0.1f;
	}
	if(key == GLFW_KEY_Q && action == GLFW_PRESS){
		eye.z += 0.1f;
	}
	if(key == GLFW_KEY_E && action == GLFW_PRESS){
		eye.z -= 0.1f;
	}
	if(key == GLFW_KEY_N && action == GLFW_PRESS){
		mode++;
		if(mode > 2)
			mode = 0;
	}
	if(key == GLFW_KEY_M && action == GLFW_PRESS){
		if(barycenter)
			barycenter = false;
		else barycenter = true;
	}
}


//mouse functions, changing view matrix and eyepos
void CursorEnterCallback(GLFWwindow *window,int entered){
    if(entered == GL_TRUE)
		inwindow = true;
	else
		inwindow = false;
}

void MouseClickCallback(GLFWwindow *window, int button, int action, int mods){
	if(action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT){
	    glfwGetCursorPos(window,&MouseX,&MouseY);
		LB = true;
	}

	if(action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT){
		glfwGetCursorPos(window,&MouseX,&MouseY);
		RB = true;
	}

	if(action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_MIDDLE){
		glfwGetCursorPos(window,&MouseX,&MouseY);
		MB = true;
	}

	if(action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_LEFT)
		LB = false;

	if(action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT)
		RB = false;

	if(action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_MIDDLE)
		MB = false;

}

void CursorCallback(GLFWwindow *window, double x, double y){
	x = glm::max(0.0, x);
	x = glm::min(x, (double)width);
	y = glm::max(0.0, y);
	y = glm::min(y, (double)height);

	int changeX = x - MouseX;
	int changeY = y - MouseY;

	if(LB&&inwindow){ //camera rotate
		vPhi -= changeX * MOUSE_SPEED;
		vTheta -= changeY * MOUSE_SPEED;
		vTheta = glm::clamp(vTheta, float(1e-6), float(PI-(1e-6)));	
	}

	if(RB&&inwindow){ //zoom in and out
		float scale = -changeX/MouseX + changeY/MouseY;
		R = (1.0f + 0.003f * scale * ZOOM_SPEED) * R;
		R = glm::clamp(R,zNear,zFar);
	}

	if(MB&&inwindow)
	{
		eye -= glm::vec3(0.00001 * MIDDLE_SPEED, 0, 0) * (float)changeX ;
		eye += glm::vec3(0,0.00001 * MIDDLE_SPEED, 0) * (float)changeY;
		center -= glm::vec3(0.00001 * MIDDLE_SPEED, 0, 0) * (float)changeX;
		center += glm::vec3(0,0.00001 * MIDDLE_SPEED, 0) * (float)changeY;
		view = glm::lookAt(eye, center, glm::vec3(0,1,0));
	}

	eye = glm::vec3(R*sin(vTheta)*sin(vPhi), R*cos(vTheta) + center.y, R*sin(vTheta)*cos(vPhi));
	view = glm::lookAt(eye, center, glm::vec3(0,1,0));
}