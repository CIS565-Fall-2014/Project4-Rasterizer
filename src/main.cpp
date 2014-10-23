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
      delete loader;
      loadedScene = true;
    }
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
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

  system("pause");
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

  float newcbo[9]= {0};
  newcbo[1] = 1.0f;
  newcbo[5] = 1.0f;
  newcbo[6] = 1.0f;

  cbo = newcbo;
  cbosize = 9;

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  nbo = mesh->getNBO();
  nbosize = mesh->getNBOsize();

  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, 
	  cbosize, ibo, ibosize,nbo,nbosize);
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
  window = glfwCreateWindow(width, height, "CIS 565 Rasterizer", NULL, NULL);
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

	if(key == GLFW_KEY_A && action == GLFW_PRESS)
	{
		AntiAliasing = !AntiAliasing;
		if(AntiAliasing)
			std::cout<<"Anti Aliasing Enabled!"<<std::endl;
		else
			std::cout<<"Anti Aliasing Disabled!"<<std::endl;
	}

	if(key == GLFW_KEY_B && action == GLFW_PRESS)
	{
		BackfaceCulling = !BackfaceCulling;
		if(BackfaceCulling)
			std::cout<<"BackfaceCulling Enabled!"<<std::endl;
		else
			std::cout<<"BackfaceCulling Disabled!"<<std::endl;
	}

	if(key == GLFW_KEY_I && action == GLFW_PRESS)
	{
		BCInterp = !BCInterp;
		if(BCInterp)
			std::cout<<"Interpolation Enabled!"<<std::endl;
		else
			std::cout<<"Interpolation Disabled!"<<std::endl;
	}

	if(key == GLFW_KEY_L && action == GLFW_PRESS)
	{
		LineMode = !LineMode;
		if(LineMode)
			std::cout<<"Show Lines!"<<std::endl;
		else
			std::cout<<"Hide Lines!"<<std::endl;
	}

	if(key == GLFW_KEY_P && action == GLFW_PRESS)
	{
		PointMode = !PointMode;
		if(PointMode)
			std::cout<<"Show Points!"<<std::endl;
		else
			std::cout<<"Hide Points!"<<std::endl;
	}

	if(key == GLFW_KEY_H && action == GLFW_PRESS)
	{
		ShowBody = !ShowBody;
		if(ShowBody)
			std::cout<<"Show Tris!"<<std::endl;
		else
			std::cout<<"Hide Tris!"<<std::endl;
	}
}

//Added mouse functions

void MouseClickCallback(GLFWwindow *window, int button, int action, int mods)
{
	if(action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_1)
	{
	    glfwGetCursorPos(window,&MouseX,&MouseY);
		LB = true;
	}

	if(action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_2)
	{
		glfwGetCursorPos(window,&MouseX,&MouseY);
		RB = true;
	}

	if(action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_3)
	{
		glfwGetCursorPos(window,&MouseX,&MouseY);
		MB = true;
	}

	if(action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_1)
		LB = false;

	if(action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_2)
		RB = false;

	if(action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_3)
		MB = false;
}

void CursorEnterCallback(GLFWwindow *window,int entered)
{
    if(entered == GL_TRUE)
		inwindow = true;
	else
		inwindow = false;
}

void CursorCallback(GLFWwindow *window, double x,double y)
{
	x = glm::max(0.0, x);
	x = glm::min(x, (double)width);
	y = glm::max(0.0, y);
	y = glm::min(y, (double)height);

	int changeX = x - MouseX;
	int changeY = y - MouseY;

	if(LB&&inwindow)
	{
		vPhi -= changeX * 0.0001f;
		vTheta -= changeY * 0.0001f;
		vTheta = glm::clamp(vTheta, float(1e-6), float(PI-(1e-6)));
		eye = glm::vec3(R*sin(vTheta)*sin(vPhi), R*cos(vTheta) + center.y, R*sin(vTheta)*cos(vPhi));
		view = glm::lookAt(eye, center, glm::vec3(0,1,0));
	}

	if(MB&&inwindow)
	{
		eye -= glm::vec3(0.00001, 0, 0) * (float)changeX;
		eye += glm::vec3(0,0.00001, 0) * (float)changeY;
		center -= glm::vec3(0.00001, 0, 0) * (float)changeX;
		center += glm::vec3(0,0.00001, 0) * (float)changeY;
		view = glm::lookAt(eye, center, glm::vec3(0,1,0));
	}

	if(RB&&inwindow)
	{
		float scale = -changeX/MouseX + changeY/MouseY;
		R = (1.0f + 0.003f * scale) * R;
		R = glm::clamp(R,nearfar.x,nearfar.y);
		eye = glm::vec3(R*sin(vTheta)*sin(vPhi), R*cos(vTheta) + center.y, R*sin(vTheta)*cos(vPhi));
		view = glm::lookAt(eye, center, glm::vec3(0,1,0));
	}
}