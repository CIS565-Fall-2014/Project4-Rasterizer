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

  return 0;
}

void mainLoop() {
  while(!glfwWindowShouldClose(window)){
    glfwPollEvents();
	alpha=alpha+1;
	//camera data
  cam.view=glm::vec3(0.0f,0.0f,0.0f);
  cam.up=glm::vec3(0.0,1.0,0.0);
  cam.position.x=D*cos(beta*PI/180.0f)*cos(alpha*PI/180.0f);
  cam.position.y=D*sin(beta*PI/180.0f)-0.5f;
  cam.position.z=D*cos(beta*PI/180.0f)*sin(alpha*PI/180.0f);
  glm::vec3 temp_lateral=glm::normalize(glm::cross(cam.view-cam.position,cam.up));
  cam.up=glm::normalize(glm::cross(temp_lateral,cam.view-cam.position));

  cam.FOV=80.0f;
  z_near=0.1f;
  z_far=100.0f;
  //Transform points from model to clip coordinates	
  model=glm::mat4(1.0f);
  view=glm::lookAt(cam.position,cam.view,cam.up);
  projection=glm::perspective(cam.FOV,(float)width/(float)height,z_near,z_far);

  //test 
  /*glm::vec4 r=glm::vec4(0.0,2.0,1.0,1.0);
  r=projection*view*r;
  cout<<r.x<<" "<<r.y<<" "<<r.z<<" "<<r.w<<endl;*/
  
  //getchar();


  n_modelview=glm::transpose(glm::inverse(view*model));
  M=projection*view*model;
  

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

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  nbo=mesh->getNBO();
  nbosize=mesh->getNBOsize();

 
  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize,nbo,nbosize,model,view,projection,M,n_modelview, z_near,z_far,cam.position);
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
void MouseCallback(GLFWwindow* window,int button,int action,int mods)
{
	if(action==GLFW_PRESS)
	{
		glfwGetCursorPos(window,&x_pos,&y_pos);
		//cout<<x_pos<<" "<<y_pos<<endl;
		if(button==GLFW_MOUSE_BUTTON_1)
		{
			left_button=true;
		}
		else if(button==GLFW_MOUSE_BUTTON_2)
		{
			right_button=true;
		}
	}
	else if(action==GLFW_RELEASE)
	{
		if(button==GLFW_MOUSE_BUTTON_1)
		{
			left_button=false;
		}
		else if(button==GLFW_MOUSE_BUTTON_2)
		{
			right_button=false;
		}
	}
}

void CursorEnterCallback(GLFWwindow* window,int entered)
{
	if(entered==GL_TRUE)
	{
		inside_window=true;
	}
	else
	{
		inside_window=false;
	}
}

void CursorCallback(GLFWwindow* window,double x,double y)
{
	x=glm::clamp(x,0.0,(double)width);
	y=glm::clamp(y,0.0,(double)height);
	int offset_x=x-x_pos;
	int offset_y=y-y_pos;
	x_pos=x;
	y_pos=y;
	if(left_button&&inside_window)
	{
		alpha=alpha+offset_x/(float)5;
		beta=beta+offset_y/(float)5;
	}
	if(right_button&&inside_window)
	{
		D=D+offset_x/(float)30;
	}
}



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
  glfwSetMouseButtonCallback(window,MouseCallback);
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
}

