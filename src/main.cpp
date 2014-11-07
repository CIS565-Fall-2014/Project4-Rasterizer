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
	
	  vbo = mesh->getVBO();
	  vbosize = mesh->getVBOsize();

	  float newcbo[] = {0.5, 0.5, 0.5, 
						0.5, 0.5, 0.5, 
						0.5, 0.5, 0.5};
	  cbo = newcbo;
	  cbosize = 9;

	  ibo = mesh->getIBO();
	  ibosize = mesh->getIBOsize();

	  nbo = mesh->getNBO();
	  nbosize = mesh->getNBOsize();

	  //==============for textures!========================
	  glm::vec3 min; 
	  glm::vec3 max;
	  utilityCore::getAABBForMesh(vbo,vbosize,min,max);
	  //utilityCore::printVec3(min);
	  //utilityCore::printVec3(max);
	  //vector<glm::vec4> * texCoord = meshes[i]->getTextureCoords();
	  //cout<< texCoord->size() << endl;
	   textureColor = new float[vbosize];
	  cbosize = vbosize ;
	  float w = texture1.width();
	  float h = texture1.height();
	  float u_scalarX = 1.0f/(max.x - min.x);
	  float u_scalarY = 1.0f/(max.y - min.y);
	  for(int i = 0; i < vbosize; i+= 3)
	  {
		  unsigned char red;
	      unsigned char green;
	      unsigned char blue;
		  int u = (vbo[i]-min.x) * u_scalarX * w;
		  int v = (vbo[i+1]-min.y) * u_scalarY * h;
		  texture1.get_pixel( u,v ,red,green,blue);
		  textureColor[i] = red/255.0f;
		  textureColor[i+1] = green/255.0f;
		  textureColor[i+2] = blue/255.0f;
		  //cout << (*texCoord)[i].x << " "  << (*texCoord)[i].y  << " " << (*texCoord)[i].z << endl;
	  }
	  //cbo = textureColor;
	  //==============================================
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
//---------UI STUFF--------------
//-------------------------------
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	if(isButtonPressed)
	{
		glm::mat4 transform;
		glm::vec4 eye(eyePos, 1.0);
		double offsetX = lastX - xpos;
		double offsetY = lastY - ypos;
		//cout << "Last: " << lastX << ", " << lastY << endl;
		//cout << "Current: " << xpos << ", " << ypos << endl;
		switch(buttonPressed)
		{
		case(GLFW_MOUSE_BUTTON_LEFT):
			transform = utilityCore::buildTransformationMatrix(glm::vec3(0.0), glm::vec3(0,10 * offsetX/800.0f, 10 * offsetY/800.0f) , glm::vec3(1.0f));
			eye = transform * eye;
			eyePos.x = eye.x;
			eyePos.y = eye.y;
			eyePos.z = eye.z;
			break;
		case(GLFW_MOUSE_BUTTON_RIGHT):
			//double offsetY = lastY - ypos;
			/*transform = utilityCore::buildTransformationMatrix(glm::vec3(0.0, 0.0, offsetX/800.0f), glm::vec3(0,0, 0) , glm::vec3(1.0f));
			eye = transform * eye;
			eyePos.x = eye.x;
			eyePos.y = eye.y;
			eyePos.z = eye.z;*/
			eyePos += (center - eyePos) * (float)offsetX/800.0f;
			break;
		}
	}
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if( action == GLFW_PRESS)
	{
		isButtonPressed = 1;
		buttonPressed = button;
		glfwGetCursorPos(window, &lastX, &lastY);
	}
	else
	{
		isButtonPressed = 0;
		lastX = -1;
		lastY = -1;
	}
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	dptr=NULL;
	cudaGLMapBufferObject((void**)&dptr, pbo);


	 glm::mat4 transform  =  utilityCore::buildTransformationMatrix(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f));
	  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, textureColor, cbosize, ibo, ibosize, nbo, nbosize, eyePos, center, transform, backgroundTexture, frame/100.0f);  
	cudaGLUnmapBufferObject(pbo);



  //vbo = NULL;
  //cbo = NULL;
  //ibo = NULL;
  //nbo = NULL;
 // textureColor = NULL;
  frame++;
  fpstracker++;

}
  
//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(int argc, char* argv[]) {
   int h = backgroundImage.height();
   int w = backgroundImage.width();
   backgroundTexture = new float[height * width * 3];
   cout << h << " " << w << endl;
   for (std::size_t y = 0; y < h; ++y)
   {
      for (std::size_t x = 0; x < w; ++x)
      {
		 unsigned char red;
	     unsigned char green;
	     unsigned char blue;
		 int index = x + y * w;
         backgroundImage.get_pixel(x,y,red,green,blue);
         backgroundTexture[3*index] = red/255.0f;
		 backgroundTexture[3*index+1] = green/255.0f;
		 backgroundTexture[3*index+2] = blue/255.0f; 
      }
   }

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
   glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_position_callback);


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