// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"
#pragma   comment(lib,"FreeImage.lib")

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
  //iniCamera();

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


double xpos, ypos, xposLast, yposLast;   //mouse position

//fovy, aspect, eye, up, lookat
cam mouseCam(65.0f, (float)width/(float)height, glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f));   //cam class


void mainLoop() {

  while(!glfwWindowShouldClose(window)){

    glfwPollEvents();
    runCuda();

	//fps calculation
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

	// check for mouse press
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (state == GLFW_PRESS){
		glfwGetCursorPos(window, &xpos, &ypos);
		rotateMouseCam(xpos, ypos, xposLast, yposLast);
	}
	int state2 = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
	if (state2 == GLFW_PRESS){
		glfwGetCursorPos(window, &xpos, &ypos);
		translateMouseCam(xpos, ypos, xposLast, yposLast);
	}
	glfwGetCursorPos(window,&xposLast,&yposLast);	

  }
  glfwDestroyWindow(window);
  glfwTerminate();
}

void rotateMouseCam(double x, double y, double xlast, double ylast){

	float PI = 3.1415926f;
	float Sensitivity  = 0.1f;
	float rotX = ((float)x - (float)xlast ) * Sensitivity;
	float rotY = ((float)y - (float)ylast ) * Sensitivity;
	mouseCam.theta -= rotY;   //theta is the vertical rotation
	glm::clamp(mouseCam.theta, 0.0001f, 179.9999f);
	mouseCam.phi += rotX;   //phi is the horizontal circle rotation
	if(mouseCam.phi > 360.0f)
		mouseCam.phi -= 360.0f;
	mouseCam.calculateCamPos();
}

void translateMouseCam(double x, double y, double xlast, double ylast){

	float Sensitivity  = 0.001f;
	float transX = ((float)x - (float)xlast ) * Sensitivity;
	float transY = ((float)y - (float)ylast ) * Sensitivity;
	glm::vec3 viewDir = glm::normalize(mouseCam.center - mouseCam.eye);
	glm::vec3 rightDir = glm::cross(viewDir, mouseCam.up);
	if(mouseCam.phi < 180.0f){
		//mouseCam.center.y += transY;
		//mouseCam.center.x += transX;
		mouseCam.center += transY * mouseCam.up;
		mouseCam.center += transX * rightDir;
	}
	else{
		//mouseCam.center.y -= transY;
		//mouseCam.center.x -= transX;
		mouseCam.center -= transY * mouseCam.up;
		mouseCam.center -= transX * rightDir;
	}
	
	mouseCam.calculateCamPos();
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
  
  /*cbo = mesh->getCBO();
  cbosize = mesh->getCBOsize();*/

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  nbo = mesh->getNBO();
  nbosize = mesh->getNBOsize();

  
  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize);
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

  glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);   //sticking mouse button
  
  glfwSetKeyCallback(window, keyCallback);    //keyboard callback
 // glfwSetMouseButtonCallback(window, mousebuttonCallback);   //mouse button callback
  glfwSetScrollCallback(window,mousescrollCallback);   //mouse scroll callback	
 //glfwSetCursorPosCallback(window,mousemoveCallback);  //mouse move callback

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

  //initialize texture
initTextureMap("C:/Users/AppleDu/Documents/GitHub/Project4-Rasterizer/textures/magicCube.png");
  
  
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
int PERFORMANCE_MEASURE = 0;
int SHADING_MODE = 2;  //0-shading based on normal, 1-shade based on depth, 2-diffuse, 3-blinn, 4-texture map,  5-original cbo,
int POINT_RASTER = 0;  //0/1 to off/on points display
int LINE_RASTER = 0; //0/1 to off/on lines display

void errorCallback(int error, const char* description){
    fputs(description, stderr);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
	else if(key == GLFW_KEY_P && action == GLFW_PRESS){  //performance analysis
		PERFORMANCE_MEASURE = 1 - PERFORMANCE_MEASURE;
	}
	else if(key == GLFW_KEY_1 && action == GLFW_PRESS){  //point render
		POINT_RASTER = 1 - POINT_RASTER;
	}
	else if(key == GLFW_KEY_2 && action == GLFW_PRESS){  //line render
		LINE_RASTER = 1 - LINE_RASTER;
	}
	else if(key == GLFW_KEY_3 && action == GLFW_PRESS){  //normal based shading
		SHADING_MODE = 0;
	}
	else if(key == GLFW_KEY_4 && action == GLFW_PRESS){  //vertex interpolation shading
		SHADING_MODE = 5;
	}
	else if(key == GLFW_KEY_5 && action == GLFW_PRESS){  //diffuse
		SHADING_MODE = 2;
	}
	else if(key == GLFW_KEY_6 && action == GLFW_PRESS){  //blinn
		SHADING_MODE = 3;
	}
	else if(key == GLFW_KEY_7 && action == GLFW_PRESS){  //texture map   //TODO
		SHADING_MODE = 4;
	}
}


void mousescrollCallback(GLFWwindow * window, double xoffset, double yoffset ){  
	// x is always 0, only y is changing, +ve for upwards, -ve for downwards
	//printf("mouse scroll %.2f %.2f\n", xoffset, yoffset);
	float Sensitivity = 0.8f;

	//correct zoom...TODO
	//mouseCam.eye.z -= yoffset * Sensitivity;
	//mouseCam.calculateMVP();

	//cheap zoom
	if(mouseCam.fovy - Sensitivity * yoffset > 5.0f && mouseCam.fovy - Sensitivity * yoffset < 180.0f){
		mouseCam.fovy -= Sensitivity * yoffset;
		mouseCam.calculateMVP();
	}
}

void mousemoveCallback(GLFWwindow *window, double xpos, double ypos){
	xposLast = xpos;
	yposLast = ypos;
}


//------------------------------
//-------TEXTURE STUFF---------
//------------------------------
//http://www.mingw.org/
//http://freeimage.sourceforge.net/download.html
//https://www.opengl.org/discussion_boards/showthread.php/163929-image-loading?p=1158293#post1158293
//http://inst.eecs.berkeley.edu/~cs184/fa09/resources/sec_UsingFreeImage.pdf

//loading and initializing texture map
tex textureMap;   //defined in "rasterizeKernel.h"
std::vector<glm::vec3> textureColor;

void initTextureMap(char* textureFileName){
	int h = 0,  w = 0;
	int tmp = loadTexture(textureFileName,textureColor,h,w);
	if( tmp != -1){
		textureMap.id = tmp;   //start index, point to textureColor
		textureMap.h = h;   //height
		textureMap.w = w;   //width
	}
}

int loadTexture(char* file, std::vector<glm::vec3> &c, int &h,int &w){
	FIBITMAP* image = FreeImage_Load( FreeImage_GetFileType(file, 0), file);
	if(!image){
		printf("Error: fail to open texture file %s\n", file );
		FreeImage_Unload(image);
		return -1;
	}
	image = FreeImage_ConvertTo32Bits(image);
	 
	w = FreeImage_GetWidth(image);
	h = FreeImage_GetHeight(image);
	if( w == 0 && h == 0 ) {
		printf("Error: texture file is empty\n");
		FreeImage_Unload(image);
		return -1;
	}

	int start = c.size();
	//int total = w * h;
	//if(n.size()>0)  //useful when load multiple picture of texture
		//total += n[n.size()-1];
	//n.push_back(total);

	//int k = 0;
	for(int i = 0; i < w; i++){
	   for(int j = 0;j < h; j++){
		   RGBQUAD color;
		   FreeImage_GetPixelColor( image, i, j, &color );
		   glm::vec3 nc(color.rgbRed, color.rgbGreen, color.rgbBlue);
	       c.push_back(nc);
	
		   //printf("color @ %d is %.2f, %.2f, %.2f\n",k, c[k].r, c[k].g, c[k].b);
		  // k++;
	   }
	}
	
	FreeImage_Unload(image);
	printf("Loaded texture %s with %dx%d pixels\n", file,w,h );
	return start;
}