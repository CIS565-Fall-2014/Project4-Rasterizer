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

	nbo = mesh->getNBO();
	nbosize = mesh->getNBOsize();





	//for(int i = 0; i < ibosize / 3; ++i){
	//	cout << "(" << vbo[9 * i] << ", " << vbo[9 * i + 1] << ", " <<vbo[9 * i + 2] << ")";
	//	cout << "(" << vbo[9 * i + 3] << ", " << vbo[9 * i + 4] << ", " <<vbo[9 * i + 5] << ")";
	//	cout << "(" << vbo[9 * i + 6] << ", " << vbo[9 * i + 7] << ", " <<vbo[9 * i + 8] << ")" << endl;

	//	cout << "(" << nbo[9 * i] << ", " << nbo[9 * i + 1] << ", " <<nbo[9 * i + 2] << ")";
	//	cout << "(" << nbo[9 * i + 3] << ", " << nbo[9 * i + 4] << ", " <<nbo[9 * i + 5] << ")";
	//	cout << "(" << nbo[9 * i + 6] << ", " << nbo[9 * i + 7] << ", " <<nbo[9 * i + 8] << ")" << endl;
	//}

	cudaGLMapBufferObject((void**)&dptr, pbo);
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, 
		nbo, nbosize, shaderMatrix, translateX, translateY, eye, light, alphaBlend, alphaValue, backCulling, scissorTest, antialiasing);
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
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetScrollCallback(window, scrollCallback);

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


  	theta = 180;

	eye.x = center.x + 1 * sin(phi * PI / 180.0) * cos(theta * PI / 180.0);
	eye.y = center.y + 1 * cos(phi * PI / 180.0);
	eye.z = center.z + 1 * sin(phi * PI / 180.0) * sin(theta * PI / 180.0);
	glm::vec3 lateral = glm::normalize(glm::cross((center - eye), up));
	glm::vec3 realUp = glm::normalize(glm::cross(lateral, (center - eye)));

	lightPhi = 125.26; //125.26  54.74
	lightTheta = 135;

	lightPos.x = center.x + 1 * sin(lightPhi * PI / 180.0) * cos(lightTheta * PI / 180.0);
	lightPos.y = center.y + 1 * cos(lightPhi * PI / 180.0);
	lightPos.z = center.z + 1 * sin(lightPhi * PI / 180.0) * sin(lightTheta * PI / 180.0);
	light = glm::normalize(center - lightPos);

    camera = glm::lookAt(eye, center, realUp);
	//glm::vec3 lighttest = glm::normalize(glm::vec3(1, 1, -1));
	fovy = 60;
	aspect = 1;
	zNear = 10;
	zFar = 100;
	
	perspective = glm::perspective(fovy, aspect, zNear, zFar);

	translationVec = glm::vec3(0,0,0);
	scaleVec = glm::vec3(scale, scale, scale);
	rotateVec = glm::vec3(0,0,0);

	transformationMat = utilityCore::buildTransformationMatrix(translationVec, rotateVec,  scaleVec);

	glm::mat4 integrateMatrix = perspective *  camera * transformationMat;
	shaderMatrix = utilityCore::glmMat4ToCudaMat4(integrateMatrix);


	translateX = width / 2;
	translateY = height / 2;


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
	if(key == GLFW_KEY_A && action == GLFW_PRESS){
		alphaBlend = !alphaBlend;
    }
	if(key == GLFW_KEY_Q && action == GLFW_PRESS){
		alphaValue += 0.05;
		alphaValue = min(alphaValue, (float)1);
    }
	if(key == GLFW_KEY_W && action == GLFW_PRESS){
		alphaValue -= 0.05;
		alphaValue = max(alphaValue, (float)0);
    }
	if(key == GLFW_KEY_S && action == GLFW_PRESS){
		scissorTest = !scissorTest;
    }
	if(key == GLFW_KEY_B && action == GLFW_PRESS){
		backCulling = !backCulling;
    }
	if(key == GLFW_KEY_T && action == GLFW_PRESS){
		antialiasing = !antialiasing;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods){
	//action: release 0;  press 1
	//button: left 0;  right 1;  middle 2
	//mods: Ctrl 2;  Alt 4;  Shift 1

	if(action == 1){
		if(button == 0){
			leftButtonPressed = true;
		}
		else if(button == 1){
			rightButtonPressed = true;
		}

		if(mods ==1)
			shiftKeyPressed = true;
		else if(mods == 2)
			ctlKeyPressed = true;
		else if(mods == 4)
			altKeyPressed = true;
	}
	else{
		leftButtonPressed = false;
		shiftKeyPressed = false;
		ctlKeyPressed = false;
		altKeyPressed = false;

	}
}

void cursorPosCallback(GLFWwindow* window, double xPos, double yPos){


	if(leftButtonPressed){

		int mouseTranslateX = xPos - cursorPosX;
		int mouseTranslateY = yPos - cursorPosY;

		cursorPosX = xPos;
		cursorPosY = yPos;
		if(ctlKeyPressed == true){
			translateX += mouseTranslateX;
			translateY -= mouseTranslateY;
		}
		else if(altKeyPressed == true){
			lightPhi += mouseTranslateY;
			lightTheta += mouseTranslateX;

			lightPos.x = center.x + 1 * sin(lightPhi * PI / 180.0) * cos(lightTheta * PI / 180.0);
			lightPos.y = center.y + 1 * cos(lightPhi * PI / 180.0);
			lightPos.z = center.z + 1 * sin(lightPhi * PI / 180.0) * sin(lightTheta * PI / 180.0);
			light = glm::normalize(center - lightPos);

		}
		else{
			theta -= mouseTranslateX;
			phi += mouseTranslateY;

			eye.x = center.x + 1 * sin(phi * PI / 180.0) * cos(theta * PI / 180);
			eye.y = center.y + 1 * cos(phi * PI / 180.0);
			eye.z = center.z + 1 * sin(phi * PI / 180.0) * sin(theta * PI / 180);
			camera = glm::lookAt(eye, center, up);

			glm::vec3 lateral = glm::normalize(glm::cross((center - eye), up));
			glm::vec3 realUp = glm::normalize(glm::cross(lateral, (center - eye)));


			camera = glm::lookAt(eye, center, realUp);

			glm::mat4 integrateMatrix = perspective *  camera * transformationMat;
			shaderMatrix = utilityCore::glmMat4ToCudaMat4(integrateMatrix);
		}


	}
	else{
		cursorPosX = xPos;
		cursorPosY = yPos;
	}
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffset){

	scale += yOffset * 20;

	translationVec = glm::vec3(0,0,0);
	scaleVec = glm::vec3(scale,scale,scale);
	rotateVec = glm::vec3(0,0,0);
	transformationMat = utilityCore::buildTransformationMatrix(translationVec, rotateVec,  scaleVec);

	glm::mat4 integrateMatrix = perspective *  camera * transformationMat;
	shaderMatrix = utilityCore::glmMat4ToCudaMat4(integrateMatrix);
}