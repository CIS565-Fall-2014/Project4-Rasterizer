// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glslUtil/glslUtility.hpp>
#include <iostream>
#include <objUtil/objloader.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "../external/include/glm/gtc/matrix_transform.hpp"


#include "rasterizeKernels.h"
#include "utilities.h"

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

GLFWwindow *window;

obj* mesh;

float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;
//Added
float* nbo;
int nbosize;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width = 800; int height = 800;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

#ifdef __APPLE__
	void display();
#else
	void display();
	void keyboard(unsigned char key, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------
bool init(int argc, char* argv[]);
void initPBO();
void initCuda();
void initTextures();
void initVAO();
GLuint initShader();

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

//Add mouse control
double MouseX = 0.0;
double MouseY = 0.0;
bool LB = false;
bool MB = false;
bool RB = false;
bool inwindow = false;
void MouseClickCallback(GLFWwindow *window, int button, int action, int mods);
void CursorCallback(GLFWwindow *window, double x,double y);
void CursorEnterCallback(GLFWwindow *window,int entered);

//Added data
glm::mat4 model =glm::mat4();
glm::vec2 nearfar = glm::vec2(0.1f, 100.0f);
glm::vec3 eye = glm::vec3(0, 0, 0.5);

//Make the camera move on the surface of sphere
float vPhi = 0.0f;
float vTheta = 3.14105926f/2.0f;
float R = glm::length(eye);

glm::vec3 center = glm::vec3(0,0,0);
glm::mat4 projection = glm::perspective(60.0f, (float)(width) / (float)(height),nearfar.x,nearfar.y);
glm::mat4 view = glm::lookAt(eye,center , glm::vec3(0, 1, 0));
glm::vec3 lightpos = glm::vec3(0,4,4);
glm::mat4 modelview = view * model;
//modelview for normal
glm::mat4 normalModelview = glm::transpose(glm::inverse(modelview));

//Added Bool
bool BackfaceCulling = false;
bool BCInterp = false;
bool AntiAliasing = false;
bool LineMode = false;
bool PointMode = false;
bool ShowBody = true;

#endif