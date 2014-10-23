// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glslUtil/glslUtility.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include <iostream>
#include <objUtil/objloader.h>
#include <sstream>
#include <string>
#include <time.h>


#include "rasterizeKernels.h"
#include "utilities.h"

using namespace std;

#define FOV_DEG 30
glm::vec3 lightPos = glm::vec3(0.5f,3.0f,5.0f);
glm::vec3 lightCol = glm::vec3(1.0f,1.0f,1.0f);

light Light;

//transformations
glm::mat4 glmProjectionTransform;
glm::mat4 glmMVtransform;

//mouse control stuff
bool mouseButtonIsDown = false;
float mouseScrollOffset = 0.0f;
double mouseClickedX = 0.0f;
double mouseClickedY = 0.0f;
double rotationX = 0.0f;
double rotationY = 0.0f;
double mouseDeltaX = 0.0f;
double mouseDeltaY = 0.0f;
//keyboard control
double deltaX = 0.0f;
double deltaZ = 0.0f;
double cameraMovementIncrement = 0.015f;

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

#endif