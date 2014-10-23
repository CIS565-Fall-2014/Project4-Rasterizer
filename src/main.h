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
#include <math.h>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"

#include "rasterizeKernels.h"
#include "utilities.h"
//#include "rasterizeTools.h"

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

glm::vec3 translationVec;
glm::vec3 scaleVec;
glm::vec3 rotateVec;

glm::vec3 light;
glm::vec3 lightPos;
glm::vec3 eye;
glm::vec3 center(0, 0, 0);
glm::vec3 up(0, 1, 0);
glm::mat4 camera;
float phi = 90;
float theta = 0;
float lightPhi = 90;
float lightTheta = 0;


float scale = 400;

float fovy;
float aspect;
float zNear; 
float zFar;
glm::mat4 perspective;
glm::mat4 transformationMat;

double cursorPosX;
double cursorPosY;


int translateX;
int translateY;

bool leftButtonPressed;
bool rightButtonPressed;
bool ctlKeyPressed;
bool shiftKeyPressed;
bool altKeyPressed;

bool scissorTest = false;
bool backCulling = true;
bool alphaBlend = false;
bool antialiasing = false;
float alphaValue = 0.5;

cudaMat4 shaderMatrix;

GLFWwindow *window;

obj* mesh;

float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;
float* nbo;
int nbosize;

int* lineIbo;
int lineIboSize;


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
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double xPos, double yPos);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
#endif