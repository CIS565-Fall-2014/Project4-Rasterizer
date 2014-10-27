// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H

#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glslUtil/glslUtility.hpp>
#include <iostream>
#include <objUtil/objloader.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>


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
float* nbo;
int nbosize;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width = 800; int height = 800;

//-------------------------------
//----------Camera STUFF-----------
//-------------------------------

glm::vec3 view(0, 0, -1);
glm::vec3 up(0, 1, 0);
glm::vec3 eye(0,0,1.5f);
glm::vec3 center(0, 0.3, 0);
float fovy = 60;
float Rx = 0;
float Ry = 0;

cudaMat4 modelViewProj;
cudaMat4 inverseMVP;
glm::mat4 viewPort(-width/2.0, 0, 0, 0,  0, -height/2.0, 0, 0,  0, 0, 0.5, 0,  width/2.0, height/2.0, 0.5, 1.0);
//glm::mat4 viewPort(-1, 0, 0, 0,  0, -1, 0, 0,  0, 0, 1, 0,  width/2.0, height/2.0, 1, 1);
glm::vec3 lightPos(10, 10, 10);
glm::vec3 lightRGB(1, 1, 1);

//interaction
int mouseMode = 0;
enum MOUSEMODE{None, TransMode, RotateMode};
glm::vec2 lastMousePos(0, 0);
float translateStep = 1.0f/256.0f;

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

//------------------------------
//------- Helper ---------------
//------------------------------

//void computeProjection(float fovy, float aspect, float zNear, float zFar);
//void computeViewMat(glm::vec3 );

//------------------------------
//------- Interactive ----------
//------------------------------
void onMouseButton(int button, int state, int x, int y);

void onMouseDrag(int x, int y);

#endif