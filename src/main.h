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
#include "glm\gtc\matrix_transform.hpp"

#define MOUSE_SPEED 1.2*0.0001f
#define ZOOM_SPEED 3
#define MIDDLE_SPEED 3

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
vector <obj*> meshes;

int mode=0;
bool barycenter = false;

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
glm::vec3 eye(0.0f, 0.8f, 3.0f);
glm::vec3 center(0.0f, 0.4f, 0.0f);
float zNear = 0.001;
float zFar = 10000;
glm::mat4 projection = glm::perspective(60.0f, (float)(width) / (float)(height), zNear, zFar);
glm::mat4 model = glm::mat4();
glm::mat4 view = glm::lookAt(eye,center , glm::vec3(0, 1, 0));
glm::mat4 modelview = view * glm::mat4();
glm::vec3 lightpos = glm::vec3(0, 2.0f, 2.0f);

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
bool RB = false;
bool MB = false;
bool inwindow = false;
void MouseClickCallback(GLFWwindow *window, int button, int action, int mods);
void CursorCallback(GLFWwindow *window, double x,double y);
void CursorEnterCallback(GLFWwindow *window,int entered);
//Make the camera move on the surface of sphere
float vPhi = 0.0f;
float vTheta = 3.14105926f/2.0f;
float R = glm::length(eye);

#endif