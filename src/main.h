// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

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
Camera cam(glm::vec2(width, height));
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

//-------------------------------
//----------Mouse Control--------
//-------------------------------

double mouse_old_x, mouse_old_y;
unsigned char button_mask = 0x00;

float viewPhi = 0.0f;
float viewTheta = PI/2.0f;
float r = glm::length(cam.pos);

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
void mouseClick(GLFWwindow* window, int button, int action, int mods);
void mouseMotion(int x, int y);
//void mouseWheel(int button, int dir, int x, int y);
void mouseWheel(GLFWwindow* window, double x, double y);
#endif