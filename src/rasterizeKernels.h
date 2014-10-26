// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "cudaMat4.h"

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, 
	int* ibo, int ibosize, float* nbo, int nbosize, cudaMat4 shaderMatrix, int translateX, int translateY, glm::vec3 eye, 
	glm::vec3 light, bool alphaBlend, float alphaValue, bool backCulling, bool scissorTest, bool antialiasing, int displayMode);

#endif //RASTERIZEKERNEL_H
