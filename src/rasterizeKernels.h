// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"


struct light{
	glm::vec3 position;
	glm::vec3 diffColor;
	glm::vec3 specColor;
	int specExp;
	glm::vec3 ambColor;
};

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, glm::mat4 glmViewTransform, glm::mat4 glmProjectionTransform, glm::mat4 glmMVtransform,light Light);

#endif //RASTERIZEKERNEL_H
