// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
//#include "rasterizeTools.h"
#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "rasterizeTools.h"

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, Camera cam, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,float* nbo, int nbosize);

#endif //RASTERIZEKERNEL_H
