// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"

void kernelCleanup();
void cudaRasterizeCore(
        uchar4* PBOpos, glm::vec2 resolution, float frame,
        int vbosize, float *pbo, float *nbo, float *cbo,
        int* ibo, int ibosize);

#endif //RASTERIZEKERNEL_H
