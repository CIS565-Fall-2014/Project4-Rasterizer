// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

//Additional Objects I think would be useful
struct camera{
  glm::vec3 position;
  glm::vec3 up;
  glm::vec3 forward;
  float fovy;
  float nearClip;
  float farClip;
  camera(){//initialize camera to default values
    position = glm::vec3(1,1,1);
    up       = glm::vec3(0,-1,0);
    forward  = glm::vec3(0,0,0);
    fovy     = 45.0f;
    nearClip = 0.1f;
    farClip  = 8.0f;
  }
};


void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, camera cam);



#endif //RASTERIZEKERNEL_H
