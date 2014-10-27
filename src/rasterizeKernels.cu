// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_nbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

__host__ __device__ void screenToNDC(int x, int resolution, float* ndcX) {
  *ndcX = - 2 * (x / (float)resolution - 0.5f);
}

__host__ __device__ void ndcToScreen(float ndcX, int resolution, int* x) {
  *x = -(ndcX - 1) * resolution / 2;
}

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
  a = (a+0x7ed55d16) + (a<<12);
  a = (a^0xc761c23c) ^ (a>>19);
  a = (a+0x165667b1) + (a<<5);
  a = (a+0xd3a2646c) ^ (a<<9);
  a = (a+0xfd7046c5) + (a<<3);
  a = (a^0xb55a4f09) ^ (a>>16);
  return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x>0 && x<resolution.x && y>0 && y<resolution.y){
    int index = (y*resolution.x) + x;
    depthbuffer[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
    image[index] = color;
  }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
    fragment f = frag;
    f.position.x = x;
    f.position.y = y;
    buffer[index] = f;
  }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){

    glm::vec3 color;      
    color.x = image[index].x*255.0;
    color.y = image[index].y*255.0;
    color.z = image[index].z*255.0;

    if(color.x>255){
      color.x = 255;
    }

    if(color.y>255){
      color.y = 255;
    }

    if(color.z>255){
      color.z = 255;
    }

    // Each thread writes one pixel location in the texture (textel)
    PBOpos[index].w = 0;
    PBOpos[index].x = color.x;     
    PBOpos[index].y = color.y;
    PBOpos[index].z = color.z;
  }
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 mvp){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
    glm::vec4 p (vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2], 1);
    p = mvp * p;
    vbo[index * 3] = p.x / p.w;
    vbo[index * 3 + 1] = p.y / p.w;
    vbo[index * 3 + 2] = p.z / p.w;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
    triangle tri;
    int i = ibo[index * 3] * 3;
    tri.p0 = glm::vec3(vbo[i], vbo[i + 1], vbo[i + 2]);
    tri.n0 = glm::vec3(nbo[i], nbo[i + 1], nbo[i + 2]);
    tri.c0 = glm::vec3(cbo[i], cbo[i + 1], cbo[i + 2]);
    i = ibo[index * 3 + 1] * 3;
    tri.p1 = glm::vec3(vbo[i], vbo[i + 1], vbo[i + 2]);
    tri.n1 = glm::vec3(nbo[i], nbo[i + 1], nbo[i + 2]);
    tri.c1 = glm::vec3(cbo[i], cbo[i + 1], cbo[i + 2]);
    i = ibo[index * 3 + 2] * 3;
    tri.p2 = glm::vec3(vbo[i], vbo[i + 1], vbo[i + 2]);
    tri.n2 = glm::vec3(nbo[i], nbo[i + 1], nbo[i + 2]);
    tri.c2 = glm::vec3(cbo[i], cbo[i + 1], cbo[i + 2]);
    /*glm::vec3 normal = glm::normalize(glm::cross(tri.p1 - tri.p0, tri.p2 - tri.p0) +
                                      glm::cross(tri.p2 - tri.p1, tri.p0 - tri.p1) +
                                      glm::cross(tri.p0 - tri.p2, tri.p1 - tri.p2));
    tri.n0 = tri.n1 = tri.n2 = normal;*/
    primitives[index] = tri;
  }
}

__global__ void backfaceCullingKernel(triangle* primitives, int primitivesCount) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){

  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
    // Draw Vertices
    /*
    int x, y;
    ndcToScreen(primitives[index].p0.x, resolution.x, &x);
    ndcToScreen(primitives[index].p0.y, resolution.y, &y);
            
    fragment frag;
    frag.color = primitives[index].c0;
    frag.normal = primitives[index].n0;
    frag.position = primitives[index].p0;
    writeToDepthbuffer(x, y, frag, depthbuffer, resolution);
    
    ndcToScreen(primitives[index].p1.x, resolution.x, &x);
    ndcToScreen(primitives[index].p1.y, resolution.y, &y);
            
    frag.color = primitives[index].c1;
    frag.normal = primitives[index].n1;
    frag.position = primitives[index].p1;
    writeToDepthbuffer(x, y, frag, depthbuffer, resolution);
    
    ndcToScreen(primitives[index].p2.x, resolution.x, &x);
    ndcToScreen(primitives[index].p2.y, resolution.y, &y);
            
    frag.color = primitives[index].c2;
    frag.normal = primitives[index].n2;
    frag.position = primitives[index].p2;
    writeToDepthbuffer(x, y, frag, depthbuffer, resolution);
    */

    // Draw Faces
    triangle t = primitives[index];
    point top, middle, bottom;
    top.position = t.p0;
    middle.position = t.p1;
    bottom.position = t.p2;
    top.color = t.c0;
    middle.color = t.c1;
    bottom.color = t.c2;
    top.normal = t.n0;
    middle.normal = t.n1;
    bottom.normal = t.n2;

    point temp;

    // Do a basic bubble sort
    for (int i = 0; i < 2; i++) {
      if (bottom.position.y > middle.position.y) {
        temp = bottom;
        bottom = middle;
        middle = temp;
      }
      if (middle.position.y > top.position.y) {
        temp = middle;
        middle = top;
        top = temp;
      }
    }

    // Ignore triangle if it's outside
    // TODO: move this to a clipper later, with the x and z clipping as well.
    if (top.position.y < -1 || bottom.position.y > 1) {
      return;
    }

    // "left" and "right" are relative to each other, not top.
    point pointLeft, pointRight;    // used for interpolation

    if (bottom.position.x > middle.position.x) {  // top->middle is on the left
      pointLeft = middle;
      pointRight = bottom;
    } else {        // top->bottom is on left
      pointLeft = bottom;
      pointRight = middle;
    }
    
    float currNDCx = top.position.x;
    float currNDCy = top.position.y;
    int currY, currX;
    ndcToScreen(currNDCx, resolution.x, &currX);
    ndcToScreen(currNDCy, resolution.y, &currY);

    while (currNDCy > middle.position.y && currNDCy > -1) {
      // only perform these operations if the current y coordinate is in the screen.
      if (currNDCy <= 1) {
        // interpolate along the edges
        float tLeft = (top.position.y - currNDCy) / (top.position.y - pointLeft.position.y);
        if (top.position.y == pointLeft.position.y) {
          tLeft = 0;
        }
        float tRight = (top.position.y - currNDCy) / (top.position.y - pointRight.position.y);
        if (top.position.y == pointRight.position.y) {
          tRight = 0;
        }

        // Would saving 1-tleft and 1-tright into variables save 1 cycle per statement (if they fit in registers?)
        glm::vec3 cLeft, cRight, pLeft, pRight, nLeft, nRight;
        pLeft = (1 - tLeft) * top.position + tLeft * pointLeft.position;
        nLeft = (1 - tLeft) * top.normal + tLeft * pointLeft.normal;
        cLeft = (1 - tLeft) * top.color + tLeft * pointLeft.color;
        pRight = (1 - tRight) * top.position + tRight * pointRight.position;
        nRight = (1 - tRight) * top.normal + tRight * pointRight.normal;
        cRight = (1 - tRight) * top.color + tRight * pointRight.color;
        int rBound = 0;
        int lBound = 0;
        ndcToScreen(pRight.x, resolution.x, &rBound);
        ndcToScreen(pLeft.x, resolution.x, &lBound);
        for (currX = lBound; currX >= rBound; currX--) {
          if (currX >= 0 && currX < resolution.x) {
            screenToNDC(currX, resolution.x, &currNDCx);
            // interpolate color, normal, and position
            float t = (currX - lBound) / (rBound - lBound);
            if (pRight.x == pLeft.x) {
              t = 0;
            }
            fragment frag;
            frag.position = (1 - t) * pLeft + t * pRight;
            frag.normal = (1 - t) * nLeft + t * nRight;
            frag.color = (1 - t) * cLeft + t * cRight;
            writeToDepthbuffer(currX, currY, frag, depthbuffer, resolution);
          }
        }
      }
      currY++;
      screenToNDC(currY, resolution.y, &currNDCy);
    }
    
    if (middle.position.x < top.position.x) {
      pointLeft = middle;
      pointRight = top;
    } else {
      pointLeft = top;
      pointRight = middle;
    }

    while (currNDCy > bottom.position.y && currNDCy > -1) {
      // only perform these operations if the current y coordinate is in the screen.
      if (currNDCy <= 1) {
        // interpolate along the edges
        float tLeft = (pointLeft.position.y - currNDCy) / (pointLeft.position.y - bottom.position.y);
        if (pointLeft.position.y == bottom.position.y) {
          tLeft = 0;
        }
        float tRight = (pointRight.position.y - currNDCy) / (pointRight.position.y - bottom.position.y);
        if (pointRight.position.y == bottom.position.y) {
          tRight = 0;
        }

        // Would saving 1-tleft and 1-tright into variables save 1 cycle per statement (if they fit in registers?)
        glm::vec3 cLeft, cRight, pLeft, pRight, nLeft, nRight;
        pLeft = (1 - tLeft) * pointLeft.position + tLeft * bottom.position;
        nLeft = (1 - tLeft) * pointLeft.normal + tLeft * bottom.normal;
        cLeft = (1 - tLeft) * pointLeft.color + tLeft * bottom.color;
        pRight = (1 - tRight) * pointRight.position + tRight * bottom.position;
        nRight = (1 - tRight) * pointRight.normal + tRight * bottom.normal;
        cRight = (1 - tRight) * pointRight.color + tRight * bottom.color;

        int rBound = 0;
        int lBound = 0;
        ndcToScreen(pRight.x, resolution.x, &rBound);
        ndcToScreen(pLeft.x, resolution.x, &lBound);
        for (currX = lBound; currX >= rBound; currX--) {
          if (currX >= 0 && currX < resolution.x) {
            screenToNDC(currX, resolution.x, &currNDCx);
            // interpolate color, normal, and position
            float t = (currX - lBound) / (rBound - lBound);
            if (pRight.x == pLeft.x) {
              t = 0;
            }
            fragment frag;
            frag.position = (1 - t) * pLeft + t * pRight;
            frag.normal = (1 - t) * nLeft + t * nRight;
            frag.color = (1 - t) * cLeft + t * cRight;
            writeToDepthbuffer(currX, currY, frag, depthbuffer, resolution);
          }
        }
      }
      currY++;
      screenToNDC(currY, resolution.y, &currNDCy);
    }
  }
}

//TODO: Implement a fragment shader
// Modifies the .color value per fragment.
// Simple Blinn-Phong shading, light needs to be transformed into clip coordinates.
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, light light, glm::mat4 matVPinv){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
    fragment f = depthbuffer[index];
    if (f.position.z > 0) { //ignore all the empty space (z = -10000)
      glm::vec4 origPos = matVPinv * glm::vec4(f.position,1);
      float diffuse = glm::dot(f.normal, glm::normalize(light.position - glm::vec3(origPos / origPos.w)));
      if (diffuse < 0) {
        diffuse = 0;
      }
      depthbuffer[index].color *= light.color * diffuse;
    }
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    // Color
    framebuffer[index] = depthbuffer[index].color;
    // Normal
    //framebuffer[index] = depthbuffer[index].normal;
    //framebuffer[index] = glm::normalize(glm::vec3(depthbuffer[index].normal.r, depthbuffer[index].normal.g, 0));
    // Distance
    //framebuffer[index] = glm::vec3(depthbuffer[index].position.z);
  }
}

struct clippingOrBackface {
  __host__ __device__
    bool operator() (const triangle t) {
      glm::vec3 normal = glm::normalize(glm::cross(t.p1 - t.p0, t.p2 - t.p0) +
                                      glm::cross(t.p2 - t.p1, t.p0 - t.p1) +
                                      glm::cross(t.p0 - t.p2, t.p1 - t.p2));
      return (normal.z < 0) || ((t.p0.x > 1 || t.p0.x < -1) && (t.p0.y > 1 || t.p0.y < -1) &&
                              (t.p1.x > 1 || t.p1.x < -1) && (t.p1.y > 1 || t.p1.y < -1) &&
                              (t.p2.x > 1 || t.p2.x < -1) && (t.p2.y > 1 || t.p2.y < -1));
  }
};

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));

  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));

  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //camera setup
  //------------------------------
  glm::vec3 eye (0, 0, 3);
  glm::vec3 center (0, 0, 0);
  glm::vec3 up (0, 1, 0);
  glm::mat4 matView = glm::lookAt(eye, center, up);
  float fovy, aspect, znear, zfar;
  fovy = 45;
  aspect = 1.0;
  znear = .01;
  zfar = 5;
  glm::mat4 matProj = glm::perspective(fovy, aspect, znear, zfar);

  glm::mat4 matVP = matProj * matView;

  //----------------------------
  //light setup
  //----------------------------
  light light;
  light.color = glm::vec3(1, 1, 1);
  light.position = glm::vec3(5, 0, 0);

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, matVP);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaDeviceSynchronize();

  //------------------------------
  //ez backface culling and clipping
  //------------------------------
	thrust::device_ptr<triangle> primitivesStart(primitives);

	float numRemoved = thrust::count_if(primitivesStart, primitivesStart + ibosize / 3, clippingOrBackface());
	thrust::remove_if(primitivesStart, primitivesStart + ibosize / 3, clippingOrBackface());
	float numPrimitives = ibosize / 3 - numRemoved;
  primitiveBlocks = ceil(((float)numPrimitives)/((float)tileSize));

  cudaDeviceSynchronize();
  //float numPrimitives = ibosize/3;
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, numPrimitives, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, light, matVP);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_nbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

