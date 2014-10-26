// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives1;
triangle* primitives2;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
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

//Handy function for clamping between two values;
__host__ __device__ float clamp(float val, float min, float max) {
  float result = val;
  if (val < min) {
    val = min;
  } else if (val > max) {
    val = max;
  }
  return val;
}

//Handy function for reflection
__host__ __device__ glm::vec3 reflect(glm::vec3 vec_in, glm::vec3 norm) {
  return (vec_in - 2.0f*glm::dot(vec_in, norm)*norm);
}

//Writes a given fragment to a fragment buffer at a given location
/*__device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  int index = x + (y * resolution.x);
  if(x<resolution.x && y<resolution.y){
    //Depth comparison
    if (frag.position.z > 0.0f || frag.position.z < depthbuffer[index].position.z) {
      return;
    }

    //Add the new fragment to the buffer
    depthbuffer[index] = frag;
  }
}*/

//Reads the fragment from a given location in a depth buffer
/*__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}*/

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

//Kernel that clears a given depth buffer
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

//A vertex shader that does model-view-projection transformation
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, glm::mat4 mvp){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
    //Pull together a vertex
    glm::vec4 vert(vbo[3*index], vbo[3*index+1], vbo[3*index+2], 1.0f);

    //Transform the vertex
    glm::vec4 new_vert = mvp*vert;

    //Store it back in the vbo
    vbo[3*index] = new_vert.x/new_vert.w;
    vbo[3*index+1] = new_vert.y/new_vert.w;
    vbo[3*index+2] = new_vert.z/new_vert.w;

    //Pull together a normal
    glm::vec4 norm(nbo[3*index], nbo[3*index+1], nbo[3*index+2], 1.0f);

    //Transform the normal
    glm::vec4 new_norm = mvp*norm;

    //Store it back in the nbp
    nbo[3*index] = new_norm.x/new_norm.w;
    nbo[3*index+1] = new_norm.y/new_norm.w;
    nbo[3*index+2] = new_norm.z/new_norm.w;
  }
}

//Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  // Add vertices
    glm::vec3 p0, p1, p2;
    p0.x = vbo[3 * ibo[3 * index]];
    p0.y = vbo[3 * ibo[3 * index] + 1];
    p0.z = vbo[3 * ibo[3 * index] + 2];
    p1.x = vbo[3 * ibo[3 * index + 1]];
    p1.y = vbo[3 * ibo[3 * index + 1] + 1];
    p1.z = vbo[3 * ibo[3 * index + 1] + 2];
    p2.x = vbo[3 * ibo[3 * index + 2]];
    p2.y = vbo[3 * ibo[3 * index + 2] + 1];
    p2.z = vbo[3 * ibo[3 * index + 2] + 2];
    primitives[index].p0 = p0;
    primitives[index].p1 = p1;
    primitives[index].p2 = p2;

	  // Add color
	  primitives[index].c0.x = cbo[3*ibo[3*index]];
	  primitives[index].c0.y = cbo[3*ibo[3*index]+1];
	  primitives[index].c0.z = cbo[3*ibo[3*index]+2];
	  primitives[index].c1.x = cbo[3*ibo[3*index+1]];
	  primitives[index].c1.y = cbo[3*ibo[3*index+1]+1];
	  primitives[index].c1.z = cbo[3*ibo[3*index+1]+2];
	  primitives[index].c2.x = cbo[3*ibo[3*index+2]];
	  primitives[index].c2.y = cbo[3*ibo[3*index+2]+1];
	  primitives[index].c2.z = cbo[3*ibo[3*index+2]+2];

    // Add normal
    primitives[index].n.x = (nbo[3 * ibo[3 * index]] + nbo[3 * ibo[3 * index+1]] + nbo[3 * ibo[3 * index+2]])/3.0f;
    primitives[index].n.y = (nbo[3 * ibo[3 * index] + 1] + nbo[3 * ibo[3 * index+1] + 1] + nbo[3 * ibo[3 * index+2] + 1])/3.0f;
    primitives[index].n.z = (nbo[3 * ibo[3 * index] + 2] + nbo[3 * ibo[3 * index+1] + 2] + nbo[3 * ibo[3 * index+2] + 2])/3.0f;
    primitives[index].n = glm::normalize(primitives[index].n);
  }
}

//Backface culling function
__host__ __device__ bool isTriangleBackfacing(triangle tri) {
  float x1 = tri.p1.x - tri.p0.x;
  float y1 = tri.p1.y - tri.p0.y;
  float x2 = tri.p2.x - tri.p0.x;
  float y2 = tri.p2.y - tri.p0.y;

  return ((x1*y2 - y1*x2) > 0.0f);
}

//Thrust predicate for triangle removal
struct check_triangle {
  __host__ __device__
    bool operator() (const triangle& t) {
      bool back = isTriangleBackfacing(t);
      bool p0 = (t.p0.x < -1.0f || t.p0.x > 1.0f) || (t.p0.x < -1.0f || t.p0.x > 1.0f) || (t.p0.z < 0.0f);
      bool p1 = (t.p1.x < -1.0f || t.p1.x > 1.0f) || (t.p1.x < -1.0f || t.p1.x > 1.0f) || (t.p1.z < 0.0f);
      bool p2 = (t.p2.x < -1.0f || t.p2.x > 1.0f) || (t.p2.x < -1.0f || t.p2.x > 1.0f) || (t.p2.z < 0.0f);
      bool outside = (p0 && p1 && p2);
      return !(back || outside);
    }
};

//Kernel to trim primitives before rasterization
__host__ void culling(triangle* primitives, triangle* new_primitives, int& numPrimitives) {
  thrust::device_ptr<triangle> in = thrust::device_pointer_cast<triangle>(primitives);
  thrust::device_ptr<triangle> out = thrust::device_pointer_cast<triangle>(new_primitives);
  numPrimitives = thrust::copy_if(in, in + numPrimitives, out, check_triangle()) - out;
}

//Function for sorting a triangle's vertices in ascending y
__host__ __device__ void sortTriangleOnY(triangle in, triangle& out) {
  bool i01 = (in.p0.y < in.p1.y);
  bool i02 = (in.p0.y < in.p2.y);
  bool i12 = (in.p1.y < in.p2.y);
  if (i01 && i02) {
    //0 is min
    out.p0 = in.p0;
    if (i12) {
      out.p1 = in.p1;
      out.p2 = in.p2;
    }
    else {
      out.p1 = in.p2;
      out.p2 = in.p1;
    }
  }
  else if (!i01 && i12) {
    //1 is min
    out.p0 = in.p1;
    if (i02) {
      out.p1 = in.p0;
      out.p2 = in.p2;
    }
    else {
      out.p1 = in.p2;
      out.p2 = in.p0;
    }
  }
  else {
    //2 is min
    out.p0 = in.p2;
    if (i01) {
      out.p1 = in.p0;
      out.p2 = in.p1;
    }
    else {
      out.p1 = in.p1;
      out.p2 = in.p0;
    }
  }
}

__host__ __device__ void fillBottomTriangle(triangle t, fragment* depthbuffer, glm::vec2 resolution) {

  float denom = (t.p2.y - t.p0.y);
  float invslope1x = (abs(denom)>=1.0f) ? (t.p2.x - t.p0.x) / denom : 0.0f;
  float invslope1z = (abs(denom)>=1.0f) ? (t.p2.z - t.p0.z) / denom : 0.0f;
  float invslope2x = (abs(denom)>=1.0f) ? (t.p2.x - t.p1.x) / denom : 0.0f;
  float invslope2z = (abs(denom)>=1.0f) ? (t.p2.z - t.p1.z) / denom : 0.0f;

  float curx1 = t.p2.x;
  float curx2 = t.p2.x;
  float curz1 = t.p2.z;
  float curz2 = t.p2.z;

  //Loop over scanlines from bottom to top
  for (int y = ((int) (t.p2.y+0.5f)); y >= ((int) (t.p0.y+0.5f)); y--) {

    curx1 -= invslope1x;
    curx2 -= invslope2x;
    curz1 -= invslope1z;
    curz2 -= invslope2z;

    float z = curz1;
    denom = (curx2 - curx1);
    float invslopezx = (abs(denom) >= 0.75f) ? (curz2 - curz1) / denom : 0.0f;

    //Draw scanline
    for (int x = (int) (min(curx1,curx2)+0.5f); x <= (int) (max(curx1,curx2)+0.5f); x++) {
      //Make sure the pixel is in the screen
      if (x < 0 || x >= resolution.x || y < 0 || y >= resolution.y) {
        continue;
      }

      //Make the fragment
      fragment frag;
      frag.position.x = ((float)x) * 2.0 / resolution.x + 1.0f;
      frag.position.y = ((float)y) * 2.0 / resolution.y + 1.0f;
      frag.position.z = z;
      frag.normal = t.n;

      //Depth comparison
      int index = x + (y * resolution.x);
      if (z < 0.0f || z < depthbuffer[index].position.z) {
        continue;
      }
      
      //Add the new fragment to the buffer
      depthbuffer[index] = frag;

      z -= invslopezx;
    }

  }
}

__host__ __device__ void fillTopTriangle(triangle t, fragment* depthbuffer, glm::vec2 resolution) {

  float denom = (t.p1.y - t.p0.y);
  float invslope1x = (abs(denom)>=1.0f) ? (t.p1.x - t.p0.x) / denom : 0.0f;
  float invslope1z = (abs(denom)>=1.0f) ? (t.p1.z - t.p0.z) / denom : 0.0f;
  float invslope2x = (abs(denom)>=1.0f) ? (t.p2.x - t.p0.x) / denom : 0.0f;
  float invslope2z = (abs(denom)>=1.0f) ? (t.p2.z - t.p0.z) / denom : 0.0f;

  float curx1 = t.p0.x;
  float curx2 = t.p0.x;
  float curz1 = t.p0.z;
  float curz2 = t.p0.z;

  for (int y = ((int) (t.p0.y+0.5f)); y <= ((int) (t.p2.y+0.5f)); y++) {
    float z = curz1;
    denom = (curx2 - curx1);
    float invslopezx = (abs(denom) >= 0.75f) ? (curz2 - curz1) / denom : 0.0f;

    for (int x = (int) (min(curx1,curx2)+0.5f); x <= (int) (max(curx1,curx2)+0.5f); x++) {
      //Make sure the pixel is in the screen
      if (x < 0 || x >= resolution.x || y < 0 || y >= resolution.y) {
        continue;
      }

      //Make the fragment
      fragment frag;
      frag.position.x = ((float)x) * 2.0 / resolution.x + 1.0f;
      frag.position.y = ((float)y) * 2.0 / resolution.y + 1.0f;
      frag.position.z = z;
      frag.normal = t.n;

      //Depth comparison
      int index = x + (y * resolution.x);
      if (z < 0.0f || z < depthbuffer[index].position.z) {
        continue;
      }

      //Add the new fragment to the buffer
      depthbuffer[index] = frag;

      z += invslopezx;
    }
    
    curx1 += invslope1x;
    curx2 += invslope2x;
    curz1 += invslope1z;
    curz2 += invslope2z;
  }
}

//Scanline algorithm for rasterization (inspired by http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html)
__host__ __device__ void scanlineTriangle(triangle tri, fragment* depthbuffer, glm::vec2 resolution) {
  //Sort the vertices by descending y
  triangle sorted_tri;
  sortTriangleOnY(tri, sorted_tri);

  //Determine the 4th point to split on
  glm::vec3 p3 = sorted_tri.p1;
  float denom = (sorted_tri.p2.y - sorted_tri.p0.y);
  float t = (abs(denom)>=0.75f) ? (sorted_tri.p1.y - sorted_tri.p0.y) / denom : 0.0f;
  p3.x = sorted_tri.p0.x + t * (sorted_tri.p2.x - sorted_tri.p0.x);
  p3.z = sorted_tri.p0.z + t * (sorted_tri.p2.z - sorted_tri.p0.z);

  //Build the two triangles
  triangle top;
  top.p0 = sorted_tri.p0;
  top.p1 = sorted_tri.p1;
  top.p2 = p3;
  top.n = tri.n;
  triangle bottom;
  bottom.p0 = sorted_tri.p1;
  bottom.p1 = p3;
  bottom.p2 = sorted_tri.p2;
  bottom.n = tri.n;

  //Fill in the two triangles
  fillTopTriangle(top, depthbuffer, resolution);
  fillBottomTriangle(bottom, depthbuffer, resolution);
}

//A rasterization kernel for triangle primitives
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
    //Copy the triangle locally for quick access
    triangle this_tri = primitives[index];

    //Scale the triangle into window coordinates
    this_tri.p0.x = resolution.x*(1.0f - this_tri.p0.x)/2;
    this_tri.p1.x = resolution.x*(1.0f - this_tri.p1.x)/2;
    this_tri.p2.x = resolution.x*(1.0f - this_tri.p2.x)/2;
    this_tri.p0.y = resolution.y*(1.0f - this_tri.p0.y)/2;
    this_tri.p1.y = resolution.y*(1.0f - this_tri.p1.y)/2;
    this_tri.p2.y = resolution.y*(1.0f - this_tri.p2.y)/2;

    //Generate fragments from a scanline function
    scanlineTriangle(this_tri, depthbuffer, resolution);
  }
}

//Phong shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, ray light){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
    if (depthbuffer[index].position.z > -10000.0f) {
      //Store the fragment info locally for accessibility
      glm::vec3 V = depthbuffer[index].position;
      glm::vec3 N = depthbuffer[index].normal;

      //Compute necessary vectors
      glm::vec3 L = glm::normalize(light.origin - V);
      glm::vec3 E = glm::normalize(-V);
      glm::vec3 R = glm::normalize(reflect(-L,N));

      //Shininess
      float specPow = 4.0f;

      //Green (TODO: read from material)
      glm::vec3 green(0.0f, 1.0f, 0.0f);

      //Compute lighting
      glm::vec3 ambient = 0.1f * green;
      glm::vec3 diffuse = 0.45f * clamp(glm::dot(N, L), 0.0f, 1.0f) * green;
      glm::vec3 specular = 0.45f * clamp(pow(max(glm::dot(R,E), 0.0f), specPow), 0.0f, 1.0f) * green;
      depthbuffer[index].color = ambient + diffuse + specular;

      //depthbuffer[index].color = green;
    }
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, ray light, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, glm::mat4 mvp){

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
  fragment frag;
  frag.color = glm::vec3(0, 0, 0);
  frag.normal = glm::vec3(0, 0, 0);
  frag.position = glm::vec3(0, 0, -10000.0f);
  clearDepthBuffer << <fullBlocksPerGrid, threadsPerBlock >> >(resolution, depthbuffer, frag);

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));

  //------------------------------
  //memory stuff
  //------------------------------
  primitives1 = NULL;
  cudaMalloc((void**)&primitives1, (ibosize/3)*sizeof(triangle));
  primitives2 = NULL;
  cudaMalloc((void**)&primitives2, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy(device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
  int numPrimitives = ibosize/3;
  
  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, mvp);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)numPrimitives)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives1);

  cudaDeviceSynchronize();

  //------------------------------
  //culling
  //------------------------------
  culling(primitives1, primitives2, numPrimitives);
  primitiveBlocks = ceil(((float)numPrimitives)/((float)tileSize));

  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives2, numPrimitives, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------ 
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, light);

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
  cudaFree( primitives1 );
  cudaFree( primitives2 );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( device_nbo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

