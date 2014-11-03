// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo, *device_vbow, *device_nbo;
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
  if(x<resolution.x && y<resolution.y){
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
__global__ void vertexShadeKernel(float* vbo, float* vbow, float* nbo, int vbosize, glm::mat4 MV, glm::mat4 P, glm::vec2 resolution, float zNear, float zFar){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  
	  int vertexIndex = index*3;

	  //world to clip
	  glm::vec4 worldSpace(vbo[vertexIndex], vbo[vertexIndex+1], vbo[vertexIndex+2], 1.0f);
	  glm::vec4 clipSpace = MV*worldSpace;

	  //save world space pos and norm
	  vbow[vertexIndex] = clipSpace.x;
	  vbow[vertexIndex+1] = clipSpace.y;
	  vbow[vertexIndex+2] = clipSpace.z;

	  glm::vec4 normal(nbo[vertexIndex], nbo[vertexIndex+1], nbo[vertexIndex+2], 0);
	  //normal = glm::normalize(glm::transpose(glm::inverse(MV))*normal);
	  //normal = glm::normalize(MV*normal);
	  nbo[vertexIndex] = normal.x;
	  nbo[vertexIndex+1] = normal.y;
	  nbo[vertexIndex+2] = normal.z;

	  clipSpace = P*clipSpace;

	  //clip to ndc
	  glm::vec3 deviceSpace;
	  deviceSpace.x = clipSpace.x / clipSpace.w;
	  deviceSpace.y = clipSpace.y / clipSpace.w;
	  deviceSpace.z = clipSpace.z / clipSpace.w;

	  //ndc to window
	  glm::vec3 windowSpace;
	  windowSpace.x = (clipSpace.x+1)*resolution.x/2.0f;
	  windowSpace.y = (-clipSpace.y+1)*resolution.y/2.0f;
	  windowSpace.z = clipSpace.z;
	  windowSpace.z = (zFar - zNear)/2.0f*clipSpace.z + (zNear + zFar)/2.0f;

	  //Store for primitive assembly
	  vbo[vertexIndex] = windowSpace.x;
	  vbo[vertexIndex+1] = windowSpace.y;
	  vbo[vertexIndex+2] = windowSpace.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, float* vbow, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, 
										int* ibo, int ibosize, triangle* primitives, glm::vec3 viewDir, glm::mat4 MV){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){

	  triangle tri;
	  
	  glm::vec3 pw0, pw1, pw2;

	  int vIndex = ibo[index*3];
	  tri.p0 = glm::vec3(vbo[vIndex*3], vbo[vIndex*3+1], vbo[vIndex*3+2]);
	  tri.n0 = glm::vec3(nbo[vIndex*3], nbo[vIndex*3+1], nbo[vIndex*3+2]);
	  tri.c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  tri.p0w = glm::vec3(vbow[vIndex*3], vbow[vIndex*3+1], vbow[vIndex*3+2]);

	  vIndex = ibo[index*3+1];
	  tri.p1 = glm::vec3(vbo[vIndex*3], vbo[vIndex*3+1], vbo[vIndex*3+2]);
	  tri.n1 = glm::vec3(nbo[vIndex*3], nbo[vIndex*3+1], nbo[vIndex*3+2]);
	  tri.c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  tri.p1w = glm::vec3(vbow[vIndex*3], vbow[vIndex*3+1], vbow[vIndex*3+2]);

	  vIndex = ibo[index*3+2];
	  tri.p2 = glm::vec3(vbo[vIndex*3], vbo[vIndex*3+1], vbo[vIndex*3+2]);
	  tri.n2 = glm::vec3(nbo[vIndex*3], nbo[vIndex*3+1], nbo[vIndex*3+2]);
	  tri.c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
	  tri.p2w = glm::vec3(vbow[vIndex*3], vbow[vIndex*3+1], vbow[vIndex*3+2]);
	  
	  bool isFrontFacing = false;
	  if (glm::dot(viewDir,tri.n0) >= 0) isFrontFacing = true;
	  else if (glm::dot(viewDir,tri.n1) >= 0) isFrontFacing = true;
	  else if (glm::dot(viewDir,tri.n2) >= 0) isFrontFacing = true;

	  glm::vec4 normal(tri.n0, 0);
	  normal = MV*normal;
	  tri.n0.x = normal.x;
	  tri.n0.y = normal.y;
	  tri.n0.z = normal.z;

	  normal = glm::vec4(tri.n1, 0);
	  normal = MV*normal;
	  tri.n1.x = normal.x;
	  tri.n1.y = normal.y;
	  tri.n1.z = normal.z;

	  normal = glm::vec4(tri.n2, 0);
	  normal = MV*normal;
	  tri.n2.x = normal.x;
	  tri.n2.y = normal.y;
	  tri.n2.z = normal.z;

	  tri.isFrontFacing = isFrontFacing;

	  primitives[index] = tri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  
	  triangle prim = primitives[index];

	  if (!prim.isFrontFacing) return;

	  glm::vec3 min, max;
	  getAABBForTriangle(prim, min, max);

	  int minX = floor(min.x);
	  int minY = floor(min.y);

	  if (minX < 0) minX = 0;
	  if (minY < 0) minY = 0;

	  int maxX = ceil(max.x);
	  int maxY = ceil(max.y);

	  for (int i=minX; i<maxX && i<resolution.x; i+=1){
		for (int j=minY; j<maxY && j<resolution.y; j+=1){

			  glm::vec3 baryCoords = calculateBarycentricCoordinate(prim, glm::vec2(i,j));
			  glm::vec3 baryCoordsW = calculateBarycentricCoordinateW(prim, glm::vec2(i,j));

			  if (isBarycentricCoordInBounds(baryCoords)){
				  float z = getZAtCoordinate(baryCoords, prim);

				  int fragmentIndex = j*resolution.x + i;
				  fragment frag = depthbuffer[fragmentIndex];

				  if (z < frag.position.z){
					  frag.position = glm::vec3(baryCoords.x,baryCoords.y,z);
					  frag.positionw = baryCoords.x*prim.p0 + baryCoords.y*prim.p1 + baryCoords.z*prim.p2;
					  frag.normal = baryCoords.x*prim.n0 + baryCoords.y*prim.n1 + baryCoords.z*prim.n2;
					  frag.color = baryCoords.x*prim.c0 + baryCoords.y*prim.c1 + baryCoords.z*prim.c2;
					  //frag.color = glm::vec3(1,1,0);
					  depthbuffer[fragmentIndex] = frag;
				  }
			  }
		  }
	  }

  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightPos){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  glm::vec3 pos = depthbuffer[index].positionw;
	  glm::vec3 col = depthbuffer[index].color;
	  glm::vec3 norm = depthbuffer[index].normal;
	  glm::vec3 l = glm::normalize(lightPos - pos);

	  float diff = glm::dot(norm,l);
	  if (diff<0.5) diff=0.5;
	  if (diff>1) diff=1;

	  depthbuffer[index].color.x = abs(norm.x);
	  depthbuffer[index].color.y = abs(norm.y);
	  depthbuffer[index].color.z = abs(norm.z);

	  depthbuffer[index].color = col*diff;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, camera cam){
  
  clock_t t;
  t = clock();

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
  frag.position = glm::vec3(0,0,10000);
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

  device_vbow = NULL;
  cudaMalloc((void**)&device_vbow, vbosize*sizeof(float));
  cudaMemcpy( device_vbow, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  glm::mat4 persp = glm::perspective(cam.fovy, cam.aspectRatio, cam.zNear, cam.zFar);
  glm::mat4 view = glm::lookAt(cam.eye, cam.center, cam.up);
  glm::mat4 model(1.0f);

  glm::mat4 MVP = persp*view*model;
  glm::mat4 MV = view*model;
  glm::mat4 P = persp;
  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbow, device_nbo, vbosize, MV, P, resolution, cam.zNear, cam.zFar);

  cudaDeviceSynchronize();
  checkCUDAError("Kernel failed1!");
  
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbow, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, glm::normalize(cam.center-cam.eye), MV);

  cudaDeviceSynchronize();
  checkCUDAError("Kernel failed2!");
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  checkCUDAError("Kernel failed3!");
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, glm::vec3(0,2.5,0));

  cudaDeviceSynchronize();
  checkCUDAError("Kernel failed4!");
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed5!");
  t = clock() - t;
  //std::cout<<(float)t/CLOCKS_PER_SEC<<std::endl;
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_vbow );
  cudaFree( device_nbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

