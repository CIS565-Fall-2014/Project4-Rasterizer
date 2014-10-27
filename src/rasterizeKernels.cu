// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "thrust\device_ptr.h"
#include "thrust\remove.h"

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, cudaMat4 modelView){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec3 v(vbo[index*3], vbo[index*3+1], vbo[index*3+2]);

	  v = multiplyMV(modelView, glm::vec4(v, 1.0));

	  //v = modelView * v;
	  //v /= v.w;
	  //glm::vec4 v2 = viewPort * glm::vec4(v,1.0);

	  vbo[index*3] = v.x;
	  vbo[index*3+1] = v.y;
	  vbo[index*3+2] = v.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives, 
										bool backFaceCulling, glm::vec3 viewDir){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int vi1 = ibo[index*3];
	  int vi2 = ibo[index*3+1];
	  int vi3 = ibo[index*3+2];

	  primitives[index].p0 = glm::vec3(vbo[vi1*3], vbo[vi1*3+1], vbo[vi1*3+2]);
	  primitives[index].p1 = glm::vec3(vbo[vi2*3], vbo[vi2*3+1], vbo[vi2*3+2]);
	  primitives[index].p2 = glm::vec3(vbo[vi3*3], vbo[vi3*3+1], vbo[vi3*3+2]);

	  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);

	  glm::vec3 n1(nbo[vi1*3], nbo[vi1*3+1], nbo[vi1*3+2]);
	  glm::vec3 n2(nbo[vi2*3], nbo[vi2*3+1], nbo[vi2*3+2]);
	  glm::vec3 n3(nbo[vi3*3], nbo[vi3*3+1], nbo[vi3*3+2]);

	  primitives[index].n = glm::normalize((n1+n2+n3) / 3.0f);

	  //back-face culling
	  if (backFaceCulling)
	  {
		  if (glm::dot(primitives[index].n, viewDir) > 0)
			  primitives[index].n = glm::vec3(0,0,0);
	  }
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index<primitivesCount){
	//  if (primitives[index].n.x != 0 || primitives[index].n.y != 0 || primitives[index].n.z != 0)
	//  {
		  triangle primitive = primitives[index];
		  glm::vec3 minpoint, maxpoint;
		  getAABBForTriangle(primitive, minpoint, maxpoint);

		  minpoint.x = (minpoint.x < 0) ? 0 : minpoint.x;
		  maxpoint.x = (maxpoint.x > resolution.x) ? resolution.x : maxpoint.x;
		  minpoint.y = (minpoint.y < 0) ? 0 : minpoint.y;
		  maxpoint.y = (maxpoint.y > resolution.y) ? resolution.y : maxpoint.y;

		  for (int y = minpoint.y; y <= maxpoint.y; y++)
		  {
			  for(int x = minpoint.x; x <= maxpoint.x; x++)
			  {
				int depthbufferId = x + y * resolution.x;

				glm::vec2 center(x+0.5, y+0.5);
				glm::vec3 barycentricP = calculateBarycentricCoordinate(primitive, center);

				float z = getZAtCoordinate(barycentricP, primitive);

				//wait until unlock
				while (!depthbuffer[depthbufferId].isLocked)
					depthbuffer[depthbufferId].isLocked = true;

				if (z > depthbuffer[depthbufferId].position.z  && isBarycentricCoordInBounds(barycentricP))
				{
					depthbuffer[depthbufferId].position = primitive.p0 * barycentricP.x + primitive.p1 * barycentricP.y + primitive.p2 * barycentricP.z;
					depthbuffer[depthbufferId].normal = primitive.n;
					depthbuffer[depthbufferId].color = primitive.c0 * barycentricP.x + primitive.c1 * barycentricP.y + primitive.c2 * barycentricP.z;
				}
				depthbuffer[depthbufferId].isLocked = false;

			  }
		  }
	//  }
	//  else;
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightPos, glm::vec3 lightRGB){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	  fragment frag = depthbuffer[index];

	  glm::vec3 incident = glm::normalize(frag.position - lightPos);
	  float cosI = glm::dot(incident, frag.normal);
	  cosI = cosI > 0.0 ? 0.0 : -cosI;
	  frag.color = frag.color * cosI * lightRGB * 0.7f + frag.color * 0.3f;

	  /*frag.color.r = abs(frag.normal.x);
	  frag.color.g = abs(frag.normal.y);
	  frag.color.b = abs(frag.normal.z);*/

	  depthbuffer[index] = frag;
  }
}

__global__ void antiAliasKernel(fragment* depthbuffer, glm::vec2 resolution){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		glm::vec3 newColor;
		if (x >= 1 && y >= 1 && x<=resolution.x-1 && y<=resolution.y-1) {
			for (int i = x-1; i <= x+1; i++){
				for (int j = y-1; j <= y+1; j++){
					int id = i + (j * resolution.x);
					newColor += depthbuffer[id].color;
				}
			}
			newColor /= 9.0f;
		}
		depthbuffer[index].color = newColor;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize,
						cudaMat4 modelViewProj, glm::vec3 lightPos, glm::vec3 lightRGB,
						bool backFaceCulling, glm::vec3 vdir, bool antiAlias){

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
  frag.isLocked = false;
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

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, modelViewProj);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------

  //vdir = multiplyMV(inverseMV, glm::vec4(vdir,1));

  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives,
							backFaceCulling, vdir);

  cudaDeviceSynchronize();

  //stream compact
  if (backFaceCulling)
  {
	  //stream compact
	  thrust::device_ptr<triangle> beginItr(primitives);
	  thrust::device_ptr<triangle> endItr = beginItr + ibosize/3;
	  endItr = thrust::remove_if(beginItr, endItr, isInvisible());
	  int numPrimitives = (int)(endItr - beginItr);
	  primitiveBlocks = ceil(((float)numPrimitives)/((float)tileSize));
  }

  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lightPos, lightRGB );

  cudaDeviceSynchronize();

  if (antiAlias)
  {
	  antiAliasKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);

	  cudaDeviceSynchronize();
  }
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
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

