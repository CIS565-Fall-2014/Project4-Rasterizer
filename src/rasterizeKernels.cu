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
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, cam theCam){
	

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  //Transform incoming vertex position from model to clip coordinates
	  glm::vec4 pModel(vbo[index*3], vbo[index*3 + 1],vbo[index*3 + 2], 1.0f);
	  glm::vec3 pClip = multiplyMV(theCam.M_mvp, pModel);
	  glm::vec4 nModel(nbo[index*3], nbo[index*3 + 1],nbo[index*3 + 2], 0.0f);
	  glm::vec3 nClip = glm::normalize( multiplyMV(theCam.M_mvp, nModel));
	  //glm::vec3 nClip = glm::normalize( multiplyMV(theCam.M_mvp_prime, nModel));

	  vbo[index*3] = pClip.x; 
	  vbo[index*3 + 1] = pClip.y; 
	  vbo[index*3 + 2] = pClip.z;

	  nbo[index*3] = nClip.x; 
	  nbo[index*3 + 1] = nClip.y; 
	  nbo[index*3 + 2] = nClip.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
	float* nbo, int nbosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  //get indice number
	  int i0 = ibo[index*3];
	  int i1 = ibo[index*3+1];
	  int i2 = ibo[index*3+2];

	  //assemble primitive points
	  primitives[index].p0 = glm::vec3(vbo[i0*3], vbo[i0*3+1], vbo[i0*3+2]);
	  primitives[index].p1 = glm::vec3(vbo[i1*3], vbo[i1*3+1], vbo[i1*3+2]);
	  primitives[index].p2 = glm::vec3(vbo[i2*3], vbo[i2*3+1], vbo[i2*3+2]);

	  //assemble primitive colors
	/*  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);*/
	  primitives[index].c0 = glm::vec3(1,1,1);
	  primitives[index].c1 = glm::vec3(1,1,1);
	  primitives[index].c2 = glm::vec3(1,1,1);

	  //assemble primitive normals;
	  glm::vec3 n0 = glm::vec3(nbo[i0*3], nbo[i0*3+1], nbo[i0*3+2]);
	  glm::vec3 n1 = glm::vec3(nbo[i1*3], nbo[i1*3+1], nbo[i1*3+2]);
	  glm::vec3 n2 = glm::vec3(nbo[i2*3], nbo[i2*3+1], nbo[i2*3+2]);
	  primitives[index].n = (n0 + n1 + n2)/3.0f;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index<primitivesCount){
	 
	  triangle tri = primitives[index];
	
	  if(tri.n.z <0){ // back facing triangles
		 // printf("back face triangle index: %d\n", index);
		  return;
	  }
	//printf("non back face triangle index: %d\n", index);
	  //get bounding box for this triangle
	  glm::vec3 minpoint, maxpoint;   // in the screen coordinates
	  getAABBForTriangle(tri, minpoint, maxpoint);
	  glm::vec2 minPoint, maxPoint;   //in image coodinates
	  minPoint = screenToImage( glm::vec2(minpoint.x, minpoint.y), resolution);   //viewport transform
	  maxPoint = screenToImage( glm::vec2(maxpoint.x, maxpoint.y), resolution);
	  int xMin = (int)floor(minPoint.x);
	  int yMin = (int)floor(maxPoint.y);
	  int xMax = (int)ceil(maxPoint.x);
	  int yMax = (int)ceil(minPoint.y);
	  //printf("min = %.2f, %.2f; max = %.2f, %.2f\n", minpoint.x, minpoint.y, maxpoint.x, maxpoint.y);
	 // printf("min = %.2f, %.2f; max = %.2f, %.2f\n", minPoint.x, minPoint.y, maxPoint.x, maxPoint.y);

	 // clipping
	  xMin = ( xMin > 0.0f )? xMin : 0.0f;
	  yMin = ( yMin > 0.0f) ? yMin : 0.0f;
	  xMax = ( xMax < resolution.x -1) ? xMax : resolution.x -1;
	  yMax = ( yMax < resolution.y -1) ? yMax : resolution.y -1;

	  if(xMin<0 || yMin<0 || xMin>=resolution.x || yMin>=resolution.y)
		  return;
	  if(xMax<0 || yMax<0 || xMax>=resolution.x || yMax>=resolution.y)
		  return;

	  for(int y = yMin; y < yMax; y++){  //top to down

		for(int x = xMin; x < xMax; x++){   //left to right
			 int pixelID = x + resolution.x*y;
			 glm::vec2 screenCoord = imageToScreen(glm::vec2(x,y),resolution);
			 //barycentric coordinate for (x,y) point
			 glm::vec3 b = calculateBarycentricCoordinate(tri, screenCoord);
			
			 //p is in the triangle bounds
			 if(isBarycentricCoordInBounds(b)){
				 float z = getZAtCoordinate(b, tri);  //depth

				 // printf("frag color: %f.2, %f.02, %f.02\n", frag.color.r, frag.color.g, frag.color.b);
				// printf("triangle index: %d, frag normal: %.2f, %.2f, %.2f\n", index,frag.normal.x, frag.normal.y, frag.normal.z);
				// printf("frag pos: %.2f, %.2f, %.2f\n", frag.position.x, frag.position.y, frag.position.z);
				 if(z > depthbuffer[pixelID].position.z){
					 fragment frag = depthbuffer[pixelID];
					 frag.color = interpolateColor(b,tri);
					 frag.position = interpolatePosition(b,tri);
				 
					 //frag.position.x = screenCoord.x;
					 //frag.position.y = screenCoord.y;
					// frag.position.z = z;
					 frag.normal = tri.n;
					 depthbuffer[pixelID] = frag;
				 }
				 
			 }
		  }

	  }
	 

	  primitives[index] = tri;  //update
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, cam theCam){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  if(SHADING_MODE == 0){   //shade by normal
		  depthbuffer[index].color.r = glm::clamp(depthbuffer[index].normal.x, 0.0f, 1.0f);
		  depthbuffer[index].color.g = glm::clamp(depthbuffer[index].normal.y, 0.0f, 1.0f);
		  depthbuffer[index].color.b = glm::clamp(depthbuffer[index].normal.z, 0.0f, 1.0f);
	  }else if(SHADING_MODE == 1){   //shade by depth
		  depthbuffer[index].color.r = glm::clamp(depthbuffer[index].position.z/10.0f, 0.0f, 1.0f);
		  depthbuffer[index].color.g = glm::clamp(depthbuffer[index].position.z/10.0f, 0.0f, 1.0f);
		  depthbuffer[index].color.b = glm::clamp(depthbuffer[index].position.z/10.0f, 0.0f, 1.0f);
	  }else if(SHADING_MODE == 2){    //diffuse shade
		  glm::vec3 viewDir = glm::normalize(theCam.eye - depthbuffer[index].position);
		  depthbuffer[index].color = depthbuffer[index].color * glm::dot(viewDir, depthbuffer[index].normal);
		  //depthbuffer[index].color.r = glm::clamp(depthbuffer[index].color.r, 0.0f, 1.0f);
		 // depthbuffer[index].color.g = glm::clamp(depthbuffer[index].color.g, 0.0f, 1.0f);
		 // depthbuffer[index].color.b = glm::clamp(depthbuffer[index].color.b, 0.0f, 1.0f);
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, 
	int* ibo, int ibosize, float * nbo, int nbosize){

  cam theCam(65.0f, (float)resolution.x/(float)resolution.y, glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, 1.0f, -1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
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

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);
  //printf("cbo size: %d \n", cbosize);
  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, theCam);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, 
	  device_nbo, nbosize, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, theCam);

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
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

