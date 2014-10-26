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
	  glm::vec3 pModel(vbo[index*3], vbo[index*3 + 1],vbo[index*3 + 2]);
	  glm::vec3 pClip = multiplyMV(theCam.M_mvp, glm::vec4(pModel, 1.0f));

	  //Transform normal into clip coordinates
	  glm::vec3 nModel(nbo[index*3], nbo[index*3 + 1],nbo[index*3 + 2]);
	 
	  /*glm::vec3 nTip_OS = pModel + nModel;
	  glm::vec3 nTip_WS = multiplyMV(theCam.M_mvp, glm::vec4(nTip_OS, 1.0f));
	  glm::vec3 nClip = glm::normalize(nTip_WS - pClip);*/
	 // glm::vec3 nClip = glm::normalize( multiplyMV(theCam.M_mvp, glm::vec4(nModel, 0.0f)));
	  glm::vec3 nClip = glm::normalize( multiplyMV(theCam.M_mv_prime, glm::vec4(nModel, 0.0f)));
	  
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
	  //if(index == 2000){
		//  printf("transformed normal: %.2f, %.2f, %.2f\n", primitives[index].n.x, primitives[index].n.y, primitives[index].n.z);
	 // }
  }
}


//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
	
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index<primitivesCount){
	 
	  triangle tri = primitives[index];
	  
	  if(tri.n.z <0){ // back facing triangles
		  return;
	  }

	  //get bounding box for this triangle
	  glm::vec3 minpoint, maxpoint;   // in the screen coordinates, floats
	  getAABBForTriangle(tri, minpoint, maxpoint);
	  glm::vec2 minPoint, maxPoint;   //in image coodinates, ints
	  minPoint = screenToImage( glm::vec2(minpoint.x, minpoint.y), resolution);   //viewport transform
	  maxPoint = screenToImage( glm::vec2(maxpoint.x, maxpoint.y), resolution);
	  int xMin = (int)floor(minPoint.x);
	  int xMax = (int)ceil(maxPoint.x);
	  int yMin = (int)floor(maxPoint.y);
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

	  //scanline approach
	  for(int y = yMin; y < yMax; y++){  //top to down
		for(int x = xMin; x < xMax; x++){   //left to right
			 int pixelID = x + resolution.x*y;
			 glm::vec2 screenCoord = imageToScreen(glm::vec2(x,y),resolution);  //perspective transformation
			 glm::vec3 b = calculateBarycentricCoordinate(tri, screenCoord);  //barycentric coordinate for (x,y) pixel
			 
			 if(isBarycentricCoordInBounds(b)){  //p is in the triangle bounds
				 float z = getZAtCoordinate(b, tri);  //depth

				 if(Z_TEST == 1){  //do the depth test with atomic function
					// while(atomicCAS(&depthbuffer[pixelID].tested, 1, 0) != 1);  //until current fragment is tested
				 }
				 if(z > depthbuffer[pixelID].position.z && z <= 1.0f){
					 fragment frag = depthbuffer[pixelID];
					 frag.color = interpolateColor(b,tri);
					 //frag.position = interpolatePosition(b,tri);
					 /*glm::vec3 tmp = interpolatePosition(b,tri);
					 frag.position.x = tmp.x;
					 frag.position.y = tmp.y;
					 frag.position.z = z;*/
					 frag.position.x = screenCoord.x;
					 frag.position.y = screenCoord.y;
					 frag.position.z = z;
					 frag.normal = tri.n;
					 
					 if(LINE_RASTER == 1){   //shade line color
						 glm::vec3 lineColor(0.0f,0.0f,1.0f);  //blue
						 glm::vec3 p = interpolatePosition(b,tri);
						 if(fabs(glm::dot(glm::normalize(tri.p0 - p), glm::normalize(tri.p0 - tri.p1))-1.0f)<0.0001f||
							 fabs(glm::dot(glm::normalize(tri.p1 - p), glm::normalize(tri.p1 - tri.p2))-1.0f)<0.0001f ||
							 fabs(glm::dot(glm::normalize(tri.p2 - p), glm::normalize(tri.p2 - tri.p0))-1.0f)<0.0001f ){
				
								 frag.color = lineColor;
								 frag.normal = glm::vec3(0.0f, 0.0f, 1.0f);
						 }
					 }

					 depthbuffer[pixelID] = frag;
				 }
				// atomicExch(&depthbuffer[pixelID].tested, 1);
				 
			 }
		  }

	  }
	 

	  primitives[index] = tri;  //update
  }
}

// display points
__global__ void rasterizationPointsKernel(float* vbo, int vbosize, float * nbo, int nbosize, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  //find the point
	  glm::vec3 point(vbo[index*3], vbo[index*3+1], vbo[index*3+2]);
	  glm::vec3 normal(nbo[index*3], nbo[index*3+1], nbo[index*3+2]);
	  if(normal.z < 0)
		  return;

	  //locate the pixel
	  glm::vec2 pixel = screenToImage( glm::vec2(point.x, point.y), resolution);   //viewport transform
	  if(pixel.x<0 || pixel.y<0 || pixel.x>=resolution.x || pixel.y>=resolution.y)
		  return;
	  int pixelID = pixel.x + pixel.y * resolution.x;

	  //shade the point representation
	  if(point.z > depthbuffer[pixelID].position.z ){
		  glm::vec3 pointColor(1.0f, 1.0f, 0.0f);   //yellow
		 /* depthbuffer[pixelID].position = point;
		  depthbuffer[pixelID].color = pointColor;   
		  depthbuffer[pixelID].normal = glm::vec3(0.0f, 0.0f, 1.0f);*/
		  for(int i=pixel.x-1; i<=pixel.x+1; i++){
			  for(int j=pixel.y-1; j<=pixel.y+1; j++){
				  if(i<0 || j<0 || i>=resolution.x || j>=resolution.y)
						return;
				  int newpixelID = i + j * resolution.x;
				  depthbuffer[newpixelID].position = point;
				  depthbuffer[newpixelID].color = pointColor;   
				  depthbuffer[newpixelID].normal = glm::vec3(0.0f, 0.0f, 1.0f);
				 // atomicExch(&depthbuffer[pixelID].tested, 1);
			  }
		  }
	  }

  }
}



//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, cam theCam, glm::vec3 light){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  if(SHADING_MODE == 0){   //shade by normal
		  depthbuffer[index].color.r = glm::clamp(depthbuffer[index].normal.x, 0.0f, 1.0f);
		  depthbuffer[index].color.g = glm::clamp(depthbuffer[index].normal.y, 0.0f, 1.0f);
		  depthbuffer[index].color.b = glm::clamp(depthbuffer[index].normal.z, 0.0f, 1.0f);
	  }
	  else if(SHADING_MODE == 1){   //shade by depth
		  depthbuffer[index].color.r = glm::clamp(depthbuffer[index].position.z/10.0f, 0.0f, 1.0f);
		  depthbuffer[index].color.g = glm::clamp(depthbuffer[index].position.z/10.0f, 0.0f, 1.0f);
		  depthbuffer[index].color.b = glm::clamp(depthbuffer[index].position.z/10.0f, 0.0f, 1.0f);
	  }
	  else if(SHADING_MODE == 2){    //diffuse shade
		  glm::vec3 lightDir = glm::normalize(light - depthbuffer[index].position);
		  float cosTerm = glm::clamp(glm::dot(lightDir, depthbuffer[index].normal), 0.0f, 1.0f);
		  depthbuffer[index].color = glm::clamp(cosTerm * depthbuffer[index].color, 0.0f, 1.0f);
	  }
	  else if (SHADING_MODE == 3){  //blinn-phong shade
		  float coeff = 5.0f;
		  glm::vec3 lightDir = glm::normalize(light - depthbuffer[index].position);
		  float cosTerm = glm::clamp(glm::dot(lightDir, depthbuffer[index].normal), 0.0f, 1.0f);
		  depthbuffer[index].color = glm::clamp( std::pow(cosTerm,coeff) * depthbuffer[index].color, 0.0f, 1.0f);
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
  
	//set up camera,
 // cam theCam = mouseCam;

		//set up light
  glm::vec3 lightPos(500.0f, 500.0f, 1000.0f);    //add a light in the scene for shading

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

  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, mouseCam);

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
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, mouseCam, lightPos);

  cudaDeviceSynchronize();
  //------------------------------
  //point raster shader
  //------------------------------
  if(POINT_RASTER ==1){
	primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
	rasterizationPointsKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, depthbuffer, resolution);   //render point out
  }
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

