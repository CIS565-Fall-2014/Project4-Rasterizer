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
triangle* primitives;

cudaMat4 * projectionTransform;
cudaMat4 * MVtransform;
cudaMat4 * MVPtransform;


#define BLOCK_SIZE 16

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
   // exit(EXIT_FAILURE); 
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

//vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, cudaMat4 * MVP){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){

	  glm::vec4 oldP = glm::vec4(vbo[3*index],vbo[3*index + 1],vbo[3*index +2], 1.0f);
	  glm::vec4 newP = multiplyMV4(*MVP,oldP); 
	  vbo[3*index] = newP.x/newP.w;
	  vbo[3*index+1] = newP.y/newP.w;
	  vbo[3*index+2] = newP.z/newP.w;

  }
}

//Primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){

	  int i0,i1,i2;

	  i0 = 3 * ibo[3*index];
	  i1 = 3 * ibo[3*index + 1];
	  i2 = 3 * ibo[3*index + 2];

	  primitives[index].p0 = glm::vec3(vbo[i0],vbo[i0+1],vbo[i0+2]);
	  primitives[index].p1 = glm::vec3(vbo[i1],vbo[i1+1],vbo[i1+2]);
	  primitives[index].p2 = glm::vec3(vbo[i2],vbo[i2+1],vbo[i2+2]);

	  primitives[index].c0 = glm::vec3(cbo[i0],cbo[i0+1],cbo[i0+2]);
	  primitives[index].c1 = glm::vec3(cbo[i1],cbo[i1+1],cbo[i1+2]);
	  primitives[index].c2 = glm::vec3(cbo[i2],cbo[i2+1],cbo[i2+2]);
	 
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		
		int totalPixel = resolution.x * resolution.y;
		float halfResoX =  0.5f * (float) resolution.x;
		float halfResoY =  0.5f * (float) resolution.y;
		glm::vec3 normal = glm::normalize(glm::cross(primitives[index].p1 - primitives[index].p0, primitives[index].p2 - primitives[index].p1));

		if(normal.z >= 0.0f) return;

		glm::vec3 Min,Max;
		getAABBForTriangle(primitives[index],Min,Max);
		float pixelWidth = 1.0f/(float) resolution.x;
		float pixelHeight = 1.0f/(float) resolution.y;

		//loop thru all pixels in the bounding box
		for(int i = 0;i < (Max.x - Min.x)/pixelWidth + 1; i++)
		{
			for(int j = 0;j <(Max.y - Min.y)/pixelHeight + 1; j++)
			{
				glm::vec2 pixelPos = glm::vec2(Min.x + (float)i * pixelWidth, Min.y + (float)j * pixelHeight);
				glm::vec3 pixelBaryPos = calculateBarycentricCoordinate(primitives[index], pixelPos);
				
				//not in triangle
				if(!isBarycentricCoordInBounds(pixelBaryPos))
				{
					continue;
				}

				else
				{
					int x,y, pixelIndex;
					x = pixelPos.x * halfResoX + halfResoX;
					y = pixelPos.y  * halfResoY+ halfResoY;
					pixelIndex = x + y * resolution.x;

					float pixelDepth = getZAtCoordinate(pixelBaryPos, primitives[index]);

					if(pixelIndex < totalPixel && pixelIndex >= 0)// && pixelDepth < depthbuffer[pixelIndex].position.z)
					{
						depthbuffer[pixelIndex].position = glm::vec3(pixelPos.x,pixelPos.y,pixelDepth);
						depthbuffer[pixelIndex].normal = normal;
						//depthbuffer[pixelIndex].color = pixelBaryPos.x * primitives[index].c0 + pixelBaryPos.y * primitives[index].c1 + pixelBaryPos.z * primitives[index].c2;
						depthbuffer[pixelIndex].color = glm::vec3(0.0f,1.0f,0.0f);
					}

				}

			}
		}
		

		//Only display vertices
		/*
		  int x0,y0,x1,y1,x2,y2, P0,P1,P2;
		  x0 = primitives[index].p0.x * 0.5f * resolution.x + 0.5f * resolution.x;
		  x1 = primitives[index].p1.x * 0.5f * resolution.x + 0.5f * resolution.x;
		  x2 = primitives[index].p2.x * 0.5f * resolution.x + 0.5f * resolution.x;

		  y0 = primitives[index].p0.y * 0.5f * resolution.y + 0.5f * resolution.y;
		  y1 = primitives[index].p1.y * 0.5f * resolution.y + 0.5f * resolution.y;
		  y2 = primitives[index].p2.y * 0.5f * resolution.y + 0.5f * resolution.y;

		  //reversed y 
		  P0 = x0 + (y0) * resolution.x;
		  P1 = x1 + (y1) * resolution.x;
		  P2 = x2 + (y2) * resolution.x;

		  int totalPixel = resolution.x * resolution.y;
		  
		  if(P0 < totalPixel && P0 >=0) depthbuffer[P0].color = primitives[index].c0;
		  if(P1 < totalPixel && P1 >=0)depthbuffer[P1].color = primitives[index].c1;
		  if(P2 < totalPixel && P2 >=0)depthbuffer[P2].color = primitives[index].c2;
		*/
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, light Light, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	//  glm::vec3 L = glm::normalize(Light.position - depthbuffer[index].position);

	//  float coe = glm::dot(L,depthbuffer[index].normal);
	//  coe = (coe<0.0f) ? 0.0f:coe;
	 // coe = (coe>1.0f) ? 1.0f:coe;

	 // coe = abs(depthbuffer[index].position.z/2.0f);
	//  depthbuffer[index].color *= coe * Light.color;
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

// Wrapper for the __global__ call that sets up the kernel calls and memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, glm::mat4 glmProjectionTransform, glm::mat4 glmMVtransform,light Light){
	
	projectionTransform = new cudaMat4;
	MVtransform = new cudaMat4;
	MVPtransform = new cudaMat4;

	cudaMat4 * dev_projectionTransform;
	cudaMat4 * dev_MVtransform;
	cudaMat4 * dev_MVPtransform;
  
	*projectionTransform = utilityCore::glmMat4ToCudaMat4(glmProjectionTransform);
	*MVtransform = utilityCore::glmMat4ToCudaMat4(glmMVtransform);
	*MVPtransform =utilityCore::glmMat4ToCudaMat4(glmProjectionTransform * glmMVtransform);

	cudaMalloc((void**) & dev_projectionTransform, sizeof(cudaMat4));
	cudaMalloc((void**) & dev_MVtransform, sizeof(cudaMat4));
	cudaMalloc((void**) & dev_MVPtransform, sizeof(cudaMat4));

	cudaMemcpy(dev_projectionTransform,projectionTransform,sizeof(cudaMat4),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MVtransform,MVtransform,sizeof(cudaMat4),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MVPtransform,MVPtransform,sizeof(cudaMat4),cudaMemcpyHostToDevice);

	light transformedLight = Light;
	transformedLight.position = multiplyMV(*MVPtransform,glm::vec4(Light.position,1.0f));

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

	tileSize = 32;
	int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

	//------------------------------
	//vertex shader
	//------------------------------
	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize,dev_MVPtransform);

	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

	cudaDeviceSynchronize();
	//------------------------------
	//rasterization
	//------------------------------
	rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

	cudaDeviceSynchronize();
	//------------------------------
	//fragment shader
	//------------------------------
	fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, transformedLight, resolution);

	cudaDeviceSynchronize();
	//------------------------------
	//write fragments to framebuffer
	//------------------------------
	render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

	cudaDeviceSynchronize();

	kernelCleanup();

	cudaFree(dev_projectionTransform);
	cudaFree(dev_MVtransform);
	cudaFree(dev_MVPtransform);

	delete projectionTransform,MVtransform,MVPtransform;


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

