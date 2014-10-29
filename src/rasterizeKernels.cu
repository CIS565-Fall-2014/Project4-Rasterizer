// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "Camera.h"

//#define DEPTH_TEST
#define BARYCENTRIC_COLOR

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
float *device_nbo;
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
	  //f.position.z = 10000.0f;
	  f.color = glm::vec3(1, 1, 1);
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, float *nbo, int nbosize, Camera *camera, glm::mat4 modelTransformMatrix){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<vbosize/3){
		glm::vec4 vertex(vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2], 1.0f);
		glm::vec4 normal(nbo[index * 3], nbo[index * 3 + 1], nbo[index * 3 + 2], 0.0f);
		glm::mat4 MVP = camera->projectionMatrix * camera->viewMatrix * modelTransformMatrix;
		glm::mat4 MV = camera->viewMatrix * modelTransformMatrix;
		// to eye
		vertex = MVP * vertex;
		normal = modelTransformMatrix * normal;
		//to clip
		vertex /= vertex.w;
		//transform to view port
		vbo[3 * index] = camera->width * 0.5f * (vertex.x + 1.0f);
		vbo[3 * index + 1] = camera->height *(1- 0.5f * (vertex.y + 1.0f));
		vbo[3 * index + 2] = vertex.z;

		nbo[3 * index] = normal.x;
		nbo[3 * index + 1] = normal.y;
		nbo[3 * index + 2] = normal.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float *nbo, int nbosize, triangle* primitives, Camera *camera){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize / 3;
	if (index < primitivesCount){
		primitives[index].p0 = glm::vec3(vbo[3 * ibo[index * 3]], vbo[3 * ibo[index * 3] + 1], vbo[3 * ibo[index * 3] + 2]);
		primitives[index].p1 = glm::vec3(vbo[3 * ibo[index * 3 + 1]], vbo[3 * ibo[index * 3 + 1] + 1], vbo[3 * ibo[index * 3 + 1] + 2]);
		primitives[index].p2 = glm::vec3(vbo[3 * ibo[index * 3 + 2]], vbo[3 * ibo[index * 3 + 2] + 1], vbo[3 * ibo[index * 3 + 2] + 2]);
		primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
		primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
		primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);

		primitives[index].n0 = glm::vec3(nbo[3 * ibo[index * 3]], nbo[3 * ibo[index * 3] + 1], nbo[3 * ibo[index * 3] + 2]);
		primitives[index].n1 = glm::vec3(nbo[3 * ibo[index * 3 + 1]], nbo[3 * ibo[index * 3 + 1] + 1], nbo[3 * ibo[index * 3 + 1] + 2]);
		primitives[index].n2 = glm::vec3(nbo[3 * ibo[index * 3 + 2]], nbo[3 * ibo[index * 3 + 2] + 1], nbo[3 * ibo[index * 3 + 2] + 2]);;

		glm::vec3 p0 = glm::vec3(vbo[3 * ibo[index * 3]], vbo[3 * ibo[index * 3] + 1], vbo[3 * ibo[index * 3] + 2]);
		glm::vec3 p1 = glm::vec3(vbo[3 * ibo[index * 3 + 1]], vbo[3 * ibo[index * 3 + 1] + 1], vbo[3 * ibo[index * 3 + 1] + 2]);
		glm::vec3 p2 = glm::vec3(vbo[3 * ibo[index * 3 + 2]], vbo[3 * ibo[index * 3 + 2] + 1], vbo[3 * ibo[index * 3 + 2] + 2]);

		primitives[index].visible = (calculateSignedArea(primitives[index]) > 1e-6);
	}
}

__device__ bool inBoundary(float x, float y)
{
	return x >= -1.0f && x<1.0f && y >= -1.0f && y<1.0f;
}

__device__ int inBoundary(int x, int y, glm::vec2 resolution)
{
	return x >= 0 && y >= 0 && x<resolution.x && y<resolution.y;
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, Camera *camera){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index<primitivesCount){
		triangle tri = primitives[index];
		if (!tri.visible)	return;
		if (calculateSignedArea(tri) < 1e-6) return;
		glm::vec3 minPoint, maxPoint;
		getAABBForTriangle(tri, minPoint, maxPoint);
		
		if (!(inBoundary(minPoint.x, 0, resolution) || inBoundary(maxPoint.x, 0, resolution))) return;
		for (int j = (int)minPoint.y; j <= (int)maxPoint.y; j++)
		{
			for (int i = (int)minPoint.x; i <=(int) maxPoint.x + 0.001f; i++)
			{
				if (!inBoundary(i, j, resolution)) continue;
				int targetIdx = i + j*(int)resolution.x;
				float pointX = (float)i;
				float pointY = (float)j;
				glm::vec2 point(pointX, pointY);
				glm::vec3 baryCoords = calculateBarycentricCoordinate(tri, point);
				if (!isBarycentricCoordInBounds(baryCoords)) continue;
				float depth = getZAtCoordinate(baryCoords, tri);
				glm::vec3 color = baryCoords.x * tri.c0 + baryCoords.y * tri.c1 + baryCoords.z * tri.c2;
				glm::vec3 normal = baryCoords.x * tri.n0 + baryCoords.y * tri.n1 + baryCoords.z * tri.n2;

				if (depthbuffer[targetIdx].position.z < -1)
				{
					depthbuffer[targetIdx].position.z = depth;
					depthbuffer[targetIdx].color = color;
					depthbuffer[targetIdx].normal = normal;
				}
				else if (depth>depthbuffer[targetIdx].position.z)
				{
					depthbuffer[targetIdx].position.z = depth;
					depthbuffer[targetIdx].color = color;
					depthbuffer[targetIdx].normal = normal;
				}

			}
		}

	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, Camera *camera){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if (x <= resolution.x && y <= resolution.y){


		float specular = 30.0;
		float ka = 0.1;
		float kd = 0.7;
		float ks = 0.2;
		glm::vec4 lightPos4(camera->lightPos, 1.0f);
		glm::vec4 lightPosEye4 = camera->viewMatrix * lightPos4;
		glm::vec3 lightPosEye(lightPosEye4.x, lightPosEye4.y, lightPosEye4.z);
		glm::vec4 dbp4(depthbuffer[index].position, 1.0f);
		glm::vec4 dbpEye4 = glm::inverse(camera->projectionMatrix) * dbp4;
		glm::vec3 dbpEye = glm::vec3(dbpEye4.x, dbpEye4.y, dbpEye4.z);

		glm::vec3 lightVectorEye = glm::normalize(lightPosEye - dbpEye);  // watch out for division by zero
		glm::vec3 normal = depthbuffer[index].normal; // watch out for division by zero
		glm::vec4 normal4(normal, 1.0f);
		glm::vec4 normalEye4 = camera->viewMatrix * normal4;
		glm::vec3 normalEye = glm::normalize( glm::vec3(normalEye4.x, normalEye4.y, normalEye4.z));
		float diffuseTerm = glm::clamp(glm::dot(normal, normalEye), 0.0f, 1.0f);

		glm::vec3 R = glm::normalize(glm::reflect(-lightVectorEye, normal)); // watch out for division by zero
		glm::vec3 V = glm::normalize(-dbpEye); // watch out for division by zero
		float specularTerm = pow(fmaxf(glm::dot(R, V), 0.0f), specular);
#ifdef BARYCENTRIC_COLOR
		glm::vec3 materialColor = depthbuffer[index].color;
#else
		glm::vec3 materialColor = glm::vec3(0.5f, 0.4f, 0.7f);
#endif
		
		glm::vec3 color = ka*materialColor + glm::vec3(1.0f) * (kd*materialColor*diffuseTerm + ks*specularTerm);
		depthbuffer[index].color = color;
		if (depthbuffer[index].position.z < -1.0f  )
		{
			depthbuffer[index].color = glm::vec3(0.6f, 0.6f, 0.6f);
		}
#ifdef DEPTH_TEST
		depthbuffer[index].color = glm::vec3(0.5 * (depthbuffer[index].position.z + 1));
#endif
#ifdef NORMAL_TEST
		depthbuffer[index].color = glm::normalize(depthbuffer[index].normal.x * glm::vec3(1, 0, 0) + depthbuffer[index].normal.y * glm::vec3(0, 1, 0) + depthbuffer[index].normal.z * glm::vec3(0, 0, 1));
		if (depthbuffer[index].position.z < -1.0f)
			depthbuffer[index].color = glm::normalize(-(camera->viewDirection.x * glm::vec3(1, 0, 0) + camera->viewDirection.y * glm::vec3(0, 1, 0) + camera->viewDirection.z * glm::vec3(0, 0, 1)));
#endif
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float *nbo, int nbosize, Camera camera, glm::mat4 modelTransformMatrix){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up camera
  Camera *cudaCamera = NULL;
  cudaMalloc((void**)&cudaCamera, sizeof(Camera));
  cudaMemcpy(cudaCamera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);

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
  cudaMemcpy(device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, cudaCamera, modelTransformMatrix);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel << <primitiveBlocks, tileSize >> >(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives, cudaCamera);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cudaCamera);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
 // glm::vec3 lightPosition(0, 0, 0);
  fragmentShadeKernel << <fullBlocksPerGrid, threadsPerBlock >> >(depthbuffer, resolution, cudaCamera);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();
  cudaFree(cudaCamera);
  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree(device_nbo);
}

