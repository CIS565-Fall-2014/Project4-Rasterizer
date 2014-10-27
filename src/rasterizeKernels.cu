// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "glm/gtc/matrix_transform.hpp"
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "utilities.h"

#define DEBUG 0
#define BLEND 1
#define DISPLACEMENT 1

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
float* device_nbo;
float* device_minZ;
float* device_texture;
int* device_ibo;
triangle* primitives;
light* device_lights;
// camera settings

float fovy = 60;

//light settings
light lights[1];
int nLight = 1;



void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	   int i;
	 std::cin >> i;
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
__global__ void vertexShadeKernel(float* vbo, float * nbo, int vbosize, glm::mat4 mvpMat,  glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  float x = vbo[3 * index];
	  float y = vbo[3 * index+1];
	  float z = vbo[3 * index+2];
	  glm::vec4 vertex(x,y,z,1.0f);

	  x = nbo[3 * index];
	  y = nbo[3 * index+1];
	  z = nbo[3 * index+2];
	  glm::vec4 normal(x,y,z,0.0f);

	  glm::vec4 v = mvpMat * vertex;
	  glm::vec4 n = mvpMat * normal;
	  // transform into clip space with normalized coordinates
	  v.x /= v.w;
	  v.y /= v.w;
	  v.z /= v.w;

	  // fit coordinates into the view port
	  v.x = 0.5f * resolution.x * (v.x + 1);
	  v.y = 0.5f * resolution.y * (v.y + 1);
	  v.z = 0.5f - 0.5f * v.z;

	  vbo[3 * index] = v.x;
	  vbo[3 * index + 1] = v.y;
	  vbo[3 * index + 2] = v.z;


	  nbo[3 * index] = n.x;
	  nbo[3 * index + 1] = n.y;
	  nbo[3 * index + 2] = n.z;
	  /*if(ret.x < resolution.x && ret.x >= 0 && ret.y < resolution.y && ret.y >= 0 && ret.z >=0 &&ret.z < 1 )
	  {
		  int pixelID = ceil(ret.x) + ceil(resolution.y-ret.y) * (resolution.x);
		  depthbuffer[pixelID].color = glm::vec3(1,1,1);
	  }*/
  }
}


__global__ void displancementKernel(float* vbo, float * nbo, int vbosize,  float t, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  float scale = 0.5f; 
	  float displacement = 0.1f * scale * (sin(t) + 1.0f);
	  vbo[3 * index] += displacement * nbo[3*index];
	  vbo[3 * index + 1] += displacement * nbo[3*index+1];
	  vbo[3 * index + 2] += displacement * nbo[3*index+2];
  }
}
//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float * nbo, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int vid1 = ibo[3*index];
	  int vid2 = ibo[3*index+1];
	  int vid3 = ibo[3*index+2];

	  glm::vec3 v1 = glm::vec3(vbo[3*vid1], vbo[3*vid1+1], vbo[3*vid1+2]);
	  glm::vec3 v2 = glm::vec3(vbo[3*vid2], vbo[3*vid2+1], vbo[3*vid2+2]);
	  glm::vec3 v3 = glm::vec3(vbo[3*vid3], vbo[3*vid3+1], vbo[3*vid3+2]);

	  /*glm::vec3 c1 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  glm::vec3 c2 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  glm::vec3 c3 = glm::vec3(cbo[6], cbo[7], cbo[8]);*/

	  glm::vec3 c1 = glm::vec3(cbo[3*vid1], cbo[3*vid1+1], cbo[3*vid1+2]);
	  glm::vec3 c2 = glm::vec3(cbo[3*vid2], cbo[3*vid2+1], cbo[3*vid2+2]);
	  glm::vec3 c3 = glm::vec3(cbo[3*vid3], cbo[3*vid3+1], cbo[3*vid3+2]);

	  glm::vec3 n1 = glm::vec3(nbo[3*vid1], nbo[3*vid1+1], nbo[3*vid1+2]);
	  glm::vec3 n2 = glm::vec3(nbo[3*vid2], nbo[3*vid2+1], nbo[3*vid2+2]);
	  glm::vec3 n3 = glm::vec3(nbo[3*vid3], nbo[3*vid3+1], nbo[3*vid3+2]);

	  primitives[index].c0 = c1;
	  primitives[index].c1 = c2;
	  primitives[index].c2 = c3;

	  primitives[index].p0 = v1;
	  primitives[index].p1 = v2;
	  primitives[index].p2 = v3;

	  primitives[index].n0 = n1;
	  primitives[index].n1 = n2;
	  primitives[index].n2 = n3;
  }
}

__device__ static float fatomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// an adapted atomicMin for float from internet
__device__ static float fatomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  glm::vec3 minPoint(0);
	  glm::vec3 maxPoint(0);
	  getAABBForTriangle(primitives[index], minPoint, maxPoint);
	  if(maxPoint.y <= resolution.x && maxPoint.x <= resolution.x && minPoint.y >= 0 && minPoint.x >= 0)
	  {
		  for(int j = (int)minPoint.y; j <= (int)maxPoint.y; ++j)
		  {
				for(int i = (int)minPoint.x; i <= (int)maxPoint.x; ++i)
				{
					glm::vec3 barycentricCoordinate = calculateBarycentricCoordinate(primitives[index], glm::vec2(i,j));
					if(isBarycentricCoordInBounds(barycentricCoordinate))
					{
						glm::vec3 intersection = barycentricCoordinate.x * primitives[index].p0 + barycentricCoordinate.y * primitives[index].p1+ barycentricCoordinate.z * primitives[index].p2;
						int pixelId = (int)i + (int)j * resolution.x;
						fatomicMin(&depthbuffer[pixelId].z, intersection.z);
						//__syncthreads();
						if(intersection.z == depthbuffer[pixelId].z)
						{
							depthbuffer[pixelId].normal = glm::normalize(glm::vec3(barycentricCoordinate.x * primitives[index].n0 + barycentricCoordinate.y * primitives[index].n1 + barycentricCoordinate.z * primitives[index].n2));
							depthbuffer[pixelId].color = barycentricCoordinate.x * primitives[index].c0 + barycentricCoordinate.y * primitives[index].c1 + barycentricCoordinate.z * primitives[index].c2;
							depthbuffer[pixelId].position = barycentricCoordinate.x * primitives[index].p0 + barycentricCoordinate.y * primitives[index].p1 + barycentricCoordinate.z * primitives[index].p2;
						}
					}
				}
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, light* lights, glm::vec3 eyePos, float * texture){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  float ks = 1.0;
	  glm::vec3 L = glm::normalize(glm::vec3(lights[0].position - depthbuffer[index].position));

	  float diffuse =  glm::clamp(glm::dot(depthbuffer[index].normal, L), 0.0f, 1.0f);
	  depthbuffer[index].color = diffuse * depthbuffer[index].color * lights[0].color;

	  if(ks > 0)
	  {
		  glm::vec3 reflectedDir = glm::normalize(glm::reflect(L,depthbuffer[index].normal));
		  float specularIntensity = pow(glm::clamp(glm::dot(reflectedDir, glm::normalize( - depthbuffer[index].position) ),0.0f,1.0f),30);
		  float specular = ks * glm::clamp(specularIntensity, 0.0f, 1.0f);

		  depthbuffer[index].color += specular * glm::vec3(1.0f);
	  }

	  if(depthbuffer[index].position.z == -10000)
	  {
		glm::vec3 background =glm::vec3( texture[3*index], texture[3*index+1], texture[3*index+2]);
		  depthbuffer[index].color = background;
	  }


	  if(DEBUG)
		  depthbuffer[index].color = depthbuffer[index].normal;
  }
}

__global__ void blendKernel(fragment* depthbuffer, glm::vec2 resolution, light* lights, glm::vec3 eyePos, float * texture, float t){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  float alpha = 0.5f * (sinf(-t) + 1.0);
	  if(depthbuffer[index].position.z != -10000)
	  {
		  glm::vec3 background =glm::vec3( texture[3*index], texture[3*index+1], texture[3*index+2]);
		   depthbuffer[index].color = depthbuffer[index].color * alpha + background*(1-alpha);
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, 
	                  float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, 
					  glm::vec3 eyePos, glm::vec3 center, glm::mat4 modelMat, float * texture, float t){

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
  frag.z = 10000;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  lights[0].position = glm::vec3(500.0f,3.0f,0);
  lights[0].color = glm::vec3(1.0f,1.0f,1.0f);

  	cudaEvent_t start, stop;
	float time;

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

  device_texture = NULL;
  cudaMalloc((void**)&device_texture, 3*800*800*sizeof(float));
  cudaMemcpy( device_texture, texture, 3*800*800*sizeof(float), cudaMemcpyHostToDevice);

  device_lights = NULL;
  cudaMalloc((void**)&device_lights, nLight*sizeof(light));
  cudaMemcpy( device_lights, lights, nLight*sizeof(light), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //Set up matrices
  //------------------------------


  glm::mat4 viewMat = glm::lookAt(eyePos, center, glm::vec3(0,-1,0));
  glm::mat4 projMat = glm::perspective((float)fovy, (float)resolution.x/(float)resolution.y, 0.01f, 100.0f);

  glm::mat4 mvpMat =  projMat * viewMat * modelMat;



  //------------------------------
  //displacement
  //------------------------------

   cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );

  if(DISPLACEMENT)
  {
   displancementKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_nbo, vbosize,  t, resolution);

   cudaDeviceSynchronize();
  }

  	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	//std::cout << "#Displacement:  "<< time << " ms" << std::endl;
  //------------------------------
  //vertex shader
  //------------------------------
	   cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );

  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_nbo, vbosize, mvpMat, resolution);

  cudaDeviceSynchronize();

    	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	//std::cout << "#vertex shader:  "<< time << " ms" << std::endl;
  //------------------------------
  //primitive assembly
  //------------------------------
		   cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );

  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo,ibosize, device_nbo,primitives);

  cudaDeviceSynchronize();
      	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	//std::cout << "#Primitive assembly:  "<< time << " ms" << std::endl;
  //------------------------------
  //rasterization
  //------------------------------

			   cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );

  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();

        	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	//std::cout << "#Rasterization:  "<< time << " ms" << std::endl;
  //------------------------------
  //fragment shader
  //------------------------------

				   cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );

  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, device_lights,eyePos, device_texture);

  cudaDeviceSynchronize();
          	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	//std::cout << "#Rasterization:  "<< time << " ms" << std::endl;

  //------------------------------
  //blending
  //------------------------------
					   cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );
  if(BLEND)
  {
	  blendKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, device_lights, eyePos, device_texture, t);

	  cudaDeviceSynchronize();
  }

  cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	//std::cout << "#Blending:  "<< time << " ms" << std::endl;
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
						   cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

    cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	//std::cout << "#Rendering:  "<< time << " ms" << std::endl;

	//int i =0;
	//std::cin >> i;

  kernelCleanup();

  checkCUDAError("Kernel failed!");

}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( device_nbo );
  cudaFree( device_texture );
  cudaFree( device_lights );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

