// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
using namespace std;
glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_old_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives;
cudaMat4* MVP;
cudaMat4* device_MVP;
cudaMat4* NMV;
cudaMat4* device_NMV;

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
__global__ void vertexShadeKernel(float* vbo, int vbosize,float* nbo,cudaMat4* M,cudaMat4* NMV,glm::vec2 resolution,float z_near,float z_far){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec4 op=glm::vec4(vbo[3*index],vbo[3*index+1],vbo[3*index+2],1.0f);
	  //glm::vec4 on=glm::vec4(nbo[3*index],nbo[3*index+1],nbo[3*index+2],0.0f);
	  glm::vec4 p= multiplyMV(*M,op);
	  //glm::vec4 n=glm::normalize(multiplyMV(*NMV,on));
	  //glm::vec4 n=glm::normalize(multiplyMV(*M,on));
	  vbo[3*index]=p.x/p.w;
	  vbo[3*index+1]=p.y/p.w;
	  vbo[3*index+2]=p.z/p.w;
	  //viewport transform
	  vbo[3*index]=(vbo[3*index]+1.0f)/2.0f*resolution.x;
	  vbo[3*index+1]=(vbo[3*index+1]+1.0f)/2.0f*resolution.y;
	  vbo[3*index+2]=(z_near+(vbo[3*index+2]+1.0f)/2.0f*(z_far-z_near));
	  /*nbo[3*index]=n.x;
	  nbo[3*index+1]=n.y;
	  nbo[3*index+2]=n.z;*/
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo,triangle* primitives,float* old_vbo){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int a=3*ibo[3*index];
	  int b=3*ibo[3*index+1];
	  int c=3*ibo[3*index+2];
	  //position
	  primitives[index].p0=glm::vec3(vbo[a],vbo[a+1],vbo[a+2]);
	  primitives[index].p1=glm::vec3(vbo[b],vbo[b+1],vbo[b+2]);
	  primitives[index].p2=glm::vec3(vbo[c],vbo[c+1],vbo[c+2]);
	  primitives[index].old_p0=glm::vec3(old_vbo[a],old_vbo[a+1],old_vbo[a+2]);
	  primitives[index].old_p1=glm::vec3(old_vbo[b],old_vbo[b+1],old_vbo[b+2]);
	  primitives[index].old_p2=glm::vec3(old_vbo[c],old_vbo[c+1],old_vbo[c+2]);
	  //color
	  //primitives[index].c0=glm::vec3(cbo[a],cbo[a+1],cbo[a+2]);
	  //primitives[index].c1=glm::vec3(cbo[b],cbo[b+1],cbo[b+2]);
	  //primitives[index].c2=glm::vec3(cbo[c],cbo[c+1],cbo[c+2]);
	  glm::vec3 color=glm::vec3(0.5,0.5,0.5);
	  primitives[index].c0=color;
	  primitives[index].c1=color;
	  primitives[index].c2=color;
	  //normal
	  primitives[index].n0=glm::vec3(nbo[a],nbo[a+1],nbo[a+2]);
	  primitives[index].n1=glm::vec3(nbo[b],nbo[b+1],nbo[b+2]);
	  primitives[index].n2=glm::vec3(nbo[c],nbo[c+1],nbo[c+2]);

  }
}

//TODO: Implement a rasterization method, such as scanline.:http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution,float z_near,float z_far){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  triangle current_tri=primitives[index];
	  glm::vec3 minpoint,maxpoint;
	  getAABBForTriangle(current_tri,minpoint,maxpoint);
	  fragment current_frag;
	  for(int i=minpoint.x;i<maxpoint.x;i++)
	  {
		  for(int j=minpoint.y;j<maxpoint.y;j++)
		  {
			  if(i>=0 &&i<resolution.x&&j>=0&&j<resolution.y)
			  {
				  glm::vec3 B_p=calculateBarycentricCoordinate(current_tri,glm::vec2(i,j));
				  if(isBarycentricCoordInBounds(B_p))
				  {
					  current_frag.position=B_p.x*current_tri.p0+B_p.y*current_tri.p1+B_p.z*current_tri.p2;
					  current_frag.originial_position=B_p.x*current_tri.old_p0+B_p.y*current_tri.old_p1+B_p.z*current_tri.old_p2;
					  current_frag.normal=B_p.x*current_tri.n0+B_p.y*current_tri.n1+B_p.z*current_tri.n2;
					  //current_frag.color=glm::vec3(1.0,1.0,1.0);
					  current_frag.color=glm::vec3(B_p.x,B_p.y,B_p.z);
					  //current_frag.color=glm::clamp((current_tri.n0+current_tri.n1+current_tri.n2)/3.0f,0.0f,1.0f);
					  //current_frag.color=(current_tri.c0+current_tri.c1+current_tri.c2)/3.0f;
					  //current_frag.color=B_p.x*current_tri.c0+B_p.y*current_tri.c1+B_p.z*current_tri.c2;
					  if(depthbuffer[int((resolution.y-j)*resolution.x+resolution.x-i)].position.z>current_frag.position.z &&current_frag.position.z>z_near &&current_frag.position.z<z_far)
					  {
						depthbuffer[int((resolution.y-j)*resolution.x+resolution.x-i)]=current_frag;
					  }
				  }
			  }
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution,glm::vec3 light,glm::vec3 eye,glm::mat4 model,glm::mat4 view,glm::mat4 projection){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  glm::vec3 p=depthbuffer[index].originial_position;
	  glm::vec3 n=depthbuffer[index].normal;
	  glm::vec3 L=glm::normalize(light-p);
	  glm::vec3 V=glm::normalize(eye-p);
	  glm::vec3 H=glm::normalize(L+V);
	  //light color
	  glm::vec3 diffuse_color=glm::vec3(0.8,0.2,0.2);
	  glm::vec3 specular_color=glm::vec3(1.0,1.0,1.0);
	  //Blinn-Phong lighting
	  float diffuse=glm::clamp(glm::dot(L,n),0.0f,1.0f);
	  float specular=glm::clamp(glm::pow(glm::dot(H,n),100.0f),0.0f,1.0f);
	  //depthbuffer[index].color=glm::clamp(diffuse*diffuse_color+specular*specular_color,0.0f,1.0f);
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,float* nbo,int nbosize,glm::mat4 model,glm::mat4 view,glm::mat4 projection,glm::mat4 M,glm::mat4 n_MV,float z_near,float z_far,glm::vec3 eye){

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
  frag.originial_position=glm::vec3(0,0,0);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  MVP = new cudaMat4;
  *MVP = utilityCore::glmMat4ToCudaMat4(M);
  device_MVP=NULL;
  cudaMalloc((void**)&device_MVP,sizeof(cudaMat4));
  cudaMemcpy(device_MVP,MVP,sizeof(cudaMat4),cudaMemcpyHostToDevice);
  NMV = new cudaMat4;
  *NMV = utilityCore::glmMat4ToCudaMat4(n_MV);
  device_NMV=NULL;
  cudaMalloc((void**)&device_NMV,sizeof(cudaMat4));
  cudaMemcpy(device_NMV,NMV,sizeof(cudaMat4),cudaMemcpyHostToDevice);


  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_old_vbo = NULL;
  cudaMalloc((void**)&device_old_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_old_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

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
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize,device_nbo,device_MVP,device_NMV,resolution,z_near,z_far);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo,primitives,device_old_vbo);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution,z_near,z_far);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  glm::vec3 light=glm::vec3(0.0f,2.0f,1.0f);
 /* glm::vec4 light_p=multiplyMV(*MVP,glm::vec4(light,1.0f));
  light.x=light_p.x/light_p.w;
  light.y=light_p.y/light_p.w;
  light.z=light_p.z/light_p.w;
  light.x=(light.x+1.0f)/2.0f*resolution.x;
  light.y=(light.y+1.0f)/2.0f*resolution.y;
  light.z=z_near+(light.z+1.0f)/2.0f*(z_far-z_near);
  cout << light.x<<"  "<<light.y<<"  "<<light.z<< endl;*/

  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,light,eye,model,view,projection);

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
  cudaFree(device_old_vbo);
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree(device_nbo);
  cudaFree(device_MVP);
  cudaFree(device_NMV);
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

