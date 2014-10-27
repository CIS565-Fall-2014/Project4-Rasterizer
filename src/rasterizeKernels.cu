// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include <vector>

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
float* device_nbo;
int* device_ibo;
triangle* primitives;
float* device_tbo;

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
	  f.type = 0;
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
__global__ void vertexShadeKernel(float* vbo,float* nbo,float* tbo, int vbosize,glm::mat4 *mvp){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec4 pos = glm::vec4(vbo[3*index],vbo[3*index+1],vbo[3*index+2],1.0f);
	  tbo[2*index] = vbo[3*index]*1.5 - floor(vbo[3*index]*1.5);
	  tbo[2*index+1] = vbo[3*index+1]*1.5 - floor(vbo[3*index+1]*1.5);
	  pos = *mvp * pos;
	  vbo[3*index] = pos.x/pos.w;
	  vbo[3*index+1] = -pos.y/pos.w;
	  vbo[3*index+2] = pos.z/pos.w;

	  glm::vec4 normal = glm::vec4(nbo[3*index],nbo[3*index+1],nbo[3*index+2],0.0f);
	  normal = *mvp * normal;
	  nbo[3*index] = normal.x;
	  nbo[3*index+1] = -normal.y;
	  nbo[3*index+2] = normal.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float *nbo, int nbosize, float *tbo, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int p0_index = ibo[3 * index] * 3;
	  int p1_index = ibo[3 * index + 1] * 3;
	  int p2_index = ibo[3 * index + 2] * 3;

	  primitives[index].p0 =  glm::vec3(vbo[p0_index],vbo[p0_index+1],vbo[p0_index+2]);
	  primitives[index].p1 =  glm::vec3(vbo[p1_index],vbo[p1_index+1],vbo[p1_index+2]);
	  primitives[index].p2 =  glm::vec3(vbo[p2_index],vbo[p2_index+1],vbo[p2_index+2]);

	  primitives[index].n0 =  glm::vec3(nbo[p0_index],nbo[p0_index+1],nbo[p0_index+2]);
	  primitives[index].n1 =  glm::vec3(nbo[p1_index],nbo[p1_index+1],nbo[p1_index+2]);
	  primitives[index].n2 =  glm::vec3(nbo[p2_index],nbo[p2_index+1],nbo[p2_index+2]);

	  primitives[index].c0 = glm::vec3(1,1,1);
	  primitives[index].c1 = glm::vec3(1,1,1);
	  primitives[index].c2 = glm::vec3(1,1,1);

	  p0_index =p0_index/3*2;
	  p1_index =p1_index/3*2;
	  p2_index =p2_index/3*2;
	  primitives[index].t0 = glm::vec2(tbo[p0_index],tbo[p0_index+1]);
	  primitives[index].t1 = glm::vec2(tbo[p1_index],tbo[p1_index+1]);
	  primitives[index].t2 = glm::vec2(tbo[p2_index],tbo[p2_index+1]);
  }
}

//Draw vertices
__global__ void rasterizationKernelVertices(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  //back face culling
	  glm::vec3 v1 = primitives[index].p1 - primitives[index].p0;
	  glm::vec3 v2 = primitives[index].p2 - primitives[index].p1;
	  glm::vec3 tmp = glm::cross(v1,v2);
	  if(tmp.z>0)
		  return;

	  int x = (primitives[index].p0.x/2+0.5f)*(int)resolution.x;
	  int y = (primitives[index].p0.y/2+0.5f)*(int)resolution.y;
	  int buffer_index = y*(int)resolution.y+x;
	  if(depthbuffer[buffer_index].position.z  < getZAtCoordinate(glm::vec3(1,0,0),primitives[index])+0.0001f){
		  depthbuffer[buffer_index].color = glm::vec3(1.0f,0,0);
		  depthbuffer[buffer_index].type = 2;

		  x = (primitives[index].p0.x/2+0.5f)*(int)resolution.x+1;
		  y = (primitives[index].p0.y/2+0.5f)*(int)resolution.y;
		  depthbuffer[y*(int)resolution.y+x].color = glm::vec3(1.0f,0,0);
		  depthbuffer[y*(int)resolution.y+x].type = 2;
		  x = (primitives[index].p0.x/2+0.5f)*(int)resolution.x;
		  y = (primitives[index].p0.y/2+0.5f)*(int)resolution.y+1;
		  depthbuffer[y*(int)resolution.y+x].color = glm::vec3(1.0f,0,0);
		  depthbuffer[y*(int)resolution.y+x].type = 2;
		  x = (primitives[index].p0.x/2+0.5f)*(int)resolution.x+1;
		  y = (primitives[index].p0.y/2+0.5f)*(int)resolution.y+1;
		  depthbuffer[y*(int)resolution.y+x].color = glm::vec3(1.0f,0,0);
		  depthbuffer[y*(int)resolution.y+x].type = 2;
	  }

	  x = (primitives[index].p1.x/2+0.5f)*(int)resolution.x;
	  y = (primitives[index].p1.y/2+0.5f)*(int)resolution.y;
	  buffer_index = y*(int)resolution.y+x;
	  if(depthbuffer[buffer_index].position.z  < getZAtCoordinate(glm::vec3(0,1,0),primitives[index])+0.0001f){
		  depthbuffer[buffer_index].color = glm::vec3(1.0f,0,0);
		  depthbuffer[buffer_index].type = 2;

		  x = (primitives[index].p1.x/2+0.5f)*(int)resolution.x+1;
		  y = (primitives[index].p1.y/2+0.5f)*(int)resolution.y;
		  depthbuffer[y*(int)resolution.y+x].color = glm::vec3(1.0f,0,0);
		  depthbuffer[y*(int)resolution.y+x].type = 2;
		  x = (primitives[index].p1.x/2+0.5f)*(int)resolution.x;
		  y = (primitives[index].p1.y/2+0.5f)*(int)resolution.y+1;
		  depthbuffer[y*(int)resolution.y+x].color = glm::vec3(1.0f,0,0);
		  depthbuffer[y*(int)resolution.y+x].type = 2;
		  x = (primitives[index].p1.x/2+0.5f)*(int)resolution.x+1;
		  y = (primitives[index].p1.y/2+0.5f)*(int)resolution.y+1;
		  depthbuffer[y*(int)resolution.y+x].color = glm::vec3(1.0f,0,0);
		  depthbuffer[y*(int)resolution.y+x].type = 2;
	  }
  }
}

//Draw Lines
__global__ void rasterizationKernelLines(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  //back face culling
	  glm::vec3 v1 = primitives[index].p1 - primitives[index].p0;
	  glm::vec3 v2 = primitives[index].p2 - primitives[index].p1;
	  glm::vec3 tmp = glm::cross(v1,v2);
	  if(tmp.z>0)
		  return;

	  glm::vec3 p1,p2,p;
	  float len;
	  p1 = primitives[index].p0;
	  p2 = primitives[index].p1;
	  len = glm::length(p1-p2);
	  for(float t=0;t<1;t+=2/(len*resolution.x)){
		  p = (p2-p1)*t + p1;
		  int x = (p.x/2+0.5f)*(int)resolution.x;
		  int y = (p.y/2+0.5f)*(int)resolution.y;
		  int buffer_index = y*(int)resolution.y+x;
		  glm::vec3 coord = calculateBarycentricCoordinate (primitives[index], glm::vec2(p.x,p.y));
		  float depth = getZAtCoordinate(coord,primitives[index]);
		  if(depthbuffer[buffer_index].position.z  < depth+0.0001f){
			  depthbuffer[buffer_index].color = glm::vec3(1.0f,1.0f,0);
			  depthbuffer[buffer_index].type = 3;
		  }
	  }
	  p1 = primitives[index].p1;
	  p2 = primitives[index].p2;
	  len = glm::length(p1-p2);
	  for(float t=0;t<1;t+=2/(len*resolution.x)){
		  p = (p2-p1)*t + p1;
		  int x = (p.x/2+0.5f)*(int)resolution.x;
		  int y = (p.y/2+0.5f)*(int)resolution.y;
		  int buffer_index = y*(int)resolution.y+x;
		  glm::vec3 coord = calculateBarycentricCoordinate (primitives[index], glm::vec2(p.x,p.y));
		  float depth = getZAtCoordinate(coord,primitives[index]);
		  if(depthbuffer[buffer_index].position.z  < depth+0.0001f){
			  depthbuffer[buffer_index].color = glm::vec3(1.0f,1.0f,0);
			  depthbuffer[buffer_index].type = 3;
		  }
	  }
	  p1 = primitives[index].p2;
	  p2 = primitives[index].p0;
	  len = glm::length(p1-p2);
	  for(float t=0;t<1;t+=2/(len*resolution.x)){
		  p = (p2-p1)*t + p1;
		  int x = (p.x/2+0.5f)*(int)resolution.x;
		  int y = (p.y/2+0.5f)*(int)resolution.y;
		  int buffer_index = y*(int)resolution.y+x;
		  glm::vec3 coord = calculateBarycentricCoordinate (primitives[index], glm::vec2(p.x,p.y));
		  float depth = getZAtCoordinate(coord,primitives[index]);
		  if(depthbuffer[buffer_index].position.z  < depth+0.0001f){
			  depthbuffer[buffer_index].color = glm::vec3(1.0f,1.0f,0);
			  depthbuffer[buffer_index].type = 3;
		  }
	  }
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution,bool back){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){

	  if(back){
	  //back face culling
		  glm::vec3 v1 = primitives[index].p1 - primitives[index].p0;
		  glm::vec3 v2 = primitives[index].p2 - primitives[index].p1;
		  glm::vec3 tmp = glm::cross(v1,v2);
		  if(tmp.z>0)
			  return;
	  }

	  float maxX,maxY,minX,minY;
	  if(primitives[index].p0.x>primitives[index].p1.x){
		  maxX = primitives[index].p0.x;
		  minX = primitives[index].p1.x;
	  }else{
		  maxX = primitives[index].p1.x;
		  minX = primitives[index].p0.x;
	  }
	  maxX = primitives[index].p2.x > maxX ? primitives[index].p2.x : maxX;
	  minX = primitives[index].p2.x < minX ? primitives[index].p2.x : minX;

	  if(primitives[index].p0.y>primitives[index].p1.y){
		  maxY = primitives[index].p0.y;
		  minY = primitives[index].p1.y;
	  }else{
		  maxY = primitives[index].p1.y;
		  minY = primitives[index].p0.y;
	  }
	  maxY = primitives[index].p2.y > maxY ? primitives[index].p2.y : maxY;
	  minY = primitives[index].p2.y < minY ? primitives[index].p2.y : minY;

	  maxX = maxX>1.0f? 1.0f:maxX;
	  maxY = maxY>1.0f? 1.0f:maxY;
	  minX = minX<-1.0f? -1.0f:minX;
	  minY = minY<-1.0f? -1.0f:minY;
	  int leftX = (minX/2+0.5f) * resolution.x;
	  int rightX = (maxX/2+0.5f) * resolution.x;
	  int topY = (maxY/2+0.5f) * resolution.y;
	  int bottomY = (minY/2+0.5f) * resolution.y;

	  for(int i = bottomY; i<= topY; i++)
		  for(int j = leftX;j <=rightX;j++){
			glm::vec3 coord = calculateBarycentricCoordinate (primitives[index], glm::vec2(j/resolution.x*2-1.0f,i/resolution.y*2-1.0f));
			int buffer_index= i*(int)resolution.y+j;
			if (isBarycentricCoordInBounds (coord))
			{
				float depth = getZAtCoordinate(coord,primitives[index]);
				if(depthbuffer[buffer_index].position.z  < depth){
					depthbuffer[buffer_index].position.z = depth;
					depthbuffer[buffer_index].color =  coord.x * primitives[index].c0 + coord.y * primitives[index].c1 + coord.z * primitives[index].c2;
					depthbuffer[buffer_index].normal =  coord.x * primitives[index].n0 + coord.y * primitives[index].n1 + coord.z * primitives[index].n2;
					depthbuffer[buffer_index].texcoord =  coord.x * primitives[index].t0 + coord.y * primitives[index].t1 + coord.z * primitives[index].t2;
					depthbuffer[buffer_index].type = 1;
				}
			}
			
		  }
  }
}



//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 *lightDir, bmp_texture *tex,glm::vec3 *device_data){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  if(depthbuffer[index].type !=1)
		  return;
	  float diffuse = glm::dot(-*lightDir,depthbuffer[index].normal);
	  diffuse = diffuse>0?diffuse:0;
	  int x = depthbuffer[index].texcoord.x * tex->width;
	  int y = depthbuffer[index].texcoord.y * tex->height;
	  glm::vec3 tex_color0 = device_data[y * tex->height + x];
	  glm::vec3 tex_color1 = device_data[y * tex->height + x+1];
	  glm::vec3 tex_color2 = device_data[(y+1) * tex->height + x];
	  glm::vec3 tex_color3 = device_data[(y+1) * tex->height + x+1];

	  float xx = depthbuffer[index].texcoord.x * tex->width - x;
	  float yy = depthbuffer[index].texcoord.y * tex->height - y;
	  glm::vec3 tex_color = (tex_color0 * (1-xx) + tex_color1 * xx) * (1-yy) + (tex_color2 * (1-xx) + tex_color3 * xx) * yy;
	  depthbuffer[index].color = tex_color*diffuse*0.9f+tex_color*0.1f;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo,int nbosize,bmp_texture *tex,vector<glm::vec4> *texcoord,float theTa, float alpha,bool line, bool vertex,bool back){

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

  device_tbo = NULL;
  cudaMalloc((void**)&device_tbo, vbosize/3*2*sizeof(float));

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  glm::mat4 projection = glm::perspective(60.0f, resolution.y/resolution.x, 0.001f, 300.0f);
  glm::mat4 camera = glm::lookAt(glm::vec3(cos(theTa)*alpha,alpha,sin(theTa)*alpha), glm::vec3(0,0,0), glm::vec3(0, 1, 0));
  projection = projection * camera;
  glm::mat4 *device_mvp = NULL;
  cudaMalloc((void**)&device_mvp, cbosize*sizeof(glm::mat4));
  cudaMemcpy( device_mvp, &projection, cbosize*sizeof(glm::mat4), cudaMemcpyHostToDevice);

  glm::vec3 lightDir = glm::vec3(1,-1,1);
  lightDir = glm::normalize(lightDir);
  glm::vec3 *device_lightDir = NULL;
  cudaMalloc((void**)&device_lightDir, cbosize*sizeof(glm::vec3));
  cudaMemcpy( device_lightDir, &lightDir, cbosize*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  bmp_texture *device_tex = NULL;
  cudaMalloc((void**)&device_tex, sizeof(bmp_texture));
  cudaMemcpy( device_tex, tex, sizeof(bmp_texture), cudaMemcpyHostToDevice);
  glm::vec3 *device_data = NULL;
  cudaMalloc((void**)&device_data, tex->width * tex->height *sizeof(glm::vec3));
  cudaMemcpy( device_data, tex->data, tex->width * tex->height *sizeof(glm::vec3), cudaMemcpyHostToDevice);


  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_nbo, device_tbo,vbosize, device_mvp);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize,device_tbo, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution,back);
  cudaDeviceSynchronize();
  if(line){
	  rasterizationKernelLines<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);
	  cudaDeviceSynchronize();
  }
  if(vertex){
	  rasterizationKernelVertices<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);
	  cudaDeviceSynchronize();
  }
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, device_lightDir,device_tex,device_data);

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
  cudaFree( device_nbo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

