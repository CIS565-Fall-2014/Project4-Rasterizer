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
fragment* depthbufferPre;
bool* cudaFlag;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
int* device_lineibo;
triangle* primitives;
line* lineCollection;

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

__global__ void clearDepthBufferPre(glm::vec2 resolution, fragment* bufferPre, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * 2*resolution.x);
    if(x<=resolution.x * 2 && y<=resolution.y * 2){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      bufferPre[index] = f;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, cudaMat4 shaderMatrix, int translateX, int translateY){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < vbosize / 3){


	  glm::vec3 afterShader = multiplyMV(shaderMatrix, glm::vec4(vbo[3*index], vbo[3*index+1], vbo[3*index+2], 1.0f));
	  vbo[3*index] = afterShader.x + translateX;
	  vbo[3*index + 1] = afterShader.y + translateY;
	  vbo[3*index + 2] = afterShader.z + 1000;


  }
}


//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index < primitivesCount){
	  //primitives[index].c0.x = cbo[3 * ibo[3 * index + 0] + 0];//cbo[0]
	  //primitives[index].c0.y = cbo[3 * ibo[3 * index + 0] + 1];//cbo[1]
	  //primitives[index].c0.z = cbo[3 * ibo[3 * index + 0] + 2];//cbo[2]
	  //primitives[index].c1.x = cbo[3 * ibo[3 * index + 1] + 0];//cbo[3]
	  //primitives[index].c1.y = cbo[3 * ibo[3 * index + 1] + 1];//cbo[4]
	  //primitives[index].c1.z = cbo[3 * ibo[3 * index + 1] + 2];//cbo[5]
	  //primitives[index].c2.x = cbo[3 * ibo[3 * index + 2] + 0];//cbo[6]
	  //primitives[index].c2.y = cbo[3 * ibo[3 * index + 2] + 1];//cbo[7]
	  //primitives[index].c2.z = cbo[3 * ibo[3 * index + 2] + 2];//cbo[8]

	  primitives[index].p0.x = vbo[3 * ibo[3 * index + 0] + 0];
	  primitives[index].p0.y = vbo[3 * ibo[3 * index + 0] + 1];
	  primitives[index].p0.z = vbo[3 * ibo[3 * index + 0] + 2];
	  primitives[index].p1.x = vbo[3 * ibo[3 * index + 1] + 0];
	  primitives[index].p1.y = vbo[3 * ibo[3 * index + 1] + 1];
	  primitives[index].p1.z = vbo[3 * ibo[3 * index + 1] + 2];
	  primitives[index].p2.x = vbo[3 * ibo[3 * index + 2] + 0];
	  primitives[index].p2.y = vbo[3 * ibo[3 * index + 2] + 1];
	  primitives[index].p2.z = vbo[3 * ibo[3 * index + 2] + 2];

	  primitives[index].n0.x = nbo[3 * ibo[3 * index + 0] + 0];
	  primitives[index].n0.y = nbo[3 * ibo[3 * index + 0] + 1];
	  primitives[index].n0.z = nbo[3 * ibo[3 * index + 0] + 2];
	  primitives[index].n1.x = nbo[3 * ibo[3 * index + 1] + 0];
	  primitives[index].n1.y = nbo[3 * ibo[3 * index + 1] + 1];
	  primitives[index].n1.z = nbo[3 * ibo[3 * index + 1] + 2];
	  primitives[index].n2.x = nbo[3 * ibo[3 * index + 2] + 0];
	  primitives[index].n2.y = nbo[3 * ibo[3 * index + 2] + 1];
	  primitives[index].n2.z = nbo[3 * ibo[3 * index + 2] + 2];

  }
}

__global__ void lineAssemblyKernel(float* vbo, int vbosize, int* lineIbo, int lineIbosize, line* lineCollection){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int lineCount = lineIbosize / 2;
	if(index < lineCount){
		lineCollection[index].p0.x = vbo[3 * lineIbo[2 * index + 0] + 0];
		lineCollection[index].p0.y = vbo[3 * lineIbo[2 * index + 0] + 1];
		lineCollection[index].p0.z = vbo[3 * lineIbo[2 * index + 0] + 2];
		lineCollection[index].p1.x = vbo[3 * lineIbo[2 * index + 1] + 0];
		lineCollection[index].p1.y = vbo[3 * lineIbo[2 * index + 1] + 1];
		lineCollection[index].p1.z = vbo[3 * lineIbo[2 * index + 1] + 2];
	}
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel_Antialiasing(triangle* primitives, int primitivesCount, fragment* depthbufferPre, glm::vec2 resolution, bool* lockFlag, glm::vec3 eye, bool backCulling){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){

		glm::vec3 normal =  primitives[index].n0;// glm::normalize(glm::cross(primitives[index].p1 - primitives[index].p0, primitives[index].p2 - primitives[index].p0));

		//Back-face culling
		if(backCulling == true){
			if(glm::dot(eye, normal) > 0)
				return;
	    }
	    glm::vec3 minPoint;
	    glm::vec3 maxPoint;
	    
	    
	    getAABBForTriangle(primitives[index], minPoint, maxPoint);
	    
	    for(int j = 2*max( (int)floor(minPoint.y)-1, 0); j < 2*min( (int)ceil(maxPoint.y)+1, (int)resolution.y ); ++j){
		    for(int i = 2*max( (int)floor(minPoint.x)-1, 0); i < 2*min( (int)ceil(maxPoint.x)+1, (int)resolution.x); ++i){


				glm::vec3 barycentricCoord = calculateBarycentricCoordinate(primitives[index], glm::vec2(i * 0.5, j * 0.5));
				if(barycentricCoord.x < 0 || barycentricCoord.y < 0 || barycentricCoord.z < 0)
			  		continue;

				float newDepth = -getZAtCoordinate(barycentricCoord, primitives[index]);


				float old = depthbufferPre[i + j * 2 * (int)resolution.x].position.z;
				float assumed;
				do{
					assumed = old;

					if(assumed == depthbufferPre[i + j * 2 * (int)resolution.x].position.z){
						if(newDepth > depthbufferPre[i + j * 2 * (int)resolution.x].position.z){
							depthbufferPre[i + j * 2 * (int)resolution.x].position.z = newDepth;
							depthbufferPre[i + j * 2 * (int)resolution.x].normal = normal;
							depthbufferPre[i + j * 2 * (int)resolution.x].color.x = abs(normal.x);
							depthbufferPre[i + j * 2 * (int)resolution.x].color.y = abs(normal.y);
							depthbufferPre[i + j * 2 * (int)resolution.x].color.z = abs(normal.z);
						}
					}
					else{
						old =depthbufferPre[i + j * 2 * (int)resolution.x].position.z;
					}
				}
				while(assumed != old);


			}
		}
	}
}

__global__ void reAssemble(glm::vec2 resolution, fragment* depthbufferPre, fragment* depthbuffer){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	int preX1 = 2 * x;
	int preX2 = 2 * x + 1;
	int preY1 = 2 * y;
	int preY2 = 2 * y + 1;
	int index1 = preX1 + preY1 * 2 * resolution.x;
	int index2 = preX1 + preY2 * 2 * resolution.x;
	int index3 = preX2 + preY1 * 2 * resolution.x;
	int index4 = preX2 + preY2 * 2 * resolution.x;

	depthbuffer[index].color = (depthbufferPre[index1].color + depthbufferPre[index2].color + depthbufferPre[index3].color + depthbufferPre[index4].color) / (float)4;
	depthbuffer[index].normal = (depthbufferPre[index1].normal + depthbufferPre[index2].normal + depthbufferPre[index3].normal + depthbufferPre[index4].normal) / (float)4;
	depthbuffer[index].position = (depthbufferPre[index1].position + depthbufferPre[index2].position + depthbufferPre[index3].position + depthbufferPre[index4].position) / (float)4;
}

__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, bool* lockFlag, glm::vec3 eye, bool backCulling){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){

		glm::vec3 normal =  primitives[index].n0;

		//Back-face culling
		if(backCulling == true){
			if(glm::dot(eye, normal) > 0)
				return;
	    }
	    glm::vec3 minPoint;
	    glm::vec3 maxPoint;
	    
	    getAABBForTriangle(primitives[index], minPoint, maxPoint);
	    
	    for(int j = max( (int)floor(minPoint.y)-1, 0); j < min( (int)ceil(maxPoint.y)+1, (int)resolution.y ); ++j){
		    for(int i = max( (int)floor(minPoint.x)-1, 0); i < min( (int)ceil(maxPoint.x)+1, (int)resolution.x); ++i){


				glm::vec3 barycentricCoord = calculateBarycentricCoordinate(primitives[index], glm::vec2(i, j));
				if(barycentricCoord.x < 0 || barycentricCoord.y < 0 || barycentricCoord.z < 0)
			  		continue;

				float newDepth = -getZAtCoordinate(barycentricCoord, primitives[index]);


				float old = depthbuffer[i + j * (int)resolution.x].position.z;
				float assumed;
				do{
					assumed = old;

					if(assumed == depthbuffer[i + j * (int)resolution.x].position.z){
						if(newDepth > depthbuffer[i + j * (int)resolution.x].position.z){
							depthbuffer[i + j * (int)resolution.x].position.z = newDepth;
							depthbuffer[i + j * (int)resolution.x].normal = normal;
							depthbuffer[i + j * (int)resolution.x].color.x = abs(normal.x);
							depthbuffer[i + j * (int)resolution.x].color.y = abs(normal.y);
							depthbuffer[i + j * (int)resolution.x].color.z = abs(normal.z);
						}
					}
					else{
						old =depthbuffer[i + j * (int)resolution.x].position.z;
					}
				}
				while(assumed != old);


			}
		}
	}
}

__global__ void rasterizationKernelWireFrameSolid(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, bool* lockFlag, glm::vec3 eye){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){

		glm::vec3 normal =  primitives[index].n0;

		//Back-face culling
		if(glm::dot(eye, normal) > 0)
			return;
	    

		glm::vec3 p0 = primitives[index].p0;
		glm::vec3 p1 = primitives[index].p1;
		glm::vec3 p2 = primitives[index].p2;

		glm::vec3 minPoint;
	    glm::vec3 maxPoint;

	    getAABBForTriangle(primitives[index], minPoint, maxPoint);
	    for(int j = max( (int)floor(minPoint.y)-1, 0); j < min( (int)ceil(maxPoint.y)+1, (int)resolution.y ); ++j){
		    for(int i = max( (int)floor(minPoint.x)-1, 0); i < min( (int)ceil(maxPoint.x)+1, (int)resolution.x); ++i){

				glm::vec3 barycentricCoord = calculateBarycentricCoordinate(primitives[index], glm::vec2(i, j));
				if(barycentricCoord.x < 0 || barycentricCoord.y < 0 || barycentricCoord.z < 0)
			  		continue;

				float newDepth = -getZAtCoordinate(barycentricCoord, primitives[index]);
				if(newDepth > depthbuffer[i + j * (int)resolution.x].position.z){
					depthbuffer[i + j * (int)resolution.x].position.z = newDepth;
				}
				else{
					continue;
				}



				float dis1 = getPtToLineDistance(i, j, p0, p1);
				float dis2 = getPtToLineDistance(i, j, p0, p2);
				float dis3 = getPtToLineDistance(i, j, p1, p2);


				if(dis1 < 0.5 && checkBoundaryForLine(i,j,p0,p1) ){
					depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
					continue;
				}

				if(dis2 < 0.5 && checkBoundaryForLine(i,j,p0,p2) ){
					depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
					continue;
				}

				if(dis3 < 0.5 && checkBoundaryForLine(i,j,p1,p2) ){
					depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
					continue;
				}
				depthbuffer[i + j * (int)resolution.x].color = glm::vec3(0,0,0);
			}
		}
	}
}

__global__ void rasterizationKernelWireFrameDashLine(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, bool* lockFlag, glm::vec3 eye){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){

		glm::vec3 normal =  primitives[index].n0;

		glm::vec3 p0 = primitives[index].p0;
		glm::vec3 p1 = primitives[index].p1;
		glm::vec3 p2 = primitives[index].p2;

		glm::vec3 minPoint;
	    glm::vec3 maxPoint;

	    getAABBForTriangle(primitives[index], minPoint, maxPoint);

		//Back-face culling
		if(glm::dot(eye, normal) > 0){
			for(int j = max( (int)floor(minPoint.y)-1, 0); j < min( (int)ceil(maxPoint.y)+1, (int)resolution.y ); ++j){
				for(int i = max( (int)floor(minPoint.x)-1, 0); i < min( (int)ceil(maxPoint.x)+1, (int)resolution.x); ++i){

					float dis1 = getPtToLineDistance(i, j, p0, p1);
					float dis2 = getPtToLineDistance(i, j, p0, p2);
					float dis3 = getPtToLineDistance(i, j, p1, p2);


					if(dis1 < 0.5 && checkBoundaryForLine(i,j,p0,p1) ){

						float disToStart = sqrt(pow(i - p0.x, 2) + pow(j - p0.y, 2));
						if((int)(disToStart / 5) % 2 == 0 )
							depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
						continue;
					}

					if(dis2 < 0.5 && checkBoundaryForLine(i,j,p0,p2) ){

						float disToStart = sqrt(pow(i - p0.x, 2) + pow(j - p0.y, 2));
						if((int)(disToStart / 5) % 2 == 0 )
							depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);

						continue;
					}

					if(dis3 < 0.5 && checkBoundaryForLine(i,j,p1,p2) ){

						float disToStart = sqrt(pow(i - p1.x, 2) + pow(j - p1.y, 2));
						if((int)(disToStart / 5) % 2 == 0 )
							depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);

						continue;
					}
				}
			}
		}
		else{
			for(int j = max( (int)floor(minPoint.y)-1, 0); j < min( (int)ceil(maxPoint.y)+1, (int)resolution.y ); ++j){
				for(int i = max( (int)floor(minPoint.x)-1, 0); i < min( (int)ceil(maxPoint.x)+1, (int)resolution.x); ++i){




					float dis1 = getPtToLineDistance(i, j, p0, p1);
					float dis2 = getPtToLineDistance(i, j, p0, p2);
					float dis3 = getPtToLineDistance(i, j, p1, p2);


					if(dis1 < 0.5 && checkBoundaryForLine(i,j,p0,p1) ){
						depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
						continue;
					}

					if(dis2 < 0.5 && checkBoundaryForLine(i,j,p0,p2) ){
						depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
						continue;
					}

					if(dis3 < 0.5 && checkBoundaryForLine(i,j,p1,p2) ){
						depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
						continue;
					}
				}
			}
		}
	}
}

__global__ void rasterizationKernelWireFrameRealLine(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, bool* lockFlag, glm::vec3 eye){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){

		glm::vec3 normal =  primitives[index].n0;

		glm::vec3 p0 = primitives[index].p0;
		glm::vec3 p1 = primitives[index].p1;
		glm::vec3 p2 = primitives[index].p2;

		glm::vec3 minPoint;
	    glm::vec3 maxPoint;

	    getAABBForTriangle(primitives[index], minPoint, maxPoint);
	    for(int j = max( (int)floor(minPoint.y)-1, 0); j < min( (int)ceil(maxPoint.y)+1, (int)resolution.y ); ++j){
		    for(int i = max( (int)floor(minPoint.x)-1, 0); i < min( (int)ceil(maxPoint.x)+1, (int)resolution.x); ++i){

				float dis1 = getPtToLineDistance(i, j, p0, p1);
				float dis2 = getPtToLineDistance(i, j, p0, p2);
				float dis3 = getPtToLineDistance(i, j, p1, p2);


				if(dis1 < 0.5 && checkBoundaryForLine(i,j,p0,p1) ){
					depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
					continue;
				}

				if(dis2 < 0.5 && checkBoundaryForLine(i,j,p0,p2) ){
					depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
					continue;
				}

				if(dis3 < 0.5 && checkBoundaryForLine(i,j,p1,p2) ){
					depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
					continue;
				}






			}
		}
	}
}

__global__ void rasterizationKernelWireFramePoint(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, bool* lockFlag, glm::vec3 eye){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){

		glm::vec3 normal =  primitives[index].n0;

		glm::vec3 p0 = primitives[index].p0;
		glm::vec3 p1 = primitives[index].p1;
		glm::vec3 p2 = primitives[index].p2;

		glm::vec3 minPoint;
	    glm::vec3 maxPoint;

	    getAABBForTriangle(primitives[index], minPoint, maxPoint);
	    for(int j = max( (int)floor(minPoint.y)-1, 0); j < min( (int)ceil(maxPoint.y)+1, (int)resolution.y ); ++j){
		    for(int i = max( (int)floor(minPoint.x)-1, 0); i < min( (int)ceil(maxPoint.x)+1, (int)resolution.x); ++i){


				if( (abs(i - p0.x) < 0.5 && abs(j - p0.y) < 0.5) || (abs(i - p1.x) < 0.5 && abs(j - p1.y) < 0.5) || (abs(i - p2.x) < 0.5 && abs(j - p2.y) < 0.5) ){
					depthbuffer[i + j * (int)resolution.x].color = glm::vec3(1,1,1);
					continue;
				}


			}
		}
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightDir, bool scissorTest, glm::vec3 eye){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //int inverseIndex = (resolution.x - x) + ((resolution.y - y) * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	  if(scissorTest == true){
		  if(x >= 300 && x <= 500 && y >=300 && y <= 500)
			  return;
	  }

	  float diffuseTerm = max(glm::dot(depthbuffer[index].normal, lightDir), (float)0.0);

	  int specularExp = 10;
	  glm::vec3 eveVector = (float)-1*glm::normalize(eye - depthbuffer[index].position);
	  glm::vec3 H = (eveVector + lightDir) / (float)2;
	  float specularTerm = max(pow(glm::dot(depthbuffer[index].normal, H), specularExp), (float)0.0);

	  //float colorR = depthbuffer[index].color.x * (diffuseTerm + specularTerm);
	  //float colorG = depthbuffer[index].color.y * (diffuseTerm + specularTerm);
	  //float colorB = depthbuffer[index].color.z * (diffuseTerm + specularTerm);


	  //depthbuffer[index].color.x = depthbuffer[index].color.x;
	  //depthbuffer[index].color.y = depthbuffer[index].color.y;
	  //depthbuffer[index].color.z = depthbuffer[index].color.z;

	  //depthbuffer[index].color.x = depthbuffer[index].position.z;
	  //depthbuffer[index].color.y = depthbuffer[index].position.z;
	  //depthbuffer[index].color.z = depthbuffer[index].position.z;

	  //depthbuffer[index].color.x = abs(depthbuffer[index].normal.x);
	  //depthbuffer[index].color.y = abs(depthbuffer[index].normal.y);
	  //depthbuffer[index].color.z = abs(depthbuffer[index].normal.z);

	  //depthbuffer[index].color.x = diffuseTerm * depthbuffer[index].color.x;
	  //depthbuffer[index].color.y = diffuseTerm * depthbuffer[index].color.y;
	  //depthbuffer[index].color.z = diffuseTerm * depthbuffer[index].color.z;

	  depthbuffer[index].color.x = specularTerm + diffuseTerm;
	  depthbuffer[index].color.y = specularTerm + diffuseTerm;
	  depthbuffer[index].color.z = specularTerm + diffuseTerm;
  }
}


//TODO:
__global__ void colorAlphaBlendingKernel(fragment* depthbuffer, glm::vec2 resolution, float alphaValue){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

		if(depthbuffer[index].color != glm::vec3(0,0,0)){

			glm::vec3 destinationColor;
			int tempX = x / 50;
			int tempY = y / 50;
			if((tempX + tempY) % 2 == 0)
				destinationColor = glm::vec3(0,0,0);
			else
				destinationColor = glm::vec3(1,1,1);

			depthbuffer[index].color = depthbuffer[index].color * alphaValue + destinationColor * (1 - alphaValue);
		}
	}
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int inverseIndex = (resolution.x - x) + ((resolution.y - y) * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[inverseIndex].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
	float* nbo, int nbosize, cudaMat4 shaderMatrix, int translateX, int translateY, glm::vec3 eye, glm::vec3 light, bool alphaBlend, float alphaValue, 
	bool backCulling, bool scissorTest, bool antialiasing, int displayMode){

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

	//set up framebuffer
	framebuffer = NULL;
	cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  

	bool* falg = new bool[ (int)resolution.x*(int)resolution.y];
	cudaFlag = NULL;
	cudaMalloc((void**)&cudaFlag, (int)resolution.x*(int)resolution.y*sizeof(bool));
	cudaMemcpy( cudaFlag, falg, (int)resolution.x*(int)resolution.y*sizeof(bool), cudaMemcpyHostToDevice);

	//set up depthbuffer
	depthbuffer = NULL;
	cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));



	//kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
	clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
	fragment frag;
	frag.color = glm::vec3(0,0,0);
	frag.normal = glm::vec3(0,0,0);
	frag.position = glm::vec3(0,0,-10000);
	clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, frag);

	depthbufferPre = NULL;
	cudaMalloc((void**)&depthbufferPre, 2*2*(int)resolution.x*(int)resolution.y*sizeof(fragment));
	dim3 fullBlocksPerGridForAntiAliasing((int)ceil(2*float(resolution.x)/float(tileSize)), (int)ceil(2*float(resolution.y)/float(tileSize)));



	//------------------------------
	//memory stuff
	//------------------------------
	primitives = NULL;
	cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

	device_ibo = NULL;
	cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
	cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

	device_nbo = NULL;
	cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
	cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

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
	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, shaderMatrix, translateX, translateY);
	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives);
	cudaDeviceSynchronize();

	if(displayMode == 0){
		//------------------------------
		//rasterization
		//------------------------------
		if(antialiasing == true){
			clearDepthBufferPre<<<fullBlocksPerGridForAntiAliasing, threadsPerBlock>>>(resolution, depthbufferPre,frag);

			rasterizationKernel_Antialiasing<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbufferPre, resolution, cudaFlag, eye, backCulling);
			cudaDeviceSynchronize();

			reAssemble<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbufferPre, depthbuffer);
			cudaDeviceSynchronize();
		}
		else{
			rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cudaFlag, eye, backCulling);
			cudaDeviceSynchronize();
		}

		//------------------------------
		//fragment shader
		//------------------------------
		fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, light, scissorTest, eye);
		cudaDeviceSynchronize();
	}
	//------------------------------
	//wireframe shader
	//------------------------------
	if(displayMode == 1)
		rasterizationKernelWireFrameSolid<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cudaFlag, eye);
	else if(displayMode == 2)
		rasterizationKernelWireFrameDashLine<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cudaFlag, eye);
	else if(displayMode == 3)
		rasterizationKernelWireFrameRealLine<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cudaFlag, eye);
	else if(displayMode == 4)	
		rasterizationKernelWireFramePoint<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, cudaFlag, eye);
	cudaDeviceSynchronize();


	//------------------------------
	//color blending
	//------------------------------
	if(alphaBlend){
		colorAlphaBlendingKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, alphaValue);
		cudaDeviceSynchronize();
	}
	//------------------------------
	//write fragments to framebuffer
	//------------------------------
	render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

	cudaDeviceSynchronize();
	delete [] falg;
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
  cudaFree( depthbufferPre );
  cudaFree( cudaFlag );
  //cudaFree( device_lineibo );
  //cudaFree( lineCollection );
}

