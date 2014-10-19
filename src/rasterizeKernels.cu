// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

//Added
#include "../external/include/glm/gtc/matrix_transform.hpp"
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

extern bool BackfaceCulling;
extern bool BCInterp;
extern bool AntiAliasing;
extern bool LineMode;
extern bool PointMode;
extern bool ShowBody;

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;

//Added
float* device_nbo;
extern int width;
extern int height;

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
__global__ void vertexShadeKernel(float* vbo, int vbosize,float* nbo, int nbosize,glm::mat4 projection, glm::mat4 view,
	glm::mat4 model, glm::vec2 resolution, glm::vec2 nearfar,glm::mat4 normalModelview){
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if(index<vbosize/3){
			glm::vec4 originpos = glm::vec4(vbo[3*index],vbo[3*index+1],vbo[3*index+2],1.0f);
			glm::vec4 originnormal = glm::vec4(nbo[3*index],nbo[3*index+1],nbo[3*index+2],0.0f);

			//Pclip = (Mprojection)(Mview)(Mmodel)(Pmodel)
			originpos = projection * view * model * originpos;

			//Perspective division
			vbo[3*index] = originpos.x/originpos.w;
			vbo[3*index+1] = originpos.y/originpos.w;
			vbo[3*index+2] = originpos.z/originpos.w;

			//Viewport transform
			vbo[3*index] = resolution.x * (vbo[3*index]/2.0f + 0.5f);
			vbo[3*index+1] = resolution.y  * (vbo[3*index+1]/2.0f + 0.5f);
			vbo[3*index+2] = (nearfar.y - nearfar.x)* (vbo[3*index+2]/2.0f + 0.5f) + nearfar.x;

			//Transform normal
			originnormal = normalModelview * originnormal;
			nbo[3*index] = originnormal.x;
			nbo[3*index+1] = originnormal.y;
			nbo[3*index+2] = originnormal.z;
		}
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo,float* nbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize/3;
	if(index<primitivesCount){
		int p0i = 3 * ibo[3*index];
		int p1i = 3 * ibo[3*index + 1];
		int p2i = 3 * ibo[3*index + 2];

		//setting color
		primitives[index].c0 = glm::vec3(cbo[0],cbo[1],cbo[2]);
		primitives[index].c1 = glm::vec3(cbo[3],cbo[4],cbo[5]);
		primitives[index].c2 = glm::vec3(cbo[6],cbo[7],cbo[8]);

		//pos
		primitives[index].p0 = glm::vec3(vbo[p0i],vbo[p0i+1],vbo[p0i+2]);
		primitives[index].p1 = glm::vec3(vbo[p1i],vbo[p1i+1],vbo[p1i+2]);
		primitives[index].p2 = glm::vec3(vbo[p2i],vbo[p2i+1],vbo[p2i+2]);

		//normal
		primitives[index].n0 = glm::vec3(nbo[p0i],nbo[p0i+1],nbo[p0i+2]);
		primitives[index].n1 = glm::vec3(nbo[p1i],nbo[p1i+1],nbo[p1i+2]);
		primitives[index].n2 = glm::vec3(nbo[p2i],nbo[p2i+1],nbo[p2i+2]);


		//Backfaceculling
		primitives[index].discard = false;
		float eps = 0.00001f;
		if(calculateSignedArea(primitives[index]) < eps) 
			primitives[index].discard = true;	  
	}
}


//Added
__device__ bool Inscreen(glm::vec2 p, glm::vec2 resolution) {
	return (p.x >= 0 && p.x <= resolution.x && p.y >= 0 && p.y <= resolution.y);
}

// Interpolation by Barycentric Coordinates
__device__ glm::vec3 InterpolateBC(glm::vec3 BC, glm::vec3 data1,glm::vec3 data2,glm::vec3 data3) {
	return BC.x * data1+ BC.y * data2 + BC.z * data3;
}

__device__ float crossvec2(glm::vec2 a,glm::vec2 b)
{
	return a.x * b.y - a.y * b.x;
}


__device__ void GetTwoLinesIntersect(glm::vec2 p1,glm::vec2 p2,glm::vec2 q1,glm::vec2 q2, float &mint,float &maxt) {
	glm::vec2 r = p2 - p1;		
	glm::vec2 s = q2 - q1;
	float eps = 0.00001f;
	glm::vec2 w;
	if(abs(crossvec2(r,s)) > eps)
	{
		float t = crossvec2(q1-p1,s)/crossvec2(r,s);
		float u = crossvec2(q1-p1,r)/crossvec2(r,s);	
		if(u > -eps && u < 1+ eps && t > -eps && t < 1+ eps)
		{
			mint = glm::min(t, mint);
			maxt = glm::max(t, maxt);
		}
	}

}

//glm::normalize maybe wrong if vector length is 0, so I add this 
//Besides, don't use .length(), as it is always 3!
__device__ glm::vec3 GetNormal(glm::vec3 origin)
{

	float eps = 0.0000001f;
	if(glm::length(origin)<eps)
		return origin;
	else
		return origin/(float)glm::length(origin);
}

//TODO: Implement a rasterization method, such as scanline.
//Per Primitive
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution,glm::vec2 nearfar,
	bool BCInterp){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		triangle tri = primitives [index];
		glm::vec3 min,max;
		getAABBForTriangle(tri,min,max);
		fragment frag;

		for(int j=min.y;j<max.y+1;j++)
		{			
			float mint = 1.0f, maxt = 0.0f;
			//get triangle intersects with the scanline
			GetTwoLinesIntersect(glm::vec2(min.x, float(j)),glm::vec2(max.x, float(j)),glm::vec2(tri.p1),glm::vec2(tri.p0),mint,maxt);
			GetTwoLinesIntersect(glm::vec2(min.x, float(j)),glm::vec2(max.x, float(j)),glm::vec2(tri.p2),glm::vec2(tri.p1),mint,maxt);			
			GetTwoLinesIntersect(glm::vec2(min.x, float(j)),glm::vec2(max.x, float(j)),glm::vec2(tri.p0),glm::vec2(tri.p2),mint,maxt);

			//minx and maxx on scanline
			int xmin = min.x + mint * (max.x - min.x);
			int xmax = min.x + maxt * (max.x - min.x)+1;

			for(int i=xmin;i<xmax;i++)
			{
				int depthIndex = ((resolution.y - j -1)*resolution.x) + resolution.x - i - 1;	
				if(Inscreen(glm::vec2(i,j),resolution))
				{
					//compute barycentric coords to detect whether the pixel is in the triangle
					glm::vec3 baryCoord = calculateBarycentricCoordinate (tri, glm::vec2 (i,j));
					if (isBarycentricCoordInBounds (baryCoord))
					{
						// Interpolation by Barycentric Coordinates	
						frag.position = InterpolateBC(baryCoord,tri.p0,tri.p1,tri.p2);	
						frag.position.z = -frag.position.z;
						if(BCInterp)
						{
							frag.color =  InterpolateBC(baryCoord,tri.c0,tri.c1,tri.c2);	
							frag.normal = InterpolateBC(baryCoord,tri.n0,tri.n1,tri.n2);
							frag.normal = GetNormal(frag.normal);
						}
						else
						{
							frag.color =  (tri.c0 + tri.c1 + tri.c2)/3.0f;	
							frag.normal = (tri.n0 + tri.n1 + tri.n2)/3.0f; //Not smooth
							frag.normal = GetNormal(frag.normal);
						}


						//only the closest triangle shows
						if (depthbuffer[depthIndex].position.z  < frag.position.z
							&&frag.position.z<-nearfar.x&&frag.position.z>-nearfar.y)
							depthbuffer[depthIndex] = frag;

					}
				}
			}
		}
	}
}

//Show Points
__global__ void PointrasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution,glm::vec2 nearfar){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		glm::vec3 p0 = primitives[index].p0;
		int i = glm::round(p0.x);
		int j = glm::round(p0.y);
		if(Inscreen(glm::vec2(i,j),resolution))
		{
			int depthIndex = ((resolution.y - j -1)*resolution.x) + resolution.x - i - 1;	
			fragment f;
			f.position = glm::vec3(p0.x, p0.y, p0.z);
			f.normal = glm::vec3(0,0,1);
			f.color = glm::vec3(1,1,1);
			if (f.position.z>nearfar.x&&f.position.z<nearfar.y)
				depthbuffer[depthIndex] = f;
		}

		glm::vec3 p1 = primitives[index].p1;
		i = glm::round(p1.x);
		j = glm::round(p1.y);

		if(Inscreen(glm::vec2(i,j),resolution))
		{
			int depthIndex = ((resolution.y - j -1)*resolution.x) + resolution.x - i - 1;	
			fragment f;
			f.position = glm::vec3(p1.x, p1.y, p1.z);
			f.normal = glm::vec3(0,0,1);
			f.color = glm::vec3(1,1,1);
			if (f.position.z>nearfar.x&&f.position.z<nearfar.y)
				depthbuffer[depthIndex] = f;
		}

		glm::vec3 p2 = primitives[index].p2;

		i = glm::round(p2.x);
		j = glm::round(p2.y);

		if(Inscreen(glm::vec2(i,j),resolution))
		{
			int depthIndex = ((resolution.y - j -1)*resolution.x) + resolution.x - i - 1;	
			fragment f;
			f.position = glm::vec3(p2.x, p2.y, p2.z);
			f.normal = glm::vec3(0,0,1);
			f.color = glm::vec3(1,1,1);
			if (f.position.z>nearfar.x&&f.position.z<nearfar.y)
				depthbuffer[depthIndex] = f;
		}
	}
}

//Show Lines
__global__ void LinerasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution,glm::vec2 nearfar){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		triangle tri = primitives [index];
		glm::vec3 min,max;
		getAABBForTriangle(tri,min,max);
		fragment frag;

		for(int j=min.y;j<max.y+1;j++)
		{			
			float mint = 1.0f, maxt = 0.0f;
			//get triangle intersects with the scanline
			GetTwoLinesIntersect(glm::vec2(min.x, float(j)),glm::vec2(max.x, float(j)),glm::vec2(tri.p1),glm::vec2(tri.p0),mint,maxt);
			GetTwoLinesIntersect(glm::vec2(min.x, float(j)),glm::vec2(max.x, float(j)),glm::vec2(tri.p2),glm::vec2(tri.p1),mint,maxt);			
			GetTwoLinesIntersect(glm::vec2(min.x, float(j)),glm::vec2(max.x, float(j)),glm::vec2(tri.p0),glm::vec2(tri.p2),mint,maxt);

			//minx and maxx on scanline
			int xmin = min.x + mint * (max.x - min.x);
			int xmax = min.x + maxt * (max.x - min.x)+1;

			for(int i=xmin;i<xmax;i++)
			{
				int depthIndex = ((resolution.y - j -1)*resolution.x) + resolution.x - i - 1;	
				if(Inscreen(glm::vec2(i,j),resolution))
				{
					//compute barycentric coords to detect whether the pixel is in the triangle
					glm::vec3 baryCoord = calculateBarycentricCoordinate (tri, glm::vec2 (i,j));
					if (!isBarycentricCoordInBounds(baryCoord)) //show lines
					{
						// Interpolation by Barycentric Coordinates	
						frag.position = InterpolateBC(baryCoord,tri.p0,tri.p1,tri.p2);	
						frag.position.z = -frag.position.z;
						frag.color =  glm::vec3(1,0,0);	
						frag.normal = glm::vec3(0,0,1);

						//only the closest triangle shows
						if (frag.position.z<-nearfar.x&&frag.position.z>-nearfar.y)
							depthbuffer[depthIndex] = frag;
					}
				}
			}
		}
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution,glm::vec3 eyepos,glm::vec3 lightpos,
	glm::mat4 projection, glm::mat4 view,glm::mat4 model,glm::vec2 nearfar){
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		int index = x + (y * resolution.x);
		if(x<=resolution.x && y<=resolution.y){	

			glm::vec3 p = depthbuffer [index].position;
			p = glm::vec3((glm::inverse(projection) * glm::inverse(model*view) * glm::vec4(p,1)));

			glm::vec3 L =  GetNormal(lightpos-p);
			glm::vec3 N = GetNormal(depthbuffer[index].normal);
			glm::vec3 E = GetNormal(eyepos - p);
			glm::vec3 H = GetNormal(E + L); 

			float shininess = 100.0f;
			//Blinn-Phong
			float diffusefactor =  glm::clamp(glm::dot(N,L),0.0f,1.0f);
			float specularfactor = glm::pow(glm::max(glm::dot(N,H),0.0f),shininess);

			glm::vec3 lightcolor(1,1,1);
			depthbuffer [index].color = 
				(lightcolor * diffusefactor + lightcolor * specularfactor)*depthbuffer [index].color;
		}
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer,bool AA){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	//AA
	if(AA)
	{
		if(x >= 1 && y >= 1 && x < resolution.x && y < resolution.y)
		{
			glm::vec3 sumc = glm::vec3(0.0f);
			for(int i = -1; i <= 1; i++){
				for(int j = -1; j <= 1; j++)
				{
					index = x + i + ((y+j) * resolution.x);
					sumc += depthbuffer[index].color;
				}
			}
			index = x + (y * resolution.x);
			depthbuffer[index].color = sumc / 9.0f;
		}

		__syncthreads();
	}


	if(x<=resolution.x && y<=resolution.y){
		framebuffer[index] = depthbuffer[index].color;
	}

}

//Stream Compact
struct Is_Exist
{
	__host__ __device__
		bool operator()(const triangle x)
	{
		if(!x.discard) return true;
		else return false;
	}
};

struct Is_Not_Exist
{
	__host__ __device__
		bool operator()(const triangle x)
	{
		if(x.discard) return true;
		else return false;
	}
};

void ThrustStreamCompact(thrust::device_ptr<triangle> origin,int &N)
{
	//Count how many rays still exist
	int finallength = thrust::count_if(origin, origin+N,Is_Exist());
	thrust::remove_if(origin, origin+N,Is_Not_Exist());
	N = finallength;
	return;
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, 
	float* cbo, int cbosize, int* ibo, int ibosize,float* nbo, int nbosize){

		//Added
		extern glm::mat4 model;
		extern glm::vec2 nearfar;
		extern glm::vec3 eye;
		extern glm::mat4 projection;
		extern glm::mat4 view;
		extern glm::vec3 lightpos;
		extern glm::mat4 modelview;
		//modelview for normal
		extern glm::mat4 normalModelview;

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

		//Add Normal
		device_nbo = NULL;
		cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
		cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

		tileSize = 32;
		int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

		//------------------------------
		//vertex shader
		//------------------------------
		vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize,device_nbo, nbosize,projection, view 
			,model,resolution,nearfar,normalModelview);

		cudaDeviceSynchronize();
		//------------------------------
		//primitive assembly
		//------------------------------
		primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
		primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo,device_nbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

		//Stream Compact after Backface Culling
		int numOfPrimitives = ibosize/3;

		if(BackfaceCulling)
		{
			//std::cout<<"Number of primitives before Backface Culling:"<<numOfPrimitives<<std::endl;
			thrust::device_ptr<triangle> primStart(primitives);	
			ThrustStreamCompact(primStart,numOfPrimitives);
			primitiveBlocks = ceil((float)numOfPrimitives/((float)tileSize));
			//std::cout<<"Number of primitives after Backface Culling:"<<numOfPrimitives<<std::endl;
		}


		cudaDeviceSynchronize();
		//------------------------------
		//rasterization
		//------------------------------
		if(ShowBody)
		{
			rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, numOfPrimitives, depthbuffer, resolution,nearfar,BCInterp);		
			cudaDeviceSynchronize();
			//------------------------------
			//fragment shader
			//------------------------------
			fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,eye,
				lightpos,projection, view,model,nearfar);

			cudaDeviceSynchronize();
		}


		//Point Mode
		if(PointMode)
		{
			PointrasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, numOfPrimitives, depthbuffer, resolution,nearfar);
			cudaDeviceSynchronize();
		}

		//Line Mode
		if(LineMode)
		{
			LinerasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, numOfPrimitives, depthbuffer, resolution,nearfar);
			cudaDeviceSynchronize();
		}

		//------------------------------
		//write fragments to framebuffer
		//------------------------------
		render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer,AntiAliasing);
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

