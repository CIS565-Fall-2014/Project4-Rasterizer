// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania


#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include "rasterizeKernels.h"
//#include "rasterizeTools.h"

#define CullingFlag

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* locvbo;
float* device_cbo;
float* device_nbo;
int* device_ibo;
float* dBuff;
int* dBuffLock;
triangle* primitives;

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__device__ bool DevepsilonCheck(float a, float b){
	if (fabs(fabs(a) - fabs(b))<.000000001){
		return true;
	}
	else{
		return false;
	}
}

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
	if (x < resolution.x && y < resolution.y){
		int index = (y*resolution.x) + x;
		depthbuffer[index] = frag;
	}
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
	if (x < resolution.x && y < resolution.y){
		int index = (y*resolution.x) + x;
		return depthbuffer[index];
	}
	else{
		fragment f;
		return f;
	}
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
	if (x < resolution.x && y < resolution.y){
		int index = (y*resolution.x) + x;
		framebuffer[index] = value;
	}
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
	if (x < resolution.x && y < resolution.y){
		int index = (y*resolution.x) + x;
		return framebuffer[index];
	}
	else{
		return glm::vec3(0, 0, 0);
	}
}

__host__ __device__ float lcalculateSignedArea(triangle tri){
	return 0.5*((tri.p1.x - tri.p0.x)*(tri.p2.y - tri.p0.y) - (tri.p2.x - tri.p0.x)*(tri.p1.y - tri.p0.y));
}

__host__ __device__ float lcalculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri){
	triangle baryTri;
	baryTri.p0 = glm::vec3(a, 0); baryTri.p1 = glm::vec3(b, 0); baryTri.p2 = glm::vec3(c, 0);
	return lcalculateSignedArea(baryTri) / lcalculateSignedArea(tri);
}

__host__ __device__ glm::vec3 lcalculateBarycentricCoordinate(triangle tri, glm::vec2 point){
	float beta = lcalculateBarycentricCoordinateValue(glm::vec2(tri.p0.x, tri.p0.y), point, glm::vec2(tri.p2.x, tri.p2.y), tri);
	float gamma = lcalculateBarycentricCoordinateValue(glm::vec2(tri.p0.x, tri.p0.y), glm::vec2(tri.p1.x, tri.p1.y), point, tri);
	float alpha = 1.0 - beta - gamma;
	return glm::vec3(alpha, beta, gamma);
}

__host__ __device__ bool lisBarycentricCoordInBounds(glm::vec3 barycentricCoord){
	return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
		barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
		barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

__host__ __device__ void lgetAABBForTriangle(triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint){
	minpoint = glm::vec3(min(min(tri.p0.x, tri.p1.x), tri.p2.x),
		min(min(tri.p0.y, tri.p1.y), tri.p2.y),
		min(min(tri.p0.z, tri.p1.z), tri.p2.z));
	maxpoint = glm::vec3(max(max(tri.p0.x, tri.p1.x), tri.p2.x),
		max(max(tri.p0.y, tri.p1.y), tri.p2.y),
		max(max(tri.p0.z, tri.p1.z), tri.p2.z));
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if (x <= resolution.x && y <= resolution.y){
		image[index] = color;
	}
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(Camera cam, fragment* buffer, fragment frag,float* dBuff, int* dBuffLock){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	glm::vec2 resolution = cam.reso;
	int index = x + (y * resolution.x);
	if (x <= resolution.x && y <= resolution.y){
		fragment f = frag;
		f.position.x = x;
		f.position.y = y;
		buffer[index] = f;

		dBuff[index] = cam.depth.y;
		dBuffLock[index] = 0;
	}
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if (x <= resolution.x && y <= resolution.y){

		glm::vec3 color;
		color.x = image[index].x*255.0;
		color.y = image[index].y*255.0;
		color.z = image[index].z*255.0;

		if (color.x > 255){
			color.x = 255;
		}

		if (color.y > 255){
			color.y = 255;
		}

		if (color.z > 255){
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
__global__ void vertexShadeKernel(Camera cam, float* vbo, int vbosize, float* nbo, int nbosize, float* locvbo){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < vbosize / 3){
		glm::vec4 v(vbo[3 * index], vbo[3 * index + 1], vbo[3 * index + 2], 1.0f);
		glm::vec4 n(nbo[3 * index], nbo[3 * index + 1], nbo[3 * index + 2], 1.0f);
		v = cam.World2LocalMat*v;
		locvbo[3 * index] = v.x;
		locvbo[3 * index + 1] = v.y;
		locvbo[3 * index + 2] = v.z;

		v = cam.PMat*v;
		v /= v.w;
		vbo[3 * index] = cam.reso.x*.5f*(v.x + 1.0f);
		vbo[3 * index + 1] = cam.reso.y*.5f*(v.y + 1.0f);
		vbo[3 * index + 2] = (cam.depth.y - cam.depth.x)*.5f*v.z + (cam.depth.y + cam.depth.x)*.5f;

		n = glm::transpose(cam.local2WorldMat)*n;
		nbo[3 * index] = n.x;
		nbo[3 * index + 1] = n.y;
		nbo[3 * index + 2] = n.z;
	}
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float * locvbo, float*nbo,Camera cam){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize / 3;
	if (index < primitivesCount){
		//premitive assign
		primitives[index].p0.x = vbo[3 * ibo[3 * index]];
		primitives[index].p0.y = vbo[3 * ibo[3 * index] + 1];
		primitives[index].p0.z = vbo[3 * ibo[3 * index] + 2];

		primitives[index].p1.x = vbo[3 * ibo[3 * index + 1]];
		primitives[index].p1.y = vbo[3 * ibo[3 * index + 1] + 1];
		primitives[index].p1.z = vbo[3 * ibo[3 * index + 1] + 2];

		primitives[index].p2.x = vbo[3 * ibo[3 * index + 2]];
		primitives[index].p2.y = vbo[3 * ibo[3 * index + 2] + 1];
		primitives[index].p2.z = vbo[3 * ibo[3 * index + 2] + 2];
		//local coordinates assign
		primitives[index].locp0.x = locvbo[3 * ibo[3 * index]];
		primitives[index].locp0.y = locvbo[3 * ibo[3 * index] + 1];
		primitives[index].locp0.z = locvbo[3 * ibo[3 * index] + 2];

		primitives[index].locp1.x = locvbo[3 * ibo[3 * index + 1]];
		primitives[index].locp1.y = locvbo[3 * ibo[3 * index + 1] + 1];
		primitives[index].locp1.z = locvbo[3 * ibo[3 * index + 1] + 2];

		primitives[index].locp2.x = locvbo[3 * ibo[3 * index + 2]];
		primitives[index].locp2.y = locvbo[3 * ibo[3 * index + 2] + 1];
		primitives[index].locp2.z = locvbo[3 * ibo[3 * index + 2] + 2];
		//normal assign
		primitives[index].locn0.x = nbo[3 * ibo[3 * index]];
		primitives[index].locn0.y = nbo[3 * ibo[3 * index] + 1];
		primitives[index].locn0.z = nbo[3 * ibo[3 * index] + 2];

		primitives[index].locn1.x = nbo[3 * ibo[3 * index + 1]];
		primitives[index].locn1.y = nbo[3 * ibo[3 * index + 1] + 1];
		primitives[index].locn1.z = nbo[3 * ibo[3 * index + 1] + 2];

		primitives[index].locn2.x = nbo[3 * ibo[3 * index + 2]];
		primitives[index].locn2.y = nbo[3 * ibo[3 * index + 2] + 1];
		primitives[index].locn2.z = nbo[3 * ibo[3 * index + 2] + 2];

		//color assign
		primitives[index].c0.x = cbo[0];
		primitives[index].c0.y = cbo[1];
		primitives[index].c0.z = cbo[2];

		primitives[index].c1.x = cbo[3];
		primitives[index].c1.y = cbo[4];
		primitives[index].c1.z = cbo[5];

		primitives[index].c2.x = cbo[6];
		primitives[index].c2.y = cbo[7];
		primitives[index].c2.z = cbo[8];

		primitives[index].CFlag = false;

#ifdef CullingFlag
		if (lcalculateSignedArea(primitives[index])< -1e-6) primitives[index].CFlag = true; // back facing triangles
		else    // triangles totally outside of screen
		{
			glm::vec3 tMin, tMax;
			glm::vec2 resolution(cam.reso);
			lgetAABBForTriangle(primitives[index], tMin, tMax);
			if (tMin.x > resolution.x ||
				tMin.y > resolution.y ||
				tMin.z > cam.depth.y ||
				tMax.x < 0 ||
				tMax.y < 0 ||
				tMax.z < cam.depth.x)
				primitives[index].CFlag = true;
		}
#endif
	}
}





//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, Camera cam, float* dBuff, int* dBuffLock){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < primitivesCount){
		glm::vec2 resolution = cam.reso;
		triangle curTri = primitives[index];

		if (DevepsilonCheck(lcalculateSignedArea(curTri), 0.0f)) return;
		else{
			glm::vec3 tMin, tMax;
			lgetAABBForTriangle(curTri, tMin, tMax);

			if (tMin.x > resolution.x ||
				tMin.y > resolution.y ||
				tMin.z > cam.depth.y ||
				tMax.x < 0 ||
				tMax.y < 0 ||
				tMax.z < cam.depth.x)
				return;

			glm::vec2 PC;
			float depth;
			glm::vec3 BC;
			int pixelIndex;
			for (int j = max(int(tMin.y), 0); j < min(int(tMax.y + 1), int(resolution.y)); j++)
			{
				glm::vec2 Q0(tMin.x, float(j + 0.5));
				glm::vec2 Q1(tMax.x, float(j + 0.5));
				glm::vec2 u = Q1 - Q0;
				float s;
				float t;
				float minS = 1.0f, maxS = 0.0f;

				glm::vec3 p( curTri.p1 - curTri.p0);
				glm::vec2 v0(p.x,p.y);
				p = curTri.p2 - curTri.p1;
				glm::vec2 v1(p.x, p.y);
				p = curTri.p0 - curTri.p2;
				glm::vec2 v2(p.x,p.y);

				glm::vec2 w;
				if (!DevepsilonCheck(u.x*v0.y - u.y*v0.x, 0)) // not parallel
				{
					w = Q0 - glm::vec2(curTri.p0.x, curTri.p0.y);
					s = (v0.y*w.x - v0.x*w.y) / (v0.x*u.y - v0.y*u.x);
					t = (u.x*w.y - u.y*w.x) / (u.x*v0.y - u.y*v0.x);
					if (s > -1e-6 && s < 1 + 1e-6 && t > -1e-6 && t < 1 + 1e-6)
					{
						minS = fminf(s, minS);
						maxS = fmaxf(s, maxS);
					}
				}

				if (!DevepsilonCheck(u.x*v1.y - u.y*v1.x, 0)) // not parallel
				{
					w = Q0 - glm::vec2(curTri.p1.x, curTri.p1.y);
					s = (v1.y*w.x - v1.x*w.y) / (v1.x*u.y - v1.y*u.x);
					t = (u.x*w.y - u.y*w.x) / (u.x*v1.y - u.y*v1.x);
					if (s > -1e-6 && s < 1 + 1e-6 && t > -1e-6 && t < 1 + 1e-6)
					{
						minS = fminf(s, minS);
						maxS = fmaxf(s, maxS);
					}
				}

				if (!DevepsilonCheck(u.x*v2.y - u.y*v2.x, 0)) // not parallel
				{
					w = Q0 - glm::vec2(curTri.p2.x, curTri.p2.y);
					s = (v2.y*w.x - v2.x*w.y) / (v2.x*u.y - v2.y*u.x);
					t = (u.x*w.y - u.y*w.x) / (u.x*v2.y - u.y*v2.x);
					if (s > -1e-6 && s < 1 + 1e-6 && t > -1e-6 && t < 1 + 1e-6)
					{
						minS = fminf(s, minS);
						maxS = fmaxf(s, maxS);
					}
				}

				for (int i = max(int(tMin.x + minS * u.x), 0); i < min(int(tMin.x + maxS * u.x + 1), int(resolution.x)); ++i)
				{
					PC = glm::vec2(float(i + 0.5), float(j + 0.5));
					BC = lcalculateBarycentricCoordinate(curTri, PC);
					depth = BC.x * curTri.p0.z + BC.y * curTri.p1.z + BC.z * curTri.p2.z;
					pixelIndex = resolution.x - 1 - i + ((resolution.y - 1 - j) * resolution.x);

					if (lisBarycentricCoordInBounds(BC) && depth > cam.depth.x && depth < cam.depth.y)
					{
						bool wait = true;

						while (wait)
						{
							if (0 == atomicExch(&dBuffLock[pixelIndex], 1))
							{

								if (depth < dBuff[pixelIndex])
								{
									dBuff[pixelIndex] = depth;

									depthbuffer[pixelIndex].position = BC.x * curTri.locp0 + BC.y * curTri.locp1 + BC.z * curTri.locp2;
									depthbuffer[pixelIndex].normal = BC.x * curTri.locn0 + BC.y * curTri.locn1 + BC.z * curTri.locn2;
									depthbuffer[pixelIndex].color = BC.x * curTri.c0 + BC.y * curTri.c1 + BC.z * curTri.c2;
								}
								dBuffLock[pixelIndex] = 0;
								wait = false;
							}
						}
					}
				}
			}


		}
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lPos){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if (x <= resolution.x && y <= resolution.y){
		float specular = 10.0;
		float ka = 0.3;
		float kd = 0.7;
		float ks = 0.1;
		fragment curFrag = depthbuffer[index];

		glm::vec3 L = glm::normalize(lPos - curFrag.position);
		glm::vec3 normal = glm::normalize(curFrag.normal);
		float diffuseTerm = glm::clamp(glm::dot(normal, L), 0.0f, 1.0f);

		glm::vec3 V = glm::normalize(-curFrag.position);
		glm::vec3 H = (L + V) / 2.0f;

		float specularTerm = pow(fmaxf(glm::dot(normal, H), 0.0f), specular);

		depthbuffer[index].color = ka*curFrag.color + kd*curFrag.color*diffuseTerm + ks*specularTerm;
	}
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if (x <= resolution.x && y <= resolution.y){
		framebuffer[index] = depthbuffer[index].color;
	}
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, Camera cam, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize){

	glm::vec2 resolution = cam.reso;
	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

	//set up framebuffer
	framebuffer = NULL;
	cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));

	//set up depthbuffer
	depthbuffer = NULL;
	cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

	//kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
	clearImage << <fullBlocksPerGrid, threadsPerBlock >> >(resolution, framebuffer, glm::vec3(0, 0, 0));

	

	//------------------------------
	//memory stuff
	//------------------------------
	primitives = NULL;
	cudaMalloc((void**)&primitives, (ibosize / 3)*sizeof(triangle));

	device_ibo = NULL;
	cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
	cudaMemcpy(device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

	device_vbo = NULL;
	cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
	cudaMemcpy(device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

	device_cbo = NULL;
	cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
	cudaMemcpy(device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

	device_nbo = NULL;
	cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
	cudaMemcpy(device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

	locvbo = NULL;
	cudaMalloc((void**)&locvbo, vbosize*sizeof(float));

	dBuff = NULL;
	cudaMalloc((void**)&dBuff, int(cam.reso.x*cam.reso.y)*sizeof(float));

	dBuffLock = NULL;
	cudaMalloc((void**)&dBuffLock, int(cam.reso.x*cam.reso.y)*sizeof(int));

	fragment frag;
	frag.color = glm::vec3(0, 0, 0);
	frag.normal = glm::vec3(0, 0, 0);
	frag.position = glm::vec3(0, 0, -10000);
	clearDepthBuffer << <fullBlocksPerGrid, threadsPerBlock >> >(cam, depthbuffer, frag, dBuff, dBuffLock);

	tileSize = 32;
	int primitiveBlocks = ceil(((float)vbosize / 3) / ((float)tileSize));

	//------------------------------
	//vertex shader
	//------------------------------
	vertexShadeKernel << <primitiveBlocks, tileSize >> >(cam, device_vbo, vbosize, device_nbo, nbosize, locvbo);

	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize / 3) / ((float)tileSize));
	primitiveAssemblyKernel << <primitiveBlocks, tileSize >> >(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, locvbo, device_nbo,cam);

	cudaDeviceSynchronize();

#ifdef CullingFlag
	thrust::device_ptr<triangle> primitive_first = thrust::device_pointer_cast(primitives);
	thrust::device_ptr<triangle> primitive_last = thrust::remove_if(primitive_first, primitive_first + ibosize / 3, CFlagTrue());
	printf("Before Culling: %d\n", ibosize / 3);
	int triCount = thrust::distance(primitive_first, primitive_last);
	printf("After Culling: %d\n", triCount);
	cudaDeviceSynchronize();
#endif


	//------------------------------
	//rasterization
	//------------------------------
	rasterizationKernel << <primitiveBlocks, tileSize >> >(primitives, ibosize / 3, depthbuffer, cam, dBuff, dBuffLock);

	cudaDeviceSynchronize();
	//------------------------------
	//fragment shader
	//------------------------------
	fragmentShadeKernel << <fullBlocksPerGrid, threadsPerBlock >> >(depthbuffer, resolution, glm::vec3(-10.0f));

	cudaDeviceSynchronize();
	//------------------------------
	//write fragments to framebuffer
	//------------------------------
	render << <fullBlocksPerGrid, threadsPerBlock >> >(resolution, depthbuffer, framebuffer);
	sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> >(PBOpos, resolution, framebuffer);

	cudaDeviceSynchronize();

	kernelCleanup();

	checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
	cudaFree(primitives);
	cudaFree(device_vbo);
	cudaFree(device_cbo);
	cudaFree(device_ibo);
	cudaFree(framebuffer);
	cudaFree(depthbuffer);
	cudaFree(device_nbo);
	cudaFree(locvbo);
	cudaFree(dBuff);
	cudaFree(dBuffLock);
}

