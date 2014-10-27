// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

#define GLM_FORCE_RADIANS
#include <glm/gtx/rotate_vector.hpp>

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_nbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution)
{
    if (x < resolution.x && y < resolution.y) {
        int index = (y * resolution.x) + x;
        depthbuffer[index] = frag;
    }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution)
{
    if (x < resolution.x && y < resolution.y) {
        int index = (y * resolution.x) + x;
        return depthbuffer[index];
    } else {
        fragment f;
        return f;
    }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution)
{
    if (x < resolution.x && y < resolution.y) {
        int index = (y * resolution.x) + x;
        framebuffer[index] = value;
    }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution)
{
    if (x < resolution.x && y < resolution.y) {
        int index = (y * resolution.x) + x;
        return framebuffer[index];
    } else {
        return glm::vec3(0, 0, 0);
    }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
        image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
        fragment f = frag;
        f.position.x = x;
        f.position.y = y;
        buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image)
{

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {

        glm::vec3 color;
        color.x = image[index].x * 255.0;
        color.y = image[index].y * 255.0;
        color.z = image[index].z * 255.0;

        if (color.x > 255) {
            color.x = 255;
        }

        if (color.y > 255) {
            color.y = 255;
        }

        if (color.z > 255) {
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
__global__ void vertexShadeKernel(float* vbo, int vbosize)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vbosize / 3) {
        glm::vec3 p(vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2]);
        //p = glm::rotate(p, 1.f, glm::vec3(0, 1, 0));
        //p = glm::rotate(p, 1.f, glm::vec3(1, 0, 0));
        vbo[index * 3 + 0] = p.x;
        vbo[index * 3 + 1] = p.y;
        vbo[index * 3 + 2] = p.z;
    }
}

__global__ void primitiveAssemblyKernel(
        float* vbo, int vbosize,
        float *nbo, int nbosize,
        float* cbo, int cbosize,
        int* ibo, int ibosize,
        triangle* primitives)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int primitivesCount = ibosize / 3;
    if (index < primitivesCount) {
        int i0 = ibo[3 * index + 0];
        int i1 = ibo[3 * index + 1];
        int i2 = ibo[3 * index + 2];

        triangle prim;

        prim.p0 = glm::vec3(vbo[3 * i0], vbo[3 * i0 + 1], vbo[3 * i0 + 2]);
        prim.p1 = glm::vec3(vbo[3 * i1], vbo[3 * i1 + 1], vbo[3 * i1 + 2]);
        prim.p2 = glm::vec3(vbo[3 * i2], vbo[3 * i2 + 1], vbo[3 * i2 + 2]);

        prim.n0 = glm::vec3(nbo[3 * i0], nbo[3 * i0 + 1], nbo[3 * i0 + 2]);
        prim.n1 = glm::vec3(nbo[3 * i1], nbo[3 * i1 + 1], nbo[3 * i1 + 2]);
        prim.n2 = glm::vec3(nbo[3 * i2], nbo[3 * i2 + 1], nbo[3 * i2 + 2]);

        prim.c0 = glm::vec3(cbo[3 * i0], cbo[3 * i0 + 1], cbo[3 * i0 + 2]);
        prim.c1 = glm::vec3(cbo[3 * i1], cbo[3 * i1 + 1], cbo[3 * i1 + 2]);
        prim.c2 = glm::vec3(cbo[3 * i2], cbo[3 * i2 + 1], cbo[3 * i2 + 2]);

        primitives[index] = prim;
    }
}

//TODO: Implement a better rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < primitivesCount) {
        triangle tri = primitives[index];
        glm::vec3 minp, maxp;
        getAABBForTriangle(tri, minp, maxp);
        glm::vec2 minc = (glm::vec2(minp) + 1.f) * 0.5f * resolution;
        glm::vec2 maxc = (glm::vec2(maxp) + 1.f) * 0.5f * resolution;

        for (int x = minc.x; x < maxc.x; ++x) {
            for (int y = minc.y; y < maxc.y; ++y) {
                glm::vec2 ndc = glm::vec2(
                        (x * 2 - resolution.x) / resolution.x,
                        (y * 2 - resolution.y) / resolution.y);
                glm::vec3 bary = calculateBarycentricCoordinate(tri, ndc);
                if (isBarycentricCoordInBounds(bary)) {
                    int i = (int) ((resolution.y - y) * resolution.x + x);
                    fragment frag = depthbuffer[i];
                    float depthold = frag.position.z;
                    float depthnew = getZAtCoordinate(bary, tri);
                    if (depthnew < 1 && depthnew > depthold) {
                        frag.color  = bary.x * tri.c0 + bary.y * tri.c1 + bary.z * tri.c2;
                        frag.normal = bary.x * tri.n0 + bary.y * tri.n1 + bary.z * tri.n2;
                        frag.position = glm::vec3(ndc, depthnew);
                        depthbuffer[i] = frag;
                    }
                }
            }
        }
    }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
    }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer)
{

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {
        fragment frag = depthbuffer[index];
        if (frag.position.z > -2.f) { // min should be -1
            framebuffer[index] = depthbuffer[index].color;
        }
    }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(
        uchar4* PBOpos, glm::vec2 resolution, float frame,
        float* vbo, int vbosize,
        float *nbo, int nbosize,
        float* cbo, int cbosize,
        int* ibo, int ibosize)
{
    // set up crucial magic
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    //set up framebuffer
    framebuffer = NULL;
    cudaMalloc((void**)&framebuffer, (int)resolution.x * (int)resolution.y * sizeof(glm::vec3));

    //set up depthbuffer
    depthbuffer = NULL;
    cudaMalloc((void**)&depthbuffer, (int)resolution.x * (int)resolution.y * sizeof(fragment));

    //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
    clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0.2f, 0.2f, 0.2f));

    fragment frag;
    frag.color = glm::vec3(1, 0, 1);
    frag.normal = glm::vec3(0, 0, 0);
    frag.position = glm::vec3(0, 0, -10000);
    clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, frag);

    //------------------------------
    //memory stuff
    //------------------------------
    primitives = NULL;
    cudaMalloc((void**)&primitives, (ibosize / 3)*sizeof(triangle));

    device_ibo = NULL;
    cudaMalloc((void**)&device_ibo, ibosize * sizeof(int));
    cudaMemcpy(device_ibo, ibo, ibosize * sizeof(int), cudaMemcpyHostToDevice);

    device_vbo = NULL;
    cudaMalloc((void**)&device_vbo, vbosize * sizeof(float));
    cudaMemcpy(device_vbo, vbo, vbosize * sizeof(float), cudaMemcpyHostToDevice);

    device_nbo = NULL;
    cudaMalloc((void**)&device_nbo, nbosize * sizeof(float));
    cudaMemcpy(device_nbo, nbo, nbosize * sizeof(float), cudaMemcpyHostToDevice);

    device_cbo = NULL;
    cudaMalloc((void**)&device_cbo, cbosize * sizeof(float));
    cudaMemcpy(device_cbo, cbo, cbosize * sizeof(float), cudaMemcpyHostToDevice);

    tileSize = 32;
    int primitiveBlocks = ceil(((float)vbosize / 3) / ((float)tileSize));

    //------------------------------
    //vertex shader
    //------------------------------
    vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize);

    cudaDeviceSynchronize();
    //------------------------------
    //primitive assembly
    //------------------------------
    primitiveBlocks = ceil(((float)ibosize / 3) / ((float)tileSize));
    primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(
            device_vbo, vbosize,
            device_nbo, nbosize,
            device_cbo, cbosize,
            device_ibo, ibosize,
            primitives);

    cudaDeviceSynchronize();
    //------------------------------
    //rasterization
    //------------------------------
    rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize / 3, depthbuffer, resolution);

    cudaDeviceSynchronize();
    //------------------------------
    //fragment shader
    //------------------------------
    fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);

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

void kernelCleanup()
{
    cudaFree(primitives);
    cudaFree(device_vbo);
    cudaFree(device_nbo);
    cudaFree(device_cbo);
    cudaFree(device_ibo);
    cudaFree(framebuffer);
    cudaFree(depthbuffer);
}

