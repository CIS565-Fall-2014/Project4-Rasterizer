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
float* device_pbo;
float* device_nbo;
float* device_cbo;
int* device_ibo;
vertO *device_vbo;
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
        f.pn.x = screen2ndc(x, resolution.x);
        f.pn.y = screen2ndc(y, resolution.y);
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
__global__ void vertexShadeKernel(int vbocount,
        const float *pbo, const float *nbo, const float *cbo, vertO *vbo)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vbocount) {
        glm::vec3 p(pbo[index * 3], pbo[index * 3 + 1], pbo[index * 3 + 2]);
        glm::vec3 n(nbo[index * 3], nbo[index * 3 + 1], nbo[index * 3 + 2]);
        glm::vec3 c(cbo[index * 3], cbo[index * 3 + 1], cbo[index * 3 + 2]);

        vertO v;

        // Apply modelview
        //p = glm::rotate(p, 1.f, glm::vec3(0, 1, 0));
        //p = glm::rotate(p, 1.f, glm::vec3(1, 0, 0));

        v.pw = p;

        // Apply projection

        v.pn = p;
        v.nw = n;
        v.c = c;
        vbo[index] = v;
    }
}

__global__ void primitiveAssemblyKernel(
        const vertO *vbo, int vbocount,
        const int* ibo, int ibosize,
        triangle* primitives)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int primitivesCount = ibosize / 3;
    if (index < primitivesCount) {
        triangle prim;
        prim.v[0] = vbo[ibo[3 * index + 0]];
        prim.v[1] = vbo[ibo[3 * index + 1]];
        prim.v[2] = vbo[ibo[3 * index + 2]];
        primitives[index] = prim;
    }
}

//TODO: Implement a better rasterization method?
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < primitivesCount) {
        triangle tri = primitives[index];

        // Backface culling
        glm::vec3 winding = glm::cross(
                tri.v[1].pn - tri.v[0].pn,
                tri.v[2].pn - tri.v[1].pn);
        if (winding.z > 0) {
            return;
        }

        // Find the AABB of the tri on the screen
        glm::vec3 minp, maxp;
        getAABBForTriangle(tri, minp, maxp);
        glm::vec2 minc = (glm::vec2(minp) + 1.f) * 0.5f * resolution;
        glm::vec2 maxc = (glm::vec2(maxp) + 1.f) * 0.5f * resolution;

        // Depth-buffer-ize pixels within the triangle
        for (int x = minc.x; x < maxc.x; ++x) {
            for (int y = minc.y; y < maxc.y; ++y) {
                glm::vec2 ndc = glm::vec2(
                        (x * 2 - resolution.x) / resolution.x,
                        (y * 2 - resolution.y) / resolution.y);
                glm::vec3 bary = calculateBarycentricCoordinate(tri, ndc);
                if (isBarycentricCoordInBounds(bary)) {
                    int i = (int) ((resolution.y - y) * resolution.x + x);
                    fragment frag = depthbuffer[i];
                    float depthold = frag.pn.z;
                    float depthnew = getZAtCoordinate(bary, tri);

                    if (depthnew < 1 && depthnew > depthold) {
                        frag.pn = glm::vec3(ndc, depthnew);
                        frag.pw = baryinterp(bary, tri.v[0].pw, tri.v[1].pw, tri.v[2].pw);
                        frag.c  = baryinterp(bary, tri.v[0].c , tri.v[1].c , tri.v[2].c );
                        frag.nw = baryinterp(bary, tri.v[0].nw, tri.v[1].nw, tri.v[2].nw);
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
        fragment frag = depthbuffer[index];

        // Render depth
        //frag.c = ndc2norm(glm::vec3(frag.pn.z));

        // Render normals
        //frag.c = ndc2norm(frag.nw);

        // Render world position
        frag.c = ndc2norm(frag.pw);

        depthbuffer[index] = frag;
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
        if (frag.nw.z > -1.0001f) { // min should be -1
            framebuffer[index] = depthbuffer[index].c;
        }
    }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(
        uchar4* PBOpos, glm::vec2 resolution, float frame,
        int vbosize, float *pbo, float *nbo, float *cbo,
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
    frag.pn = glm::vec3(0, 0, -10000);
    frag.pw = glm::vec3(0, 0, 0);
    frag.c  = glm::vec3(1, 0, 1);
    frag.nw = glm::vec3(0, 0, 0);
    clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, frag);

    //------------------------------
    //memory stuff
    //------------------------------
    primitives = NULL;
    cudaMalloc((void**)&primitives, (ibosize / 3)*sizeof(triangle));

    device_ibo = NULL;
    cudaMalloc((void**)&device_ibo, ibosize * sizeof(int));
    cudaMemcpy(device_ibo, ibo, ibosize * sizeof(int), cudaMemcpyHostToDevice);

    device_pbo = NULL;
    cudaMalloc((void**)&device_pbo, vbosize * sizeof(float));
    cudaMemcpy(device_pbo, pbo, vbosize * sizeof(float), cudaMemcpyHostToDevice);

    device_nbo = NULL;
    cudaMalloc((void**)&device_nbo, vbosize * sizeof(float));
    cudaMemcpy(device_nbo, nbo, vbosize * sizeof(float), cudaMemcpyHostToDevice);

    device_cbo = NULL;
    cudaMalloc((void**)&device_cbo, vbosize * sizeof(float));
    cudaMemcpy(device_cbo, cbo, vbosize * sizeof(float), cudaMemcpyHostToDevice);

    int vbocount = vbosize / 3;

    tileSize = 32;
    int primitiveBlocks = ceil(((float) vbocount) / ((float)tileSize));

    //------------------------------
    //vertex shader
    //------------------------------
    device_vbo = NULL;
    cudaMalloc((void **) &device_vbo, vbocount * sizeof(vertO));
    vertexShadeKernel<<<primitiveBlocks, tileSize>>>(vbocount,
            device_pbo, device_nbo, device_cbo, device_vbo);

    cudaDeviceSynchronize();
    //------------------------------
    //primitive assembly
    //------------------------------
    primitiveBlocks = ceil(((float)ibosize / 3) / ((float)tileSize));
    primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(
            device_vbo, vbocount,
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
    cudaFree(device_pbo);
    cudaFree(device_nbo);
    cudaFree(device_cbo);
    cudaFree(device_ibo);
    cudaFree(device_vbo);
    cudaFree(framebuffer);
    cudaFree(depthbuffer);
}

