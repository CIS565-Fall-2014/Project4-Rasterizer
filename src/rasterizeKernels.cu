// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <climits>
#include <cmath>
#include <thrust/random.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>

static const int INT_ABS_MAX = -INT_MIN;

using namespace utilityCore;

glm::vec3* framebuffer;
fragment* depthbuffer;
int *deptharray;
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
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, int *deptharray, fragment frag)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
        fragment f = frag;
        f.pn.x = screen2ndc(x, resolution.x);
        f.pn.y = screen2ndc(y, resolution.y);
        buffer[index] = f;
        deptharray[index] = f.pn.z * INT_ABS_MAX;
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
__global__ void vertexShadeKernel(vertU *vunifs, int vbocount,
        const float *pbo, const float *nbo, const float *cbo, vertO *vbo)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < vbocount) {
        vertU u = *vunifs;
        glm::vec4 p(pbo[index * 3], pbo[index * 3 + 1], pbo[index * 3 + 2], 1);
        glm::vec4 n(nbo[index * 3], nbo[index * 3 + 1], nbo[index * 3 + 2], 0);
        glm::vec3 c(cbo[index * 3], cbo[index * 3 + 1], cbo[index * 3 + 2]);

        vertO v;

        // Apply model
        glm::vec4 mp = multiplyMV(u.model, p);
        v.pw = glm::vec3(mp) / mp.w;
        glm::vec4 mn = multiplyMV(u.modelinvtr, n);
        v.nw = glm::vec3(mn);

        // Apply viewproj
        glm::vec4 pvp = multiplyMV(u.viewproj, mp);
        v.pn = glm::vec3(pvp) / pvp.w;

        v.c = c;
        vbo[index] = v;
    }
}

__global__ void primitiveAssemblyKernel(
        const vertO *vbo, int vbocount,
        const int* ibo, int tricount, int stride,
        triangle* primitives)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < tricount) {
        triangle prim;
        prim.discard = false;
        prim.v[0] = vbo[ibo[3 * index + 0]];
        prim.v[1] = vbo[ibo[3 * index + 1]];
        prim.v[2] = vbo[ibo[3 * index + 2]];
        primitives[index * stride] = prim;
    }
}

__global__ void geometryShadeKernel(geomU *gunifs,
        triangle *prims, int tricount, int stride)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < tricount) {
        triangle prim = prims[index * stride];
        vertO v0 = prim.v[0];
        vertO v1 = prim.v[1];
        vertO v2 = prim.v[2];

        // Backface culling
        glm::vec3 winding = glm::cross(v1.pn - v0.pn, v2.pn - v1.pn);
        if (winding.z > 0) {
            prims[index * stride].discard = true;
            return;
        }

#if 1
        // Tessellation
        vertO v01;
        v01.pn = (v0.pn + v1.pn) * .5f;
        v01.pw = (v0.pw + v1.pw) * .5f;
        v01.nw = (v0.nw + v1.nw) * .5f;
        v01.c  = (v0.c  + v1.c ) * .5f;
        vertO v12;
        v12.pn = (v1.pn + v2.pn) * .5f;
        v12.pw = (v1.pw + v2.pw) * .5f;
        v12.nw = (v1.nw + v2.nw) * .5f;
        v12.c  = (v1.c  + v2.c ) * .5f;
        vertO v20;
        v20.pn = (v2.pn + v0.pn) * .5f;
        v20.pw = (v2.pw + v0.pw) * .5f;
        v20.nw = (v2.nw + v0.nw) * .5f;
        v20.c  = (v2.c  + v0.c ) * .5f;

        if (stride >= 4) {
            prim.v[0] = v0;
            prim.v[1] = v01;
            prim.v[2] = v20;
            prims[index * stride + 0] = prim;

            prim.v[0] = v1;
            prim.v[1] = v12;
            prim.v[2] = v01;
            prims[index * stride + 1] = prim;

            prim.v[0] = v2;
            prim.v[1] = v20;
            prim.v[2] = v12;
            prims[index * stride + 2] = prim;

            // this one is red for visualization
            prim.v[0] = v01;
            prim.v[1] = v12;
            prim.v[2] = v20;
            prim.v[0].c = prim.v[1].c = prim.v[2].c = glm::vec3(1, 0, 0);
            prims[index * stride + 3] = prim;
        }
#endif
    }
}

//TODO: Implement a better rasterization method?
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, int *deptharray, glm::vec2 resolution)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < primitivesCount) {
        triangle tri = primitives[index];

        // Find the AABB of the tri on the screen
        glm::vec3 minp, maxp;
        getAABBForTriangle(tri, minp, maxp);
        glm::vec2 minc = glm::max(glm::vec2(), (glm::vec2(minp) + 1.f) * 0.5f * resolution);
        glm::vec2 maxc = glm::min(resolution , (glm::vec2(maxp) + 1.f) * 0.5f * resolution);

        // Depth-buffer-ize pixels within the triangle
        for (int x = minc.x; x < maxc.x; ++x) {
            for (int y = minc.y; y < maxc.y; ++y) {
                glm::vec2 ndc = glm::vec2(
                        (x * 2 - resolution.x) / resolution.x,
                        (y * 2 - resolution.y) / resolution.y);
                glm::vec3 bary = calculateBarycentricCoordinate(tri, ndc);
                if (isBarycentricCoordInBounds(bary)) {
                    int i = (int) ((resolution.y - y - 1) * resolution.x + x);

                    float depthnew = getZAtCoordinate(bary, tri);
                    fragment frag;
                    frag.pn = glm::vec3(ndc, depthnew);
                    frag.pw = baryinterp(bary, tri.v[0].pw, tri.v[1].pw, tri.v[2].pw);
                    frag.c  = baryinterp(bary, tri.v[0].c , tri.v[1].c , tri.v[2].c );
                    frag.nw = baryinterp(bary, tri.v[0].nw, tri.v[1].nw, tri.v[2].nw);

                    int mydepth = (int) (depthnew * INT_ABS_MAX);
                    atomicMin(&deptharray[i], mydepth);
                    if (deptharray[i] == mydepth) {
                        depthbuffer[i] = frag;
                    }
                }
            }
        }
    }
}

__global__ void fragmentShadeKernel(fragU *funifs,
        fragment* depthbuffer, glm::vec2 resolution)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
        fragU u = *funifs;
        fragment frag = depthbuffer[index];

        // Render depth
        //frag.c = ndc2norm(glm::vec3(frag.pn.z));

        // Render normals
        //frag.c = ndc2norm(frag.nw);

        // Render world position
        //frag.c = ndc2norm(frag.pw);

        // Diffuse
        glm::vec3 lightdir = glm::normalize(u.lightpos - frag.pw);
        float coeff = glm::max(0.f, glm::dot(frag.nw, lightdir));
        frag.c = (u.ambcol + coeff * u.lightcol) * frag.c;

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
        framebuffer[index] = depthbuffer[index].c;
    }
}

struct is_discard {
    __host__ __device__ bool operator()(const triangle tri)
    {
        return tri.discard;
    }
};

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

    deptharray = NULL;
    cudaMalloc((void**)&deptharray, (int)resolution.x * (int)resolution.y * sizeof(fragment));

    //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
    clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(1, 0, 1));

    fragment frag;
    frag.pn = glm::vec3(0, 0, 1);
    frag.pw = glm::vec3(0, 0, 0);
    frag.c  = glm::vec3(0.2f, 0.2f, 0.2f);
    frag.nw = glm::vec3(0, 0, 0);
    clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, deptharray, frag);

    // CONFIG: this is the max number of output tris per input tri
    int tricount = ibosize / 3;
    int geomstride = 4;

    //------------------------------
    //memory stuff
    //------------------------------
    primitives = NULL;
    cudaMalloc((void**)&primitives, tricount * geomstride * sizeof(triangle));
    triangle defaulttri;
    defaulttri.discard = true;
    thrust::fill(thrust::device, primitives, &primitives[tricount], defaulttri);

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

    tileSize = 256;
    int primitiveBlocks = ceil(((float) vbocount) / ((float)tileSize));

    //------------------------------
    //vertex shader
    //------------------------------
    vertU *device_vunifs;
    cudaMalloc((void **) &device_vunifs, sizeof(vertU));
    fragU *device_funifs;
    cudaMalloc((void **) &device_funifs, sizeof(fragU));
    geomU *device_gunifs;
    cudaMalloc((void **) &device_gunifs, sizeof(geomU));
    {
        float fovy = glm::radians(30.f);
        float aspect = resolution.x / resolution.y;
        glm::vec3 eye(1.5f, 1, 2);
        glm::vec3 center(0, 0, 0);
        glm::vec3 up(0, 1, 0);

        glm::mat4 model;
        //glm::mat4 model = glm::rotate(glm::mat4(), glm::radians(30.f), glm::vec3(1, 1, 0));
        glm::mat4 modelinvtr = glm::inverse(glm::transpose(model));
        glm::mat4 view = glm::lookAt(eye, center, up);
        glm::mat4 proj = glm::perspective(fovy, aspect, 0.1f, 100.f);
        glm::mat4 viewproj = proj * view;

        vertU vunifs;
        vunifs.model = glmMat4ToCudaMat4(model);
        vunifs.modelinvtr = glmMat4ToCudaMat4(modelinvtr);
        vunifs.viewproj = glmMat4ToCudaMat4(viewproj);
        cudaMemcpy(device_vunifs, &vunifs, sizeof(vertU), cudaMemcpyHostToDevice);

        fragU funifs;
        funifs.eye = eye;
        funifs.lightpos = glm::vec3(8, 4, -5);
        funifs.lightcol = glm::vec3(1.0, 1.0, 1.0);
        funifs.ambcol = glm::vec3(0.1, 0.1, 0.1);
        cudaMemcpy(device_funifs, &funifs, sizeof(fragU), cudaMemcpyHostToDevice);

        geomU gunifs;
        gunifs.x = 0;
        cudaMemcpy(device_gunifs, &gunifs, sizeof(geomU), cudaMemcpyHostToDevice);
    }

    device_vbo = NULL;
    cudaMalloc((void **) &device_vbo, vbocount * sizeof(vertO));
    vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vunifs, vbocount,
            device_pbo, device_nbo, device_cbo, device_vbo);

    cudaDeviceSynchronize();
    //------------------------------
    //primitive assembly
    //------------------------------
    primitiveBlocks = ceil(((float)tricount) / ((float)tileSize));
    primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(
            device_vbo, vbocount,
            device_ibo, tricount, geomstride,
            primitives);

    cudaDeviceSynchronize();
    //------------------------------
    // Geometry shader
    //------------------------------
    primitiveBlocks = ceil(((float)tricount) / ((float)tileSize));
    //geometryShadeKernel<<<primitiveBlocks, tileSize>>>(
    //        device_gunifs, primitives, tricount, geomstride);

#if 0
    triangle *lastprim = thrust::remove_if(thrust::device,
            primitives, &primitives[tricount * geomstride], is_discard());
    tricount = lastprim - primitives;
#endif

    cudaDeviceSynchronize();
    //------------------------------
    //rasterization
    //------------------------------
    primitiveBlocks = ceil(((float)tricount) / ((float)tileSize));
    rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, tricount, depthbuffer, deptharray, resolution);

    cudaDeviceSynchronize();
    //------------------------------
    //fragment shader
    //------------------------------
    fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(device_funifs,
            depthbuffer, resolution);

    cudaDeviceSynchronize();
    //------------------------------
    //write fragments to framebuffer
    //------------------------------
    render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
    sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

    cudaDeviceSynchronize();

    cudaFree(device_vunifs);
    cudaFree(device_funifs);
    cudaFree(device_gunifs);
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
    cudaFree(deptharray);
}

