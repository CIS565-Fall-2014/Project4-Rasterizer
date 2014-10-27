// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"


struct vertU {
    cudaMat4 model;
    cudaMat4 modelinvtr;
    cudaMat4 viewproj;
};

struct fragU {
    glm::vec3 eye;
    glm::vec3 lightpos;
    glm::vec3 lightcol;
    glm::vec3 ambcol;
};

struct vertI {
    glm::vec3 pw;  // pos world
    glm::vec3 nw;  // nor world
    glm::vec3 c;   // color
};

struct vertO {
    glm::vec3 pn;  // pos ndc
    glm::vec3 pw;  // pos world
    glm::vec3 nw;  // nor world
    glm::vec3 c;   // color
};

struct triangle {
    vertO v[3];
};

struct fragment {
    glm::vec3 pn;  // pos ndc
    glm::vec3 pw;  // pos world
    glm::vec3 c;   // color
    glm::vec3 nw;  // nor world
};


__host__ __device__ float screen2ndc(float x, float res)
{
    return (x / res) * 2.f - 1.f;
}

__host__ __device__ glm::vec3 ndc2norm(glm::vec3 x)
{
    return (x + glm::vec3(1.f)) * 0.5f;
}

__host__ __device__ glm::vec3 baryinterp(
        glm::vec3 bary, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
{
    return bary.x * v0 + bary.y * v1 + bary.z * v2;
}

//Multiplies a cudaMat4 matrix and a vec4
__host__ __device__ glm::vec4 multiplyMV(cudaMat4 m, glm::vec4 v)
{
    glm::vec4 r;
    r.x = (m.x.x * v.x) + (m.x.y * v.y) + (m.x.z * v.z) + (m.x.w * v.w);
    r.y = (m.y.x * v.x) + (m.y.y * v.y) + (m.y.z * v.z) + (m.y.w * v.w);
    r.z = (m.z.x * v.x) + (m.z.y * v.y) + (m.z.z * v.z) + (m.z.w * v.w);
    r.w = (m.w.x * v.x) + (m.w.y * v.y) + (m.w.z * v.z) + (m.w.w * v.w);
    return r;
}

//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getAABBForTriangle(
        triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint)
{
    minpoint = glm::min(glm::min(tri.v[0].pn, tri.v[1].pn), tri.v[2].pn);
    maxpoint = glm::max(glm::max(tri.v[0].pn, tri.v[1].pn), tri.v[2].pn);
}

//LOOK: calculates the signed area of a given triangle
__host__ __device__ float calculateSignedArea(triangle tri)
{
    return 0.5 *
        ((tri.v[2].pn.x - tri.v[0].pn.x) * (tri.v[1].pn.y - tri.v[0].pn.y) -
         (tri.v[1].pn.x - tri.v[0].pn.x) * (tri.v[2].pn.y - tri.v[0].pn.y));
}

//LOOK: helper function for calculating barycentric coordinates
__host__ __device__ float calculateBarycentricCoordinateValue(
        glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri)
{
    triangle baryTri;
    baryTri.v[0].pn = glm::vec3(a, 0);
    baryTri.v[1].pn = glm::vec3(b, 0);
    baryTri.v[2].pn = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

//LOOK: calculates barycentric coordinates
__host__ __device__ glm::vec3 calculateBarycentricCoordinate(
        triangle tri, glm::vec2 point)
{
    float beta  = calculateBarycentricCoordinateValue(
            glm::vec2(tri.v[0].pn.x, tri.v[0].pn.y),
            point,
            glm::vec2(tri.v[2].pn.x, tri.v[2].pn.y),
            tri);
    float gamma = calculateBarycentricCoordinateValue(
            glm::vec2(tri.v[0].pn.x, tri.v[0].pn.y),
            glm::vec2(tri.v[1].pn.x, tri.v[1].pn.y),
            point,
            tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

//LOOK: checks if a barycentric coordinate is within the boundaries of a triangle
__host__ __device__ bool isBarycentricCoordInBounds(glm::vec3 barycentricCoord)
{
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

//LOOK: for a given barycentric coordinate, return the corresponding z position on the triangle
__host__ __device__ float getZAtCoordinate(glm::vec3 barycentricCoord, triangle tri)
{
    return -(barycentricCoord.x * tri.v[0].pn.z +
             barycentricCoord.y * tri.v[1].pn.z +
             barycentricCoord.z * tri.v[2].pn.z);
}

#endif
