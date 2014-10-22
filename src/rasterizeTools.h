// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"


struct triangle {
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;
	glm::vec3 c0;
	glm::vec3 c1;
	glm::vec3 c2;
};

struct fragment{
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec3 position;
};

//Multiplies a cudaMat4 matrix and a vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);

//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getAABBForTriangle(triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint);

//LOOK: calculates the signed area of a given triangle
__host__ __device__ float calculateSignedArea(triangle tri);

//LOOK: helper function for calculating barycentric coordinates
__host__ __device__ float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri);

//LOOK: calculates barycentric coordinates
__host__ __device__ glm::vec3 calculateBarycentricCoordinate(triangle tri, glm::vec2 point);

//LOOK: checks if a barycentric coordinate is within the boundaries of a triangle
__host__ __device__ bool isBarycentricCoordInBounds(glm::vec3 barycentricCoord);

//LOOK: for a given barycentric coordinate, return the corresponding z position on the triangle
__host__ __device__ float getZAtCoordinate(glm::vec3 barycentricCoord, triangle tri);

//perspective view matrix
__host__ __device__ cudaMat4 myFrustum(float left, float right, float bottom, float top, float near, float far);

struct Camera
{
	glm::vec3 pos;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
	glm::vec2 depth;

	glm::vec2 reso;
	//glm::vec3 ScreenV;
	//glm::vec3 ScreenH;
	//cudaMat4 Frustum;

	//Transform from world coordinates to clip coordinates;
	glm::mat4 VP_mat;
	__host__ __device__ Camera(
		glm::vec2 r = glm::vec2(0.0f, 0.0f),
		glm::vec3 p = glm::vec3(0.0f, 0.0f, 10.0f),
		glm::vec3 v = glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3 u = glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec2 f = glm::vec2(45.0f, 45.0f),
		glm::vec2 d = glm::vec2(1.0f, 1000.0f)
		) :
		pos(p),
		view(glm::normalize(v)),
		up(glm::normalize(u)),
		fov(f),
		depth(d),
		reso(r)
	{
		//MVP matrix is calculated as:			MVP = M_{Cam}^{Clip} * M_{world}^{Cam} * M_{model}^{world}
		//View-Clip matrix is calculated as:	VP = M_{Cam}^{Clip} * M_{world}^{Cam}
		
		glm::mat4 ViewMat;
		ViewMat[0] = glm::vec4(glm::normalize(glm::cross(view, up)), 0.0f);
		ViewMat[1] = glm::vec4(glm::normalize(up), 0.0f);
		ViewMat[2] = glm::vec4(glm::normalize(view), 0.0f);
		ViewMat[3] = glm::vec4(pos, 1.0f);
		ViewMat = glm::inverse(ViewMat);

		glm::vec3 viewport = depth.x*view*glm::vec3(tan(fov.x*PI/180.f),tan(fov.y*PI/180.0f),0.0f);
		glm::mat4 FrustumMat = utilityCore::cudaMat4ToGlmMat4(myFrustum(-viewport.x, viewport.x, -viewport.y, viewport.y, depth.x, depth.y));

		VP_mat = FrustumMat*ViewMat;
	}

};

#endif