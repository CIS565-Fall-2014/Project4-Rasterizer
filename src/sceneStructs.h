#pragma once

#ifndef _SCENE_STRUCTS
#define _SCENE_STRUCTS

#include "glm/glm.hpp"
#include "cudaMat4.h"

struct simpleCamera
{
	glm::vec3 position;
	glm::vec3 target;
	glm::vec3 up;
	float fov_y;
	glm::vec2 resolution;
	float near_clip;
	float far_clip;
	//glm::vec3 translation;
	//glm::vec3 rotation;
	//cudaMat4 transform;
};

struct edge
{
	__host__ __device__ void setEdge( glm::vec3 vb, glm::vec3 vt, float y )
	{
		dxdy = ( vt.x - vb.x ) / ( vt.y - vb.y );
		x = vb.x + ( y - vb.y ) * dxdy;
	}

	float dxdy;
	float x;
};

#endif